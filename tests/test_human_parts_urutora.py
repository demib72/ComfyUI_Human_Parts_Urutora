from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "human_parts_urutora_golden.json"
SCHEMA_FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "v1_node_schemas.json"
sys.path.insert(0, str(PROJECT_PARENT))
# Custom nodes normally run inside ComfyUI. Make its V3 API importable when the
# test checkout is installed beside ComfyUI, matching the documented layout.
COMFYUI_ROOT = PROJECT_PARENT / "ComfyUI"
if COMFYUI_ROOT.is_dir():
    sys.path.insert(0, str(COMFYUI_ROOT))

from ComfyUI_Human_Parts_Urutora import comfy_entrypoint
from ComfyUI_Human_Parts_Urutora.comfy_compat import HAS_COMFY_V3
from ComfyUI_Human_Parts_Urutora.nodes import HumanPartsUrutoraMaskGenerator
from ComfyUI_Human_Parts_Urutora.detector.human_parts import get_mask
from ComfyUI_Human_Parts_Urutora.detector import matting
from ComfyUI_Human_Parts_Urutora.detector.face_parsing import (
    FACE_PARSING_CLASSES,
    segment_face_parts,
)
from ComfyUI_Human_Parts_Urutora.nodes_urutora import (
    URUTORA_ANATOMICAL_PARTS,
    URUTORA_CLASSES,
    HumanPartsUrutora,
    LayerStyleHumanPartsUrutora,
    _execution_providers,
    lifecycle_execution_providers,
    _load_session,
    _segment_parts,
)
from ComfyUI_Human_Parts_Urutora.utils import _fallback_models_dir


class GoldenSession:
    def __init__(self, class_map: np.ndarray):
        self.class_map = class_map

    def get_inputs(self):
        return [SimpleNamespace(name="image")]

    def get_outputs(self):
        return [SimpleNamespace(name="segmentation")]

    def run(self, output_names, inputs):
        height, width = self.class_map.shape
        logits = np.zeros((1, height, width, 22), dtype=np.float32)
        for row in range(height):
            for column in range(width):
                logits[0, row, column, self.class_map[row, column]] = 1.0
        return [logits]


class RecordingSession(GoldenSession):
    def __init__(self, class_map: np.ndarray):
        super().__init__(class_map)
        self.inputs = []

    def run(self, output_names, inputs):
        self.inputs.append(next(iter(inputs.values())))
        return super().run(output_names, inputs)


class FaceParsingSession:
    def __init__(self, class_map: np.ndarray):
        self.class_map = class_map
        self.inputs = []

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, output_names, inputs):
        self.inputs.append(next(iter(inputs.values())))
        height, width = self.class_map.shape
        logits = np.zeros((1, 19, height, width), dtype=np.float32)
        for class_index in range(19):
            logits[0, class_index][self.class_map == class_index] = 1.0
        return [logits]


def _load_golden_fixture() -> dict:
    with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


class ModelDirectoryDiscoveryTests(unittest.TestCase):
    def test_checkout_beside_comfyui_uses_sibling_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plugin = root / "Human_Parts_Urutora"
            models = root / "ComfyUI" / "models"
            plugin.mkdir()
            models.mkdir(parents=True)

            self.assertEqual(_fallback_models_dir(plugin), str(models))

    def test_custom_nodes_install_uses_comfyui_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ComfyUI"
            plugin = root / "custom_nodes" / "Human_Parts_Urutora"
            models = root / "models"
            plugin.mkdir(parents=True)
            models.mkdir()

            self.assertEqual(_fallback_models_dir(plugin), str(models))


def _node_arguments(image: torch.Tensor, **overrides) -> dict:
    arguments = {
        "image": image,
        **{name: False for name in URUTORA_CLASSES},
        **{name: False for name in URUTORA_ANATOMICAL_PARTS},
        "detail_method": "GuidedFilter",
        "detail_erode": 8,
        "detail_dilate": 6,
        "black_point": 0.01,
        "white_point": 0.99,
        "process_detail": False,
        "device": "cpu",
        "max_megapixels": 2.0,
    }
    arguments.update(overrides)
    return arguments


class HumanPartsUrutoraMaskGeneratorTests(unittest.TestCase):
    def test_batch_returns_standard_float32_comfyui_masks(self):
        class_map = np.asarray([[13, 0], [0, 13]], dtype=np.int64)
        session = RecordingSession(class_map)
        first = torch.zeros((3, 5, 3), dtype=torch.float32)
        second = torch.ones((3, 5, 3), dtype=torch.float32)

        mask, score = get_mask(
            torch.stack((first, second)), session, rotation=0, face=True
        )

        self.assertEqual(mask.shape, (2, 3, 5))
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(set(mask.unique().tolist()), {0.0, 1.0})
        self.assertEqual(len(session.inputs), 2)
        self.assertTrue(
            all(
                value.shape == (1, 512, 512, 3)
                for value in session.inputs
            )
        )
        self.assertGreater(score, 0)

    def test_execute_uses_the_shared_session_loader(self):
        image = torch.zeros((1, 3, 5, 3), dtype=torch.float32)
        session = GoldenSession(np.zeros((2, 2), dtype=np.int64))

        with (
            patch(
                "ComfyUI_Human_Parts_Urutora.nodes.execution_providers",
                return_value=("CPUExecutionProvider",),
            ),
            patch(
                "ComfyUI_Human_Parts_Urutora.nodes.load_session", return_value=session
            ) as load,
        ):
            first = HumanPartsUrutoraMaskGenerator.execute(image, face=True).result[0]
            second = HumanPartsUrutoraMaskGenerator.execute(image, face=True).result[0]

        self.assertEqual(first.shape, (1, 3, 5))
        self.assertEqual(second.dtype, torch.float32)
        self.assertEqual(load.call_count, 2)
        self.assertEqual(load.call_args_list[0], load.call_args_list[1])


class HumanPartsUrutoraGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = _load_golden_fixture()
        cls.rgb = np.asarray(cls.golden["rgb"], dtype=np.uint8)
        cls.image = Image.fromarray(cls.rgb, mode="RGB")
        cls.image_tensor = torch.from_numpy(
            cls.rgb.astype(np.float32) / 255.0
        ).unsqueeze(0)
        cls.session = GoldenSession(
            np.asarray(cls.golden["class_map"], dtype=np.int64)
        )

    def assertGoldenMask(self, actual: torch.Tensor, expected) -> None:
        expected_tensor = torch.tensor([expected], dtype=torch.float32)
        self.assertTrue(
            torch.equal(actual, expected_tensor),
            f"\nactual:\n{actual}\nexpected:\n{expected_tensor}",
        )

    def test_every_part_matches_the_pre_modernization_golden_output(self):
        for part_name, expected in self.golden["parts"].items():
            with self.subTest(part=part_name):
                actual = _segment_parts(
                    self.image, self.session, {part_name: True}
                )
                self.assertGoldenMask(actual, expected)

    def test_empty_all_and_combined_selections_match_golden_outputs(self):
        selections = {
            "face_hair_glasses": {"face": True, "hair": True, "glasses": True},
            "left_side": {"left_arm": True, "left_leg": True, "left_foot": True},
            "empty": {},
            "all": {name: True for name in URUTORA_CLASSES},
        }
        for combination, selected in selections.items():
            with self.subTest(combination=combination):
                actual = _segment_parts(self.image, self.session, selected)
                self.assertGoldenMask(
                    actual, self.golden["combinations"][combination]
                )

    def test_left_and_right_foot_remain_independent(self):
        left = _segment_parts(self.image, self.session, {"left_foot": True})
        right = _segment_parts(self.image, self.session, {"right_foot": True})

        self.assertGoldenMask(left, self.golden["parts"]["left_foot"])
        self.assertGoldenMask(right, self.golden["parts"]["right_foot"])
        self.assertFalse(torch.equal(left, right))

    def test_multiple_image_batches_and_rgba_alpha_match_mask_exactly(self):
        image_batch = torch.cat((self.image_tensor, self.image_tensor), dim=0)
        with patch(
            "ComfyUI_Human_Parts_Urutora.nodes_urutora._load_session",
            return_value=self.session,
        ):
            rgba, mask = HumanPartsUrutora.execute(
                **_node_arguments(image_batch, face=True, left_foot=True)
            ).result

        expected = np.maximum(
            np.asarray(self.golden["parts"]["face"]),
            np.asarray(self.golden["parts"]["left_foot"]),
        )
        expected_batch = torch.tensor(
            np.stack((expected, expected)), dtype=torch.float32
        )
        self.assertEqual(rgba.shape, (2, 4, 4, 4))
        self.assertEqual(mask.shape, (2, 4, 4))
        self.assertEqual(rgba.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertTrue(torch.equal(mask, expected_batch))
        self.assertTrue(torch.equal(rgba[..., 3], mask))
        self.assertTrue(torch.equal(rgba[..., :3], image_batch))

    def test_golden_mask_is_not_brightness_enhanced(self):
        mask = _segment_parts(self.image, self.session, {"face": True})
        self.assertEqual(set(mask.unique().tolist()), {0.0, 1.0})
        self.assertGoldenMask(mask, self.golden["parts"]["face"])

    def test_anatomical_masks_are_nonempty_and_stay_in_source_classes(self):
        class_map = np.zeros((100, 80), dtype=np.int64)
        class_map[5:30, 20:60] = URUTORA_CLASSES["face"]
        class_map[30:70, 15:65] = URUTORA_CLASSES["torso_skin"]
        class_map[70:100, 20:38] = URUTORA_CLASSES["left_leg"]
        class_map[70:100, 42:60] = URUTORA_CLASSES["right_leg"]
        image = Image.new("RGB", (80, 100), "white")
        session = GoldenSession(class_map)

        allowed_classes = {
            "eyes": {URUTORA_CLASSES["face"]},
            "breasts": {URUTORA_CLASSES["torso_skin"]},
            "groin": {
                URUTORA_CLASSES["torso_skin"],
                URUTORA_CLASSES["left_leg"],
                URUTORA_CLASSES["right_leg"],
            },
        }
        for part_name in URUTORA_ANATOMICAL_PARTS:
            with self.subTest(part=part_name):
                mask = _segment_parts(image, session, {part_name: True})
                selected = mask.squeeze(0).numpy().astype(bool)
                self.assertTrue(selected.any())
                self.assertTrue(
                    np.isin(class_map[selected], list(allowed_classes[part_name])).all()
                )

    def test_eye_selection_uses_native_face_parser_classes_when_available(self):
        coarse_map = np.zeros((8, 8), dtype=np.int64)
        coarse_map[2:6, 2:6] = URUTORA_CLASSES["face"]
        face_map = np.zeros((8, 8), dtype=np.int64)
        face_map[3, 2:4] = FACE_PARSING_CLASSES["left_eye"]
        face_map[3, 5:7] = FACE_PARSING_CLASSES["right_eye"]
        coarse_session = GoldenSession(coarse_map)
        face_session = FaceParsingSession(face_map)
        image = Image.new("RGB", (80, 80), "white")

        mask = _segment_parts(
            image,
            coarse_session,
            {"eyes": True},
            face_session,
        )

        self.assertGreater(mask.sum().item(), 0)
        self.assertEqual(face_session.inputs[0].shape, (1, 3, 512, 512))
        self.assertEqual(face_session.inputs[0].dtype, np.float32)


class FaceParsingTests(unittest.TestCase):
    def test_cci_hp_face_components_are_parsed_as_independent_crops(self):
        coarse_map = np.zeros((20, 20), dtype=np.int64)
        coarse_map[2:7, 2:7] = URUTORA_CLASSES["face"]
        coarse_map[12:18, 13:19] = URUTORA_CLASSES["face"]
        face_map = np.zeros((16, 16), dtype=np.int64)
        face_map[7:9, 4:7] = FACE_PARSING_CLASSES["left_eye"]
        face_map[7:9, 10:13] = FACE_PARSING_CLASSES["right_eye"]
        session = FaceParsingSession(face_map)

        result = segment_face_parts(
            Image.new("RGB", (200, 100), "white"),
            coarse_map,
            URUTORA_CLASSES["face"],
            session,
            (
                FACE_PARSING_CLASSES["left_eye"],
                FACE_PARSING_CLASSES["right_eye"],
            ),
        )

        self.assertEqual(result.shape, (100, 200))
        self.assertEqual(len(session.inputs), 2)
        self.assertTrue((result[:, :100] > 0).any())
        self.assertTrue((result[:, 100:] > 0).any())


class HumanPartsUrutoraWorkflowCompatibilityTests(unittest.TestCase):
    EXPECTED_INPUT_ORDER = [
        "image",
        "face",
        "eyes",
        "hair",
        "glasses",
        "top_clothes",
        "bottom_clothes",
        "torso_skin",
        "breasts",
        "groin",
        "left_arm",
        "right_arm",
        "left_leg",
        "right_leg",
        "left_foot",
        "right_foot",
        "detail_method",
        "detail_erode",
        "detail_dilate",
        "black_point",
        "white_point",
        "process_detail",
        "device",
        "max_megapixels",
    ]

    def test_urutora_workflow_identifiers_are_registered(self):
        import asyncio

        extension = asyncio.run(comfy_entrypoint())
        node_classes = asyncio.run(extension.get_node_list())
        node_ids = [node.define_schema().node_id for node in node_classes]
        self.assertEqual(
            node_ids,
            ["HumanPartsUrutoraMaskGenerator", "LayerMask: HumanPartsUrutora", "HumanPartsUrutora"],
        )

    def test_widget_order_matches_layerstyle_workflows(self):
        schema = HumanPartsUrutora.define_schema()
        input_ids = [item.id for item in schema.inputs]
        self.assertEqual(input_ids, self.EXPECTED_INPUT_ORDER)
        self.assertEqual(input_ids.index("eyes"), input_ids.index("face") + 1)
        self.assertEqual(
            input_ids[input_ids.index("torso_skin") + 1 : input_ids.index("torso_skin") + 3],
            ["breasts", "groin"],
        )
        device = next(item for item in schema.inputs if item.id == "device")
        self.assertEqual(device.options, ["cuda", "cpu", "auto"])
        self.assertEqual(device.default, "auto")
        max_megapixels = next(
            item for item in schema.inputs if item.id == "max_megapixels"
        )
        self.assertEqual(max_megapixels.default, 2.0)
        self.assertEqual(max_megapixels.min, 1.0)
        groin = next(item for item in schema.inputs if item.id == "groin")
        self.assertEqual(groin.display_name, "female groin")

    def test_v3_schemas_match_captured_v1_compatibility_fixture(self):
        with SCHEMA_FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
            fixtures = json.load(fixture_file)

        classes = [HumanPartsUrutoraMaskGenerator, LayerStyleHumanPartsUrutora, HumanPartsUrutora]
        actual = {}
        for node_class in classes:
            schema = node_class.define_schema()
            actual[schema.node_id] = {
                "display_name": schema.display_name,
                "category": schema.category,
                "inputs": [item.id for item in schema.inputs],
                "input_types": [item.io_type for item in schema.inputs],
                "outputs": [item.io_type for item in schema.outputs],
                "output_names": [item.id for item in schema.outputs],
            }
        self.assertEqual(actual, fixtures)

    def test_v3_schemas_pass_comfyui_validation(self):
        for node_class in [HumanPartsUrutoraMaskGenerator, LayerStyleHumanPartsUrutora, HumanPartsUrutora]:
            with self.subTest(node=node_class.__name__):
                self.assertEqual(
                    node_class.GET_SCHEMA().node_id,
                    node_class.define_schema().node_id,
                )

    @unittest.skipUnless(HAS_COMFY_V3, "requires ComfyUI's V3 node API")
    def test_v3_input_types_use_hashable_combo_markers(self):
        """V3 validation hashes input types while following connected nodes."""
        required = HumanPartsUrutora.INPUT_TYPES()["required"]

        self.assertEqual(required["detail_method"][0], "COMBO")
        self.assertEqual(required["device"][0], "COMBO")
        for input_type, *_ in required.values():
            hash(input_type)

    def test_current_positional_widget_values_select_right_foot(self):
        golden = _load_golden_fixture()
        rgb = np.asarray(golden["rgb"], dtype=np.uint8)
        image = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)
        session = GoldenSession(np.asarray(golden["class_map"], dtype=np.int64))
        # Widget values are stored positionally in legacy workflow JSON.
        saved_widget_values = [
            False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, True,
            "GuidedFilter", 8, 6, 0.01, 0.99, False, "cpu", 2.0,
        ]
        arguments = dict(zip(self.EXPECTED_INPUT_ORDER[1:], saved_widget_values))
        arguments["image"] = image

        with patch(
            "ComfyUI_Human_Parts_Urutora.nodes_urutora._load_session",
            return_value=session,
        ):
            _, mask = HumanPartsUrutora.execute(**arguments).result

        expected = torch.tensor(
            [golden["parts"]["right_foot"]], dtype=torch.float32
        )
        self.assertTrue(torch.equal(mask, expected))


class HumanPartsUrutoraRefinementDispatchTests(unittest.TestCase):
    def setUp(self):
        self.golden = _load_golden_fixture()
        rgb = np.asarray(self.golden["rgb"], dtype=np.uint8)
        self.image = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)
        self.session = GoldenSession(
            np.asarray(self.golden["class_map"], dtype=np.int64)
        )

    def test_guided_filter_refinement(self):
        refined = torch.full((1, 4, 4), 0.5, dtype=torch.float32)
        with (
            patch(
                "ComfyUI_Human_Parts_Urutora.nodes_urutora._load_session",
                return_value=self.session,
            ),
            patch(
                "ComfyUI_Human_Parts_Urutora.nodes_urutora.guided_filter_alpha",
                return_value=refined,
            ) as guided,
        ):
            _, mask = HumanPartsUrutora.execute(
                **_node_arguments(self.image, face=True, process_detail=True)
            ).result
        guided.assert_called_once()

    def test_eye_refinement_cannot_create_alpha_outside_trimap_support(self):
        face_map = np.zeros((4, 4), dtype=np.int64)
        face_map[1, 1] = FACE_PARSING_CLASSES["left_eye"]
        face_session = FaceParsingSession(face_map)
        refined = torch.ones((1, 4, 4), dtype=torch.float32)
        trimap = Image.fromarray(
            np.asarray(
                [[0, 0, 0, 0], [0, 128, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=np.uint8,
            ),
            mode="L",
        )
        with (
            patch(
                "ComfyUI_Human_Parts_Urutora.nodes_urutora._load_session",
                side_effect=[self.session, face_session],
            ),
            patch(
                "ComfyUI_Human_Parts_Urutora.nodes_urutora.guided_filter_alpha",
                return_value=refined,
            ),
            patch(
                "ComfyUI_Human_Parts_Urutora.nodes_urutora.generate_vitmatte_trimap",
                return_value=trimap,
            ),
        ):
            _, mask = HumanPartsUrutora.execute(
                **_node_arguments(self.image, eyes=True, process_detail=True)
            ).result

        self.assertEqual(mask.sum().item(), 1.0)
        self.assertEqual(mask[0, 1, 1].item(), 1.0)
        self.assertEqual(mask.shape, (1, 4, 4))

    def test_eye_refinement_cannot_erase_known_foreground(self):
        face_map = np.zeros((4, 4), dtype=np.int64)
        face_session = FaceParsingSession(face_map)
        refined = torch.zeros((1, 4, 4), dtype=torch.float32)
        trimap = Image.fromarray(
            np.asarray(
                [[0, 0, 0, 0], [0, 255, 128, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=np.uint8,
            ),
            mode="L",
        )
        with (
            patch(
                "ComfyUI_Human_Parts_Urutora.nodes_urutora._load_session",
                side_effect=[self.session, face_session],
            ),
            patch(
                "ComfyUI_Human_Parts_Urutora.nodes_urutora.guided_filter_alpha",
                return_value=refined,
            ),
            patch(
                "ComfyUI_Human_Parts_Urutora.nodes_urutora.generate_vitmatte_trimap",
                return_value=trimap,
            ),
        ):
            _, mask = HumanPartsUrutora.execute(
                **_node_arguments(self.image, eyes=True, process_detail=True)
            ).result

        self.assertEqual(mask[0, 1, 1].item(), 1.0)
        self.assertEqual(mask[0, 1, 2].item(), 0.0)

    def test_pymatting_refinement(self):
        refined = torch.full((1, 4, 4), 0.25, dtype=torch.float32)
        with (
            patch(
                "ComfyUI_Human_Parts_Urutora.nodes_urutora._load_session",
                return_value=self.session,
            ),
            patch(
                "ComfyUI_Human_Parts_Urutora.nodes_urutora.pymatting_alpha",
                return_value=refined,
            ) as pymatting,
        ):
            _, mask = HumanPartsUrutora.execute(
                **_node_arguments(
                    self.image,
                    face=True,
                    process_detail=True,
                    detail_method="PyMatting",
                )
            ).result
        pymatting.assert_called_once()
        self.assertTrue(torch.equal(mask, refined))

    def test_every_vitmatte_method_refines_the_mask(self):
        for method in matting.VITMATTE_REPOSITORIES:
            with self.subTest(method=method):
                matte = Image.new("L", (4, 4), 128)
                with (
                    patch(
                        "ComfyUI_Human_Parts_Urutora.nodes_urutora._load_session",
                        return_value=self.session,
                    ),
                    patch(
                        "ComfyUI_Human_Parts_Urutora.nodes_urutora.generate_vitmatte_trimap",
                        return_value=matte,
                    ) as trimap,
                    patch(
                        "ComfyUI_Human_Parts_Urutora.nodes_urutora.generate_vitmatte",
                        return_value=matte,
                    ) as vitmatte,
                ):
                    _, mask = HumanPartsUrutora.execute(
                        **_node_arguments(
                            self.image,
                            face=True,
                            process_detail=True,
                            detail_method=method,
                        )
                    ).result
                trimap.assert_called_once()
                vitmatte.assert_called_once()
                self.assertEqual(mask.shape, (1, 4, 4))


class VitMatteTrimapTests(unittest.TestCase):
    def test_small_components_retain_known_foreground(self):
        mask = torch.zeros((1, 80, 120), dtype=torch.float32)
        mask[:, 20:27, 20:38] = 1.0
        mask[:, 20:27, 82:100] = 1.0

        trimap = np.asarray(
            matting.generate_vitmatte_trimap(mask, 8, 6), dtype=np.uint8
        )

        self.assertTrue((trimap[20:27, 20:38] == 255).any())
        self.assertTrue((trimap[20:27, 82:100] == 255).any())
        self.assertTrue((trimap[20:27, 38:82] == 0).any())

    def test_large_component_keeps_requested_erosion(self):
        mask = torch.zeros((1, 100, 100), dtype=torch.float32)
        mask[:, 10:90, 10:90] = 1.0

        trimap = np.asarray(
            matting.generate_vitmatte_trimap(mask, 3, 3), dtype=np.uint8
        )

        self.assertEqual(trimap[10, 10], 128)
        self.assertEqual(trimap[50, 50], 255)


class TorchGuidedFilterTests(unittest.TestCase):
    def test_refinement_does_not_require_opencv_ximgproc(self):
        image = torch.rand((1, 7, 9, 3), dtype=torch.float32)
        mask = torch.rand((1, 7, 9), dtype=torch.float32)

        with patch.dict(sys.modules, {"cv2": None, "cv2.ximgproc": None}):
            refined = matting.guided_filter_alpha(image, mask, 2)

        self.assertEqual(refined.shape, mask.shape)
        self.assertEqual(refined.dtype, torch.float32)
        self.assertEqual(refined.device, image.device)
        self.assertTrue(torch.isfinite(refined).all())

    def test_constant_alpha_is_preserved(self):
        image = torch.rand((2, 5, 6, 3), dtype=torch.float32)
        mask = torch.full((2, 5, 6), 0.37, dtype=torch.float32)

        refined = matting.guided_filter_alpha(image, mask, 3)

        self.assertTrue(torch.allclose(refined, mask, atol=1e-6))

    def test_guide_contrast_preserves_a_matching_mask_edge(self):
        mask = torch.zeros((1, 9, 9), dtype=torch.float32)
        mask[:, :, 4:] = 1.0
        contrast_guide = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        flat_guide = torch.zeros_like(contrast_guide)

        contrast_result = matting.guided_filter_alpha(contrast_guide, mask, 3)
        flat_result = matting.guided_filter_alpha(flat_guide, mask, 3)

        contrast_error = (contrast_result - mask).abs().mean()
        flat_error = (flat_result - mask).abs().mean()
        self.assertLess(contrast_error, flat_error)

    def test_invalid_image_and_mask_shapes_are_actionable(self):
        with self.assertRaisesRegex(ValueError, "dimensions must match"):
            matting.guided_filter_alpha(
                torch.zeros((1, 4, 4, 3)), torch.zeros((1, 5, 4)), 1
            )


class OnnxSessionLifecycleTests(unittest.TestCase):
    def tearDown(self):
        _load_session.cache_clear()

    def test_missing_model_has_an_actionable_error(self):
        missing = str(Path(tempfile.gettempdir()) / "missing-human-parts.onnx")
        with self.assertRaisesRegex(FileNotFoundError, "Run install.py"):
            _load_session(missing, ("CPUExecutionProvider",))

    def test_corrupt_model_reports_provider_attempts(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx") as model_file:
            with patch(
                "ComfyUI_Human_Parts_Urutora.nodes_urutora.ort.InferenceSession",
                side_effect=ValueError("invalid protobuf"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "CPUExecutionProvider.*invalid protobuf"
                ):
                    _load_session(model_file.name, ("CPUExecutionProvider",))

    def test_identical_requests_reuse_one_inference_session(self):
        session = object()
        providers = ("CPUExecutionProvider",)

        with tempfile.NamedTemporaryFile(suffix=".onnx") as model_file:
            with patch(
                "ComfyUI_Human_Parts_Urutora.onnx_lifecycle.ort.InferenceSession",
                return_value=session,
            ) as inference_session:
                first = _load_session(model_file.name, providers)
                second = _load_session(model_file.name, providers)

        self.assertIs(first, session)
        self.assertIs(second, session)
        inference_session.assert_called_once_with(
            model_file.name, providers=["CPUExecutionProvider"]
        )

    def test_provider_initialization_falls_back_to_cpu(self):
        cpu_session = object()

        def create_session(path, providers=None):
            if providers == ["CPUExecutionProvider"]:
                return cpu_session
            raise RuntimeError("provider unavailable")

        with tempfile.NamedTemporaryFile(suffix=".onnx") as model_file:
            with patch(
                "ComfyUI_Human_Parts_Urutora.nodes_urutora.ort.InferenceSession",
                side_effect=create_session,
            ) as inference_session:
                actual = _load_session(
                    model_file.name,
                    ("CUDAExecutionProvider", "CPUExecutionProvider"),
                )

        self.assertIs(actual, cpu_session)
        self.assertEqual(
            inference_session.call_args_list,
            [
                call(
                    model_file.name,
                    providers=[
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                ),
                call(model_file.name, providers=["CPUExecutionProvider"]),
            ],
        )

    def test_provider_preference_only_contains_available_providers(self):
        with patch(
            "ComfyUI_Human_Parts_Urutora.nodes_urutora.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ):
            self.assertEqual(_execution_providers(), ("CPUExecutionProvider",))

    def test_auto_policy_keeps_accelerated_and_cpu_fallbacks(self):
        with patch(
            "ComfyUI_Human_Parts_Urutora.nodes_urutora.ort.get_available_providers",
            return_value=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
                "TensorrtExecutionProvider",
            ],
        ):
            self.assertEqual(
                lifecycle_execution_providers("auto"),
                (
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ),
            )
            self.assertEqual(
                lifecycle_execution_providers("cuda"),
                ("CUDAExecutionProvider", "CPUExecutionProvider"),
            )

    def test_unavailable_explicit_policy_is_actionable(self):
        with patch(
            "ComfyUI_Human_Parts_Urutora.nodes_urutora.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ):
            with self.assertRaisesRegex(
                RuntimeError, "CUDAExecutionProvider.*=auto"
            ):
                lifecycle_execution_providers("cuda")

    def test_invalid_provider_policy_lists_choices(self):
        with self.assertRaisesRegex(ValueError, "auto, cuda, cpu"):
            lifecycle_execution_providers("fastest")


class FakeMovableTensor:
    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(torch.device(device))
        return self


class FakeVitMatteModel:
    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(torch.device(device))
        return self

    def __call__(self, **inputs):
        return SimpleNamespace(alphas=torch.ones((1, 1, 2, 2)))


class FakeVitMatteProcessor:
    def __call__(self, **kwargs):
        return {"pixel_values": FakeMovableTensor()}


class VitMatteDeviceLifecycleTests(unittest.TestCase):
    def test_explicit_cpu_device(self):
        self.assertEqual(matting._resolve_torch_device("cpu"), torch.device("cpu"))

    def test_explicit_cuda_falls_back_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(
                matting._resolve_torch_device("cuda"), torch.device("cpu")
            )

    def test_explicit_cuda_is_preserved_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(
                matting._resolve_torch_device("cuda"), torch.device("cuda")
            )

    def test_auto_uses_comfyui_model_management_device(self):
        comfy_module = ModuleType("comfy")
        management_module = ModuleType("comfy.model_management")
        management_module.get_torch_device = Mock(return_value="mps")
        comfy_module.model_management = management_module
        with patch.dict(
            sys.modules,
            {
                "comfy": comfy_module,
                "comfy.model_management": management_module,
            },
        ):
            self.assertEqual(
                matting._resolve_torch_device("auto"), torch.device("mps")
            )

    def test_vitmatte_offloads_after_each_repeated_execution(self):
        model = FakeVitMatteModel()
        processor = FakeVitMatteProcessor()
        image = Image.new("RGB", (2, 2), "white")
        trimap = Image.new("L", (2, 2), 128)
        comfy_module = ModuleType("comfy")
        management_module = ModuleType("comfy.model_management")
        comfy_module.model_management = management_module

        with (
            patch.dict(
                sys.modules,
                {"comfy": comfy_module, "comfy.model_management": management_module},
            ),
            patch(
                "ComfyUI_Human_Parts_Urutora.detector.matting._load_vitmatte",
                return_value=(model, processor),
            ) as loader,
            patch(
                "ComfyUI_Human_Parts_Urutora.detector.matting._resolve_torch_device",
                return_value=torch.device("cuda"),
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as empty_cache,
            patch.object(matting._load_vitmatte, "cache_clear") as cache_clear,
        ):
            first = matting.generate_vitmatte(
                image, trimap, "VITMatte(local)", "cuda", 2.0
            )
            second = matting.generate_vitmatte(
                image, trimap, "VITMatte(local)", "cuda", 2.0
            )

        self.assertEqual(first.size, (2, 2))
        self.assertEqual(second.size, (2, 2))
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(
            model.devices,
            [
                torch.device("cuda"),
                torch.device("cpu"),
                torch.device("cuda"),
                torch.device("cpu"),
            ],
        )
        self.assertEqual(empty_cache.call_count, 2)
        self.assertEqual(cache_clear.call_count, 2)

    def test_vitmatte_cleans_up_after_inference_failure(self):
        model = FakeVitMatteModel()
        processor = Mock(side_effect=RuntimeError("preprocessing failed"))
        comfy_module = ModuleType("comfy")
        management_module = ModuleType("comfy.model_management")
        comfy_module.model_management = management_module

        with (
            patch.dict(
                sys.modules,
                {"comfy": comfy_module, "comfy.model_management": management_module},
            ),
            patch(
                "ComfyUI_Human_Parts_Urutora.detector.matting._load_vitmatte",
                return_value=(model, processor),
            ),
            patch(
                "ComfyUI_Human_Parts_Urutora.detector.matting._resolve_torch_device",
                return_value=torch.device("cuda"),
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as empty_cache,
            patch.object(matting._load_vitmatte, "cache_clear") as cache_clear,
            self.assertRaisesRegex(RuntimeError, "preprocessing failed"),
        ):
            matting.generate_vitmatte(
                Image.new("RGB", (2, 2)),
                Image.new("L", (2, 2)),
                "VITMatte(local)",
                "cuda",
                2.0,
            )

        self.assertEqual(model.devices, [torch.device("cuda"), torch.device("cpu")])
        cache_clear.assert_called_once_with()
        empty_cache.assert_called_once_with()

    def test_missing_local_vitmatte_model_is_reported(self):
        transformers_module = ModuleType("transformers")
        transformers_module.VitMatteForImageMatting = object
        transformers_module.VitMatteImageProcessor = object
        with (
            patch.dict(sys.modules, {"transformers": transformers_module}),
            patch(
                "ComfyUI_Human_Parts_Urutora.detector.matting.os.path.isdir",
                return_value=False,
            ),
            self.assertRaisesRegex(FileNotFoundError, "VITMatte model not found"),
        ):
            matting._load_vitmatte.__wrapped__(
                "hustvl/vitmatte-small-composition-1k", True
            )


if __name__ == "__main__":
    unittest.main()
