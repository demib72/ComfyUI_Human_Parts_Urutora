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
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "human_parts_ultra_golden.json"
SCHEMA_FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "v1_node_schemas.json"
sys.path.insert(0, str(PROJECT_PARENT))
# Custom nodes normally run inside ComfyUI. Make its V3 API importable when the
# test checkout is installed beside ComfyUI, matching the documented layout.
COMFYUI_ROOT = PROJECT_PARENT / "ComfyUI"
if COMFYUI_ROOT.is_dir():
    sys.path.insert(0, str(COMFYUI_ROOT))

from ComfyUI_Human_Parts import comfy_entrypoint
from ComfyUI_Human_Parts.nodes import HumanParts
from ComfyUI_Human_Parts.detector.human_parts import get_mask
from ComfyUI_Human_Parts.detector import matting
from ComfyUI_Human_Parts.nodes_ultra import (
    ULTRA_ANATOMICAL_PARTS,
    ULTRA_CLASSES,
    HumanPartsUltra,
    LayerStyleHumanPartsUltra,
    _execution_providers,
    lifecycle_execution_providers,
    _load_session,
    _segment_parts,
)


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


def _load_golden_fixture() -> dict:
    with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def _node_arguments(image: torch.Tensor, **overrides) -> dict:
    arguments = {
        "image": image,
        **{name: False for name in ULTRA_CLASSES},
        **{name: False for name in ULTRA_ANATOMICAL_PARTS},
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


class LegacyHumanPartsTests(unittest.TestCase):
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
                "ComfyUI_Human_Parts.nodes.execution_providers",
                return_value=("CPUExecutionProvider",),
            ),
            patch(
                "ComfyUI_Human_Parts.nodes.load_session", return_value=session
            ) as load,
        ):
            first = HumanParts.execute(image, face=True).result[0]
            second = HumanParts.execute(image, face=True).result[0]

        self.assertEqual(first.shape, (1, 3, 5))
        self.assertEqual(second.dtype, torch.float32)
        self.assertEqual(load.call_count, 2)
        self.assertEqual(load.call_args_list[0], load.call_args_list[1])


class HumanPartsUltraGoldenTests(unittest.TestCase):
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
            "all": {name: True for name in ULTRA_CLASSES},
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
            "ComfyUI_Human_Parts.nodes_ultra._load_session",
            return_value=self.session,
        ):
            rgba, mask = HumanPartsUltra.execute(
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
        class_map[5:30, 20:60] = ULTRA_CLASSES["face"]
        class_map[30:70, 15:65] = ULTRA_CLASSES["torso_skin"]
        class_map[70:100, 20:38] = ULTRA_CLASSES["left_leg"]
        class_map[70:100, 42:60] = ULTRA_CLASSES["right_leg"]
        image = Image.new("RGB", (80, 100), "white")
        session = GoldenSession(class_map)

        allowed_classes = {
            "eyes": {ULTRA_CLASSES["face"]},
            "breasts": {ULTRA_CLASSES["torso_skin"]},
            "groin": {
                ULTRA_CLASSES["torso_skin"],
                ULTRA_CLASSES["left_leg"],
                ULTRA_CLASSES["right_leg"],
            },
        }
        for part_name in ULTRA_ANATOMICAL_PARTS:
            with self.subTest(part=part_name):
                mask = _segment_parts(image, session, {part_name: True})
                selected = mask.squeeze(0).numpy().astype(bool)
                self.assertTrue(selected.any())
                self.assertTrue(
                    np.isin(class_map[selected], list(allowed_classes[part_name])).all()
                )


class HumanPartsUltraWorkflowCompatibilityTests(unittest.TestCase):
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

    def test_legacy_workflow_identifiers_are_registered(self):
        import asyncio

        extension = asyncio.run(comfy_entrypoint())
        node_classes = asyncio.run(extension.get_node_list())
        node_ids = [node.define_schema().node_id for node in node_classes]
        self.assertEqual(
            node_ids,
            ["HumanParts", "LayerMask: HumanPartsUltra", "HumanPartsUltra"],
        )

    def test_widget_order_matches_layerstyle_workflows(self):
        schema = HumanPartsUltra.define_schema()
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

        classes = [HumanParts, LayerStyleHumanPartsUltra, HumanPartsUltra]
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
        for node_class in [HumanParts, LayerStyleHumanPartsUltra, HumanPartsUltra]:
            with self.subTest(node=node_class.__name__):
                self.assertEqual(
                    node_class.GET_SCHEMA().node_id,
                    node_class.define_schema().node_id,
                )

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
            "ComfyUI_Human_Parts.nodes_ultra._load_session",
            return_value=session,
        ):
            _, mask = HumanPartsUltra.execute(**arguments).result

        expected = torch.tensor(
            [golden["parts"]["right_foot"]], dtype=torch.float32
        )
        self.assertTrue(torch.equal(mask, expected))


class HumanPartsUltraRefinementDispatchTests(unittest.TestCase):
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
                "ComfyUI_Human_Parts.nodes_ultra._load_session",
                return_value=self.session,
            ),
            patch(
                "ComfyUI_Human_Parts.nodes_ultra.guided_filter_alpha",
                return_value=refined,
            ) as guided,
        ):
            _, mask = HumanPartsUltra.execute(
                **_node_arguments(self.image, face=True, process_detail=True)
            ).result
        guided.assert_called_once()
        self.assertEqual(mask.shape, (1, 4, 4))

    def test_pymatting_refinement(self):
        refined = torch.full((1, 4, 4), 0.25, dtype=torch.float32)
        with (
            patch(
                "ComfyUI_Human_Parts.nodes_ultra._load_session",
                return_value=self.session,
            ),
            patch(
                "ComfyUI_Human_Parts.nodes_ultra.pymatting_alpha",
                return_value=refined,
            ) as pymatting,
        ):
            _, mask = HumanPartsUltra.execute(
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
                        "ComfyUI_Human_Parts.nodes_ultra._load_session",
                        return_value=self.session,
                    ),
                    patch(
                        "ComfyUI_Human_Parts.nodes_ultra.generate_vitmatte_trimap",
                        return_value=matte,
                    ) as trimap,
                    patch(
                        "ComfyUI_Human_Parts.nodes_ultra.generate_vitmatte",
                        return_value=matte,
                    ) as vitmatte,
                ):
                    _, mask = HumanPartsUltra.execute(
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
                "ComfyUI_Human_Parts.nodes_ultra.ort.InferenceSession",
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
                "ComfyUI_Human_Parts.onnx_lifecycle.ort.InferenceSession",
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
                "ComfyUI_Human_Parts.nodes_ultra.ort.InferenceSession",
                side_effect=create_session,
            ) as inference_session:
                actual = _load_session(
                    model_file.name,
                    ("TensorrtExecutionProvider", "CPUExecutionProvider"),
                )

        self.assertIs(actual, cpu_session)
        self.assertEqual(
            inference_session.call_args_list,
            [
                call(
                    model_file.name,
                    providers=[
                        "TensorrtExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                ),
                call(model_file.name, providers=["CPUExecutionProvider"]),
            ],
        )

    def test_provider_preference_only_contains_available_providers(self):
        with patch(
            "ComfyUI_Human_Parts.nodes_ultra.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ):
            self.assertEqual(_execution_providers(), ("CPUExecutionProvider",))

    def test_auto_policy_keeps_accelerated_and_cpu_fallbacks(self):
        with patch(
            "ComfyUI_Human_Parts.nodes_ultra.ort.get_available_providers",
            return_value=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
                "TensorrtExecutionProvider",
            ],
        ):
            self.assertEqual(
                lifecycle_execution_providers("auto"),
                (
                    "TensorrtExecutionProvider",
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
            "ComfyUI_Human_Parts.nodes_ultra.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ):
            with self.assertRaisesRegex(
                RuntimeError, "CUDAExecutionProvider.*=auto"
            ):
                lifecycle_execution_providers("cuda")

    def test_invalid_provider_policy_lists_choices(self):
        with self.assertRaisesRegex(ValueError, "auto, tensorrt, cuda, cpu"):
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
                "ComfyUI_Human_Parts.detector.matting._load_vitmatte",
                return_value=(model, processor),
            ) as loader,
            patch(
                "ComfyUI_Human_Parts.detector.matting._resolve_torch_device",
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
                "ComfyUI_Human_Parts.detector.matting._load_vitmatte",
                return_value=(model, processor),
            ),
            patch(
                "ComfyUI_Human_Parts.detector.matting._resolve_torch_device",
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
                "ComfyUI_Human_Parts.detector.matting.os.path.isdir",
                return_value=False,
            ),
            self.assertRaisesRegex(FileNotFoundError, "VITMatte model not found"),
        ):
            matting._load_vitmatte.__wrapped__(
                "hustvl/vitmatte-small-composition-1k", True
            )


if __name__ == "__main__":
    unittest.main()
