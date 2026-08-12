from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


PROJECT_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_PARENT))

from ComfyUI_Human_Parts import NODE_CLASS_MAPPINGS
from ComfyUI_Human_Parts.nodes_ultra import HumanPartsUltra, _segment_parts


class FakeSession:
    def get_inputs(self):
        return [SimpleNamespace(name="image")]

    def get_outputs(self):
        return [SimpleNamespace(name="segmentation")]

    def run(self, output_names, inputs):
        logits = np.zeros((1, 2, 2, 22), dtype=np.float32)
        logits[0, 0, 0, 18] = 1.0
        logits[0, 0, 1, 19] = 1.0
        logits[0, 1, :, 0] = 1.0
        return [logits]


def _node_arguments(image: torch.Tensor) -> dict:
    return {
        "image": image,
        "face": False,
        "hair": False,
        "glasses": False,
        "top_clothes": False,
        "bottom_clothes": False,
        "torso_skin": False,
        "left_arm": False,
        "right_arm": False,
        "left_leg": False,
        "right_leg": False,
        "left_foot": True,
        "right_foot": False,
        "detail_method": "GuidedFilter",
        "detail_erode": 8,
        "detail_dilate": 6,
        "black_point": 0.01,
        "white_point": 0.99,
        "process_detail": False,
        "device": "cpu",
        "max_megapixels": 2.0,
    }


class HumanPartsUltraTests(unittest.TestCase):
    def test_layerstyle_workflow_identifier_is_registered(self):
        self.assertIs(
            NODE_CLASS_MAPPINGS["LayerMask: HumanPartsUltra"], HumanPartsUltra
        )

    def test_widget_order_matches_layerstyle_workflows(self):
        self.assertEqual(
            list(HumanPartsUltra.INPUT_TYPES()["required"]),
            [
                "image",
                "face",
                "hair",
                "glasses",
                "top_clothes",
                "bottom_clothes",
                "torso_skin",
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
            ],
        )

    def test_left_and_right_foot_selections_are_independent(self):
        image = Image.new("RGB", (2, 2), "black")
        left = _segment_parts(image, FakeSession(), {"left_foot": True})
        right = _segment_parts(image, FakeSession(), {"right_foot": True})

        self.assertEqual(left.shape, (1, 2, 2))
        self.assertEqual(right.shape, (1, 2, 2))
        self.assertTrue(
            torch.equal(left, torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]))
        )
        self.assertTrue(
            torch.equal(right, torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]))
        )

    def test_node_returns_modern_batch_shapes(self):
        image = torch.zeros((2, 2, 2, 3), dtype=torch.float32)

        with patch(
            "ComfyUI_Human_Parts.nodes_ultra._load_session",
            return_value=FakeSession(),
        ):
            rgba, mask = HumanPartsUltra().human_parts_ultra(
                **_node_arguments(image)
            )

        self.assertEqual(rgba.shape, (2, 2, 2, 4))
        self.assertEqual(rgba.dtype, torch.float32)
        self.assertEqual(mask.shape, (2, 2, 2))
        self.assertEqual(mask.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
