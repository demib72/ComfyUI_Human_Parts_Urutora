from typing import Tuple

import onnxruntime as ort
import torch

from .detector.human_parts import get_mask
from .utils import model_path


class HumanParts:
    """
    This node is used to get a mask of the human parts in the image.

    The model used is DeepLabV3+ with a ResNet50 backbone trained
    by Keras-io, converted to ONNX format.

    """

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "get_mask"
    CATEGORY = "Metal3d"
    OUTPU_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        def _bool_widget(is_enabled=False, tooltip: str | None = None):
            """Helper function to create a boolean widget"""
            return (
                "BOOLEAN",
                {
                    "default": is_enabled,
                    "label_on": "Enabled",
                    "label_off": "Disabled",
                    "tooltip": tooltip,
                },
            )

        return {
            "required": {
                "image": ("IMAGE",),
                "background": _bool_widget(  # 0
                    tooltip="Background, excluding human parts, invert this mask to get the human parts",
                ),
                "hat": _bool_widget(  # 1
                    tooltip="Hat, cap, et.",
                ),
                "hair": _bool_widget(  # x2
                    tooltip="Hair, including beard, mustache, etc.",
                ),
                "gloves": _bool_widget(  # 3
                    tooltip="Gloves, mittens, etc.",
                ),
                "glasses": _bool_widget(  # 4
                    tooltip="Glasses, sunglasses, etc. Eyes can be included"
                ),
                "top-clothes": _bool_widget(  # 5
                    tooltip="Shirt, T-shirt, etc.",
                ),
                "dress": _bool_widget(  # 6
                    tooltip="Dress, skirt, etc.",
                ),
                "coat": _bool_widget(  # 7
                    tooltip="Coat, jacket, etc.",
                ),
                "socks": _bool_widget(  # uint8
                    tooltip="Socks",
                ),
                "bottom-clothes": _bool_widget(  # 9
                    tooltip="Pants, shorts, etc.",
                ),
                "torso-skin": _bool_widget(  # 10
                    tooltip="Skin of the torso, excluding clothes. Neck can be included"
                ),
                "scarf": _bool_widget(  # 11
                    tooltip="Scarf, bandana, etc.",
                ),
                "skirt": _bool_widget(  # 12
                    tooltip="Skirt",
                ),
                "face": _bool_widget(  # 13
                    is_enabled=True,
                    tooltip="Face, including eyes, mouth, etc.",
                ),
                "left-arm": _bool_widget(  # 14
                    tooltip="Left arm, excluding clothes, hand can be included"
                ),
                "right-arm": _bool_widget(  # 15
                    tooltip="Right arm, excluding clothes, hand can be included"
                ),
                "left-leg": _bool_widget(  # 16
                    tooltip="Left leg, excluding clothes, foot can be included"
                ),
                "right-leg": _bool_widget(  # 17
                    tooltip="Right leg, excluding clothes, foot can be included"
                ),
                "left-foot": _bool_widget(  # 18
                    tooltip="Left foot, excluding shoes",
                ),
                "right-foot": _bool_widget(  # 19
                    tooltip="Right foot, excluding shoes",
                ),
            }
        }

    def get_mask(self, image: torch.Tensor, **kwargs) -> Tuple[torch.Tensor]:
        """
        Return a Tensor with the mask of the human parts in the image.
        """

        model = ort.InferenceSession(model_path)
        ret_tensor, _ = get_mask(image, model=model, rotation=0, **kwargs)

        return (ret_tensor,)
