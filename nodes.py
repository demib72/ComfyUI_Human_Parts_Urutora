import torch
from comfy_api.latest import io

from .detector.human_parts import get_mask, labels
from .onnx_lifecycle import execution_providers, load_session
from .utils import model_path


class HumanParts(io.ComfyNode):
    """
    This node is used to get a mask of the human parts in the image.

    The model used is DeepLabV3+ with a ResNet50 backbone trained
    by Keras-io, converted to ONNX format.

    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HumanParts",
            display_name="🧍 Human Parts Urutora mask generator",
            category="Human Parts Urutora",
            description=cls.__doc__ or "",
            inputs=[
                io.Image.Input(
                    "image",
                    display_name="Image",
                    tooltip="The image in which to detect human parts",
                ),
                *[
                    io.Boolean.Input(
                        segment_id,
                        default=False,
                        label_on="Enabled",
                        label_off="Disabled",
                        tooltip=tooltip,
                    )
                    for segment_id, tooltip in labels.values()
                    if segment_id
                ],
            ],
            outputs=[io.Mask.Output("mask")],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, **kwargs) -> io.NodeOutput:
        """
        Return a Tensor with the mask of the human parts in the image.
        """

        model = load_session(model_path, execution_providers())
        ret_tensor, _ = get_mask(image, model=model, rotation=0, **kwargs)

        return io.NodeOutput(ret_tensor)
