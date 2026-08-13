"""Human Parts Urutora node adapted from ComfyUI LayerStyle Advance.

The original implementation is Copyright (c) 2024 chflame163 and is used
under the MIT License. See THIRD_PARTY_NOTICES for the complete notice.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from .comfy_compat import HAS_COMFY_V3, io
from .detector.face_parsing import FACE_PARSING_CLASSES, segment_face_parts
from .detector.matting import (
    VITMATTE_REPOSITORIES,
    generate_vitmatte,
    generate_vitmatte_trimap,
    guided_filter_alpha,
    histogram_remap,
    mask_to_pil,
    pil_to_mask,
    pil_to_tensor,
    pymatting_alpha,
    rgba_with_mask,
    tensor_to_pil,
)
from .onnx_lifecycle import (
    execution_providers as _execution_providers,
    lifecycle_execution_providers,
    load_session as _load_session,
)
from .utils import face_model_path, model_path


URUTORA_CLASSES = {
    "hair": 2,
    "glasses": 4,
    "top_clothes": 5,
    "bottom_clothes": 9,
    "torso_skin": 10,
    "face": 13,
    "left_arm": 14,
    "right_arm": 15,
    "left_leg": 16,
    "right_leg": 17,
    "left_foot": 18,
    "right_foot": 19,
}

URUTORA_INPUT_ORDER = (
    "face",
    "face_skin",
    "eyebrows",
    "eyes",
    "nose",
    "mouth",
    "lips",
    "ears",
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
)

# These regions are not native classes in the 22-class CCIHP model. Face skin
# uses the finer BiSeNet parser; eyes retain a geometric fallback for installs
# that do not yet have the face-parsing model.
URUTORA_FACE_PART_CLASSES = {
    "face_skin": ("skin",),
    "eyebrows": ("left_eyebrow", "right_eyebrow"),
    "eyes": ("left_eye", "right_eye"),
    "nose": ("nose",),
    "mouth": ("mouth",),
    "lips": ("upper_lip", "lower_lip"),
    "ears": ("left_ear", "right_ear"),
}
URUTORA_ANATOMICAL_PARTS = tuple(URUTORA_FACE_PART_CLASSES)
URUTORA_ANATOMICAL_TOOLTIPS = {
    "face_skin": (
        "Mask facial skin while preserving eyebrows, eyes, nose, mouth, lips, "
        "and ears unless their separate controls are enabled. Includes the "
        "skin around the eye sockets. Do not enable the whole-face option at "
        "the same time."
    ),
    "eyebrows": "Mask both eyebrows using the face parser.",
    "eyes": "Estimate both eye regions within the detected face.",
    "nose": "Mask the nose using the face parser.",
    "mouth": "Mask the inner mouth region using the face parser.",
    "lips": "Mask both upper and lower lips using the face parser.",
    "ears": "Mask both ears using the face parser.",
}
URUTORA_ANATOMICAL_DISPLAY_NAMES = {
    "face_skin": "face skin (preserve features)"
}


def _ellipse_mask(
    shape: tuple[int, int],
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> np.ndarray:
    """Return a boolean ellipse without introducing another image dependency."""
    height, width = shape
    if radius_x <= 0 or radius_y <= 0:
        return np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[:height, :width]
    return (
        ((xx - center_x) / radius_x) ** 2
        + ((yy - center_y) / radius_y) ** 2
        <= 1.0
    )


def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return columns.min(), rows.min(), columns.max() + 1, rows.max() + 1


def _anatomical_mask(class_map: np.ndarray, part_name: str) -> np.ndarray:
    """Estimate a smaller anatomical region from the coarse CCIHP classes.

    The estimates deliberately remain inside the relevant semantic region, so
    they do not turn nearby background into an inpainting mask.
    """
    result = np.zeros(class_map.shape, dtype=bool)

    if part_name == "eyes":
        source = class_map == URUTORA_CLASSES["face"]
        bounds = _mask_bounds(source)
        if bounds is None:
            return result
        x0, y0, x1, y1 = bounds
        width, height = x1 - x0, y1 - y0
        for horizontal_position in (0.32, 0.68):
            result |= _ellipse_mask(
                class_map.shape,
                x0 + horizontal_position * width,
                y0 + 0.43 * height,
                max(1.0, 0.16 * width),
                max(1.0, 0.10 * height),
            )
        return result & source

    return result


def _segment_parts(
    image: Image.Image,
    model: ort.InferenceSession,
    selections: dict[str, bool],
    face_model: ort.InferenceSession | None = None,
) -> torch.Tensor:
    original_size = image.size
    resized = image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    input_array = np.asarray(resized).astype(np.float32) / 127.5 - 1.0
    input_array = np.expand_dims(input_array, axis=0)

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    output = model.run([output_name], {input_name: input_array})[0]
    class_map = np.asarray(output).argmax(axis=3).squeeze(0)

    selected_indices = [
        URUTORA_CLASSES[name]
        for name, enabled in selections.items()
        if enabled and name in URUTORA_CLASSES
    ]
    if selected_indices:
        mask = np.isin(class_map, selected_indices).astype(np.uint8) * 255
    else:
        mask = np.zeros_like(class_map, dtype=np.uint8)

    parsed_face_mask: np.ndarray | None = None
    parsed_classes = [
        FACE_PARSING_CLASSES[class_name]
        for part_name, class_names in URUTORA_FACE_PART_CLASSES.items()
        if selections.get(part_name, False)
        for class_name in class_names
    ]
    if parsed_classes and face_model is not None:
        parsed_face_mask = segment_face_parts(
            image,
            class_map,
            URUTORA_CLASSES["face"],
            face_model,
            tuple(parsed_classes),
            skin_class=(
                FACE_PARSING_CLASSES["skin"]
                if selections.get("face_skin", False)
                else None
            ),
        )

    for part_name in URUTORA_ANATOMICAL_PARTS:
        if selections.get(part_name, False):
            if part_name == "eyes" and face_model is None:
                mask[_anatomical_mask(class_map, part_name)] = 255

    mask_image = Image.fromarray(mask, mode="L").resize(
        original_size, Image.Resampling.NEAREST
    )
    if parsed_face_mask is not None:
        combined = np.maximum(
            np.asarray(mask_image, dtype=np.uint8), parsed_face_mask
        )
        mask_image = Image.fromarray(combined, mode="L")
    return pil_to_mask(mask_image)


class HumanPartsUrutora(io.ComfyNode):
    """Generate and optionally refine masks for selected human parts."""

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute_legacy"
    CATEGORY = "Human Parts Urutora"
    DESCRIPTION = __doc__ or ""

    if not HAS_COMFY_V3:

        @classmethod
        def INPUT_TYPES(cls):
            methods = list(VITMATTE_REPOSITORIES) + ["PyMatting", "GuidedFilter"]
            booleans = {
                name: (
                    "BOOLEAN",
                    {
                        "default": False,
                        **(
                            {"label": URUTORA_ANATOMICAL_DISPLAY_NAMES[name]}
                            if name in URUTORA_ANATOMICAL_DISPLAY_NAMES
                            else {}
                        ),
                        **(
                            {"tooltip": URUTORA_ANATOMICAL_TOOLTIPS[name]}
                            if name in URUTORA_ANATOMICAL_TOOLTIPS
                            else {}
                        ),
                    },
                )
                for name in URUTORA_INPUT_ORDER
            }
            return {
                "required": {
                    "image": ("IMAGE",),
                    **booleans,
                    "detail_method": (methods,),
                    "detail_erode": (
                        "INT",
                        {"default": 8, "min": 1, "max": 255, "step": 1},
                    ),
                    "detail_dilate": (
                        "INT",
                        {"default": 6, "min": 1, "max": 255, "step": 1},
                    ),
                    "black_point": (
                        "FLOAT",
                        {
                            "default": 0.01,
                            "min": 0.01,
                            "max": 0.98,
                            "step": 0.01,
                        },
                    ),
                    "white_point": (
                        "FLOAT",
                        {
                            "default": 0.99,
                            "min": 0.02,
                            "max": 0.99,
                            "step": 0.01,
                        },
                    ),
                    "process_detail": ("BOOLEAN", {"default": True}),
                    "device": (["cuda", "cpu", "auto"], {"default": "auto"}),
                    "max_megapixels": (
                        "FLOAT",
                        {"default": 2.0, "min": 1.0, "max": 999.0, "step": 0.1},
                    ),
                }
            }

    @classmethod
    def define_schema(cls) -> io.Schema:
        methods = list(VITMATTE_REPOSITORIES) + ["PyMatting", "GuidedFilter"]
        return io.Schema(
            node_id="HumanPartsUrutora",
            display_name="🧍 Human Parts Urutora",
            category="Human Parts Urutora",
            description=cls.__doc__ or "",
            inputs=[
                io.Image.Input("image"),
                *[
                    io.Boolean.Input(
                        name,
                        default=False,
                        display_name=URUTORA_ANATOMICAL_DISPLAY_NAMES.get(name),
                        tooltip=URUTORA_ANATOMICAL_TOOLTIPS.get(name),
                    )
                    for name in URUTORA_INPUT_ORDER
                ],
                io.Combo.Input("detail_method", options=methods),
                io.Int.Input("detail_erode", default=8, min=1, max=255, step=1),
                io.Int.Input("detail_dilate", default=6, min=1, max=255, step=1),
                io.Float.Input(
                    "black_point",
                    default=0.01,
                    min=0.01,
                    max=0.98,
                    step=0.01,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Float.Input(
                    "white_point",
                    default=0.99,
                    min=0.02,
                    max=0.99,
                    step=0.01,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Boolean.Input("process_detail", default=True),
                # Keep legacy choices and ordering for positional workflow data.
                io.Combo.Input(
                    "device", options=["cuda", "cpu", "auto"], default="auto"
                ),
                io.Float.Input(
                    "max_megapixels",
                    default=2.0,
                    min=1.0,
                    max=999.0,
                    step=0.1,
                ),
            ],
            outputs=[io.Image.Output("image"), io.Mask.Output("mask")],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        face: bool,
        hair: bool,
        glasses: bool,
        top_clothes: bool,
        bottom_clothes: bool,
        torso_skin: bool,
        left_arm: bool,
        right_arm: bool,
        left_leg: bool,
        right_leg: bool,
        left_foot: bool,
        right_foot: bool,
        detail_method: str,
        detail_erode: int,
        detail_dilate: int,
        black_point: float,
        white_point: float,
        process_detail: bool,
        device: str,
        max_megapixels: float,
        eyes: bool = False,
        face_skin: bool = False,
        eyebrows: bool = False,
        nose: bool = False,
        mouth: bool = False,
        lips: bool = False,
        ears: bool = False,
        # Retained as hidden keyword arguments so API callers using an older
        # workflow fail safely after the corresponding widgets are removed.
        breasts: bool = False,
        groin: bool = False,
    ):
        providers = _execution_providers()
        model = _load_session(model_path, providers)
        selections = {
            "face": face,
            "hair": hair,
            "glasses": glasses,
            "top_clothes": top_clothes,
            "bottom_clothes": bottom_clothes,
            "torso_skin": torso_skin,
            "left_arm": left_arm,
            "right_arm": right_arm,
            "left_leg": left_leg,
            "right_leg": right_leg,
            "left_foot": left_foot,
            "right_foot": right_foot,
            "eyes": eyes,
            "face_skin": face_skin,
            "eyebrows": eyebrows,
            "nose": nose,
            "mouth": mouth,
            "lips": lips,
            "ears": ears,
        }

        face_model = None
        selected_face_parts = [
            name for name in URUTORA_ANATOMICAL_PARTS if selections[name]
        ]
        if selected_face_parts:
            try:
                face_model = _load_session(face_model_path, providers)
            except FileNotFoundError:
                print(
                    "[HumanPartsUrutora] Face-parsing model is not installed; "
                    "Run install.py to enable native facial segmentation. "
                    "Eyes will use the legacy estimate; other fine facial "
                    "parts will remain empty rather than being guessed."
                )

        output_images = []
        output_masks = []
        for image_item in image:
            original = tensor_to_pil(image_item).convert("RGB")
            mask = _segment_parts(original, model, selections, face_model)

            if process_detail:
                detail_range = detail_erode + detail_dilate
                image_batch = image_item.unsqueeze(0)
                # Small anatomical masks are especially vulnerable to a
                # matting model erasing the seed or introducing confident
                # alpha elsewhere.
                # Retain the trimap support so refinement can improve edges but
                # cannot create disconnected anatomy far from the seed mask.
                refinement_trimap = (
                    generate_vitmatte_trimap(mask, detail_erode, detail_dilate)
                    if any(
                        selections.get(name, False)
                        for name in URUTORA_ANATOMICAL_PARTS
                    )
                    else None
                )
                if detail_method == "GuidedFilter":
                    mask = guided_filter_alpha(
                        image_batch, mask, detail_range // 6 + 1
                    )
                    mask = histogram_remap(mask, black_point, white_point)
                elif detail_method == "PyMatting":
                    mask = pymatting_alpha(
                        image_batch,
                        mask,
                        detail_range // 8 + 1,
                        black_point,
                        white_point,
                    )
                else:
                    trimap = (
                        refinement_trimap
                        or generate_vitmatte_trimap(
                            mask, detail_erode, detail_dilate
                        )
                    )
                    matte = generate_vitmatte(
                        original,
                        trimap,
                        detail_method,
                        device,
                        max_megapixels,
                    )
                    mask = histogram_remap(
                        pil_to_mask(matte), black_point, white_point
                    )
                if refinement_trimap is not None:
                    refinement_constraints = pil_to_mask(refinement_trimap)
                    support = refinement_constraints > 0
                    mask = torch.where(support, mask, torch.zeros_like(mask))
                    known_foreground = refinement_constraints == 1
                    mask = torch.where(
                        known_foreground, torch.ones_like(mask), mask
                    )

            mask = mask.float().clamp(0.0, 1.0)
            mask_image = mask_to_pil(mask)
            output_images.append(pil_to_tensor(rgba_with_mask(original, mask_image)))
            output_masks.append(mask)

        print(f"[HumanPartsUrutora] Processed {len(output_images)} image(s).")
        return io.NodeOutput(
            torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0)
        )

    @classmethod
    def execute_legacy(cls, **kwargs):
        output = cls.execute(**kwargs)
        return tuple(getattr(output, "result", output))


class LayerStyleHumanPartsUrutora(HumanPartsUrutora):
    """Compatibility registration for the original LayerStyle workflow ID."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "LayerMask: HumanPartsUrutora"
        return schema
