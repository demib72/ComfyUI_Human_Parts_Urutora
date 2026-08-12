"""Human Parts Ultra node adapted from ComfyUI LayerStyle Advance.

The original implementation is Copyright (c) 2024 chflame163 and is used
under the MIT License. See THIRD_PARTY_NOTICES for the complete notice.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

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
from .utils import model_path


ULTRA_CLASSES = {
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


def _execution_providers() -> tuple[str, ...]:
    available = set(ort.get_available_providers())
    preferred = (
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    providers = tuple(provider for provider in preferred if provider in available)
    return providers or tuple(ort.get_available_providers())


@lru_cache(maxsize=1)
def _load_session(path: str, providers: tuple[str, ...]) -> ort.InferenceSession:
    """Load the ONNX model, retrying with less specialized providers.

    ONNX Runtime may advertise a provider even when it cannot initialize it on
    the current machine. Keep provider selection deterministic, but fall back
    through the available providers so CPU-only and partially configured GPU
    installations can still run the node.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Human Parts ONNX model not found at {path}. Run install.py or "
            "place deeplabv3p-resnet50-human.onnx in that directory."
        )

    attempts: list[tuple[str, ...]] = []
    if providers:
        attempts.append(providers)
        attempts.extend((provider,) for provider in providers)
    else:
        attempts.append(())

    errors: list[tuple[tuple[str, ...], Exception]] = []
    seen: set[tuple[str, ...]] = set()
    for provider_attempt in attempts:
        if provider_attempt in seen:
            continue
        seen.add(provider_attempt)
        try:
            if provider_attempt:
                return ort.InferenceSession(path, providers=list(provider_attempt))
            return ort.InferenceSession(path)
        except Exception as error:  # ONNX Runtime uses several exception types.
            errors.append((provider_attempt, error))

    attempted = ", ".join(
        "+".join(provider_attempt) or "ONNX Runtime default"
        for provider_attempt, _ in errors
    )
    raise RuntimeError(
        f"Unable to load Human Parts ONNX model at {path}. "
        f"Provider attempts: {attempted}. Last error: {errors[-1][1]}"
    ) from errors[-1][1]


def _segment_parts(
    image: Image.Image,
    model: ort.InferenceSession,
    selections: dict[str, bool],
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
        ULTRA_CLASSES[name]
        for name, enabled in selections.items()
        if enabled and name in ULTRA_CLASSES
    ]
    if selected_indices:
        mask = np.isin(class_map, selected_indices).astype(np.uint8) * 255
    else:
        mask = np.zeros_like(class_map, dtype=np.uint8)

    mask_image = Image.fromarray(mask, mode="L").resize(
        original_size, Image.Resampling.NEAREST
    )
    return pil_to_mask(mask_image)


class HumanPartsUltra:
    """Generate and optionally refine masks for selected human parts."""

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "human_parts_ultra"
    CATEGORY = "Human Parts"

    @classmethod
    def INPUT_TYPES(cls):
        methods = list(VITMATTE_REPOSITORIES) + ["PyMatting", "GuidedFilter"]
        return {
            "required": {
                "image": ("IMAGE",),
                "face": ("BOOLEAN", {"default": False}),
                "hair": ("BOOLEAN", {"default": False}),
                "glasses": ("BOOLEAN", {"default": False}),
                "top_clothes": ("BOOLEAN", {"default": False}),
                "bottom_clothes": ("BOOLEAN", {"default": False}),
                "torso_skin": ("BOOLEAN", {"default": False}),
                "left_arm": ("BOOLEAN", {"default": False}),
                "right_arm": ("BOOLEAN", {"default": False}),
                "left_leg": ("BOOLEAN", {"default": False}),
                "right_leg": ("BOOLEAN", {"default": False}),
                "left_foot": ("BOOLEAN", {"default": False}),
                "right_foot": ("BOOLEAN", {"default": False}),
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
                        "display": "slider",
                    },
                ),
                "white_point": (
                    "FLOAT",
                    {
                        "default": 0.99,
                        "min": 0.02,
                        "max": 0.99,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "process_detail": ("BOOLEAN", {"default": True}),
                # Keep the legacy choices in place for saved workflows while
                # making ComfyUI-managed device selection the new default.
                "device": (
                    ["cuda", "cpu", "auto"],
                    {"default": "auto"},
                ),
                "max_megapixels": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 999.0, "step": 0.1},
                ),
            }
        }

    def human_parts_ultra(
        self,
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
    ):
        model = _load_session(model_path, _execution_providers())
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
        }

        output_images = []
        output_masks = []
        for image_item in image:
            original = tensor_to_pil(image_item).convert("RGB")
            mask = _segment_parts(original, model, selections)

            if process_detail:
                detail_range = detail_erode + detail_dilate
                image_batch = image_item.unsqueeze(0)
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
                    trimap = generate_vitmatte_trimap(
                        mask, detail_erode, detail_dilate
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

            mask = mask.float().clamp(0.0, 1.0)
            mask_image = mask_to_pil(mask)
            output_images.append(pil_to_tensor(rgba_with_mask(original, mask_image)))
            output_masks.append(mask)

        print(f"[HumanPartsUltra] Processed {len(output_images)} image(s).")
        return (torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0))
