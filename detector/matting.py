"""Mask refinement helpers adapted from ComfyUI LayerStyle Advance.

The original implementation is Copyright (c) 2024 chflame163 and is used
under the MIT License. See THIRD_PARTY_NOTICES for the complete notice.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from ..utils import models_dir


VITMATTE_REPOSITORIES = {
    "VITMatte": "hustvl/vitmatte-small-composition-1k",
    "VITMatte(local)": "hustvl/vitmatte-small-composition-1k",
    "vitmatte-base-composition-1k": "hustvl/vitmatte-base-composition-1k",
}


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a ComfyUI IMAGE tensor."""
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def pil_to_mask(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a ComfyUI MASK tensor shaped [B, H, W]."""
    array = np.asarray(image.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert one IMAGE or MASK tensor item to PIL."""
    array = image.detach().cpu().float().numpy()
    array = np.squeeze(array)
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def mask_to_pil(mask: torch.Tensor) -> Image.Image:
    return tensor_to_pil(mask).convert("L")


def rgba_with_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    red, green, blue = image.convert("RGB").split()
    return Image.merge("RGBA", (red, green, blue, mask.convert("L")))


def histogram_remap(
    mask: torch.Tensor, black_point: float, white_point: float
) -> torch.Tensor:
    black_point = min(black_point, white_point - 0.001)
    return ((mask - black_point) / (white_point - black_point)).clamp(0.0, 1.0)


def guided_filter_alpha(
    image: torch.Tensor, mask: torch.Tensor, filter_radius: int
) -> torch.Tensor:
    """Refine a mask with OpenCV's edge-aware guided filter."""
    try:
        from cv2.ximgproc import guidedFilter
    except ImportError as error:
        raise RuntimeError(
            "GuidedFilter refinement requires opencv-contrib-python."
        ) from error

    guide = image.detach().cpu().float().numpy()
    alpha = mask.detach().cpu().float().numpy()
    if guide.ndim == 4:
        guide = guide[0]
    if alpha.ndim == 3:
        alpha = alpha[0]

    radius = max(1, int(filter_radius))
    refined = guidedFilter(guide, alpha, radius, 0.0015)
    return torch.from_numpy(np.asarray(refined, dtype=np.float32)).unsqueeze(0)


def pymatting_alpha(
    image: torch.Tensor,
    mask: torch.Tensor,
    detail_range: int,
    black_point: float,
    white_point: float,
) -> torch.Tensor:
    """Estimate a soft alpha matte using PyMatting closed-form matting."""
    try:
        import cv2
        from pymatting import estimate_alpha_cf, fix_trimap
    except ImportError as error:
        raise RuntimeError(
            "PyMatting refinement requires pymatting and opencv-python."
        ) from error

    rgb = image.detach().cpu().double().numpy()
    trimap = mask.detach().cpu().double().numpy()
    if rgb.ndim == 4:
        rgb = rgb[0]
    if trimap.ndim == 3:
        trimap = trimap[0]

    if detail_range > 0:
        kernel_size = detail_range * 5 + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        trimap = cv2.GaussianBlur(trimap, (kernel_size, kernel_size), 0)

    trimap = fix_trimap(trimap, black_point, white_point)
    alpha = estimate_alpha_cf(
        rgb,
        trimap,
        laplacian_kwargs={"epsilon": 1e-6},
        cg_kwargs={"maxiter": 500},
    )
    return torch.from_numpy(np.asarray(alpha, dtype=np.float32)).unsqueeze(0)


def generate_vitmatte_trimap(
    mask: torch.Tensor, erode_kernel_size: int, dilate_kernel_size: int
) -> Image.Image:
    """Generate the three-level trimap expected by VITMatte."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("VITMatte refinement requires opencv-python.") from error

    mask_array = np.asarray(mask_to_pil(mask), dtype=np.uint8)
    erode_kernel = np.ones(
        (max(1, erode_kernel_size), max(1, erode_kernel_size)), np.uint8
    )
    dilate_kernel = np.ones(
        (max(1, dilate_kernel_size), max(1, dilate_kernel_size)), np.uint8
    )
    eroded = cv2.erode(mask_array, erode_kernel, iterations=5)
    dilated = cv2.dilate(mask_array, dilate_kernel, iterations=5)

    trimap = np.zeros_like(mask_array)
    trimap[dilated == 255] = 128
    trimap[eroded == 255] = 255
    return Image.fromarray(trimap, mode="L")


def _vitmatte_directory(repository: str) -> str:
    return os.path.join(models_dir, "vitmatte", repository.rsplit("/", 1)[-1])


@lru_cache(maxsize=2)
def _load_vitmatte(repository: str, local_files_only: bool):
    try:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
    except ImportError as error:
        raise RuntimeError(
            "VITMatte refinement requires transformers>=4.45.0."
        ) from error

    model_directory = _vitmatte_directory(repository)
    if not os.path.isdir(model_directory):
        if local_files_only:
            raise FileNotFoundError(
                f"VITMatte model not found at {model_directory}. "
                "Run VITMatte once before selecting VITMatte(local)."
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "Downloading VITMatte requires huggingface-hub."
            ) from error
        snapshot_download(
            repo_id=repository,
            local_dir=model_directory,
            ignore_patterns=["*.md", "*.txt", "onnx", ".git"],
        )

    processor = VitMatteImageProcessor.from_pretrained(
        model_directory, local_files_only=True
    )
    model = VitMatteForImageMatting.from_pretrained(
        model_directory, local_files_only=True
    )
    model.eval()
    return model, processor


def generate_vitmatte(
    image: Image.Image,
    trimap: Image.Image,
    method: str,
    device: str,
    max_megapixels: float,
) -> Image.Image:
    """Generate a refined alpha matte using a small or base VITMatte model."""
    repository = VITMATTE_REPOSITORIES[method]
    local_files_only = method == "VITMatte(local)"
    model, processor = _load_vitmatte(repository, local_files_only)

    image = image.convert("RGB")
    trimap = trimap.convert("L")
    original_size = image.size
    maximum_pixels = max_megapixels * 1_048_576
    width, height = original_size
    if width * height > maximum_pixels:
        ratio = width / height
        target_width = max(1, int(math.sqrt(ratio * maximum_pixels)))
        target_height = max(1, int(target_width / ratio))
        image = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
        trimap = trimap.resize(
            (target_width, target_height), Image.Resampling.NEAREST
        )

    requested_device = device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
        print("[HumanPartsUltra] CUDA is unavailable; using CPU for VITMatte.")
    torch_device = torch.device(requested_device)
    model.to(torch_device)

    inputs = processor(images=image, trimaps=trimap, return_tensors="pt")
    inputs = {name: value.to(torch_device) for name, value in inputs.items()}
    with torch.inference_mode():
        prediction = model(**inputs).alphas

    matte = mask_to_pil(prediction).crop((0, 0, image.width, image.height))
    if matte.size != original_size:
        matte = matte.resize(original_size, Image.Resampling.BILINEAR)
    return matte
