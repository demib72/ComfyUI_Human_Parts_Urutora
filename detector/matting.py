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
import torch.nn.functional as torch_functional
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
    """Refine a mask with a device-preserving Torch guided filter.

    ComfyUI IMAGE tensors use ``[B, H, W, C]`` while MASK tensors use
    ``[B, H, W]``. The scalar guided-filter formulation uses luminance as its
    guide, which avoids the optional OpenCV ximgproc module and works on the
    same CPU, CUDA, or MPS device as the input tensors.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError("GuidedFilter image must have shape [B, H, W, C].")

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError("GuidedFilter mask must have shape [B, H, W].")
    if image.shape[:3] != mask.shape:
        raise ValueError("GuidedFilter image and mask dimensions must match.")

    # Guided-filter statistics are susceptible to precision loss in float16.
    # ComfyUI masks are conventionally returned as float32, matching the old
    # OpenCV implementation's output type.
    guide = image.detach().to(dtype=torch.float32)
    alpha = mask.detach().to(device=guide.device, dtype=torch.float32)
    channels = guide.shape[-1]
    if channels >= 3:
        luminance_weights = guide.new_tensor((0.2126, 0.7152, 0.0722))
        guide = (guide[..., :3] * luminance_weights).sum(dim=-1)
    elif channels == 1:
        guide = guide[..., 0]
    else:
        raise ValueError("GuidedFilter image must have at least one channel.")

    guide = guide.unsqueeze(1)
    alpha = alpha.unsqueeze(1)
    radius = max(1, int(filter_radius))
    kernel_size = radius * 2 + 1

    def box_mean(value: torch.Tensor) -> torch.Tensor:
        # An integral image makes the box filter O(H*W), independent of the
        # requested radius. This matters for large masks and keeps the Torch
        # fallback competitive with OpenCV's optimized box filter.
        padded = torch_functional.pad(
            value, (radius, radius, radius, radius), mode="replicate"
        )
        integral = padded.cumsum(dim=-1).cumsum(dim=-2)
        integral = torch_functional.pad(integral, (1, 0, 1, 0))
        window_sum = (
            integral[..., kernel_size:, kernel_size:]
            - integral[..., :-kernel_size, kernel_size:]
            - integral[..., kernel_size:, :-kernel_size]
            + integral[..., :-kernel_size, :-kernel_size]
        )
        return window_sum / float(kernel_size * kernel_size)

    statistics = box_mean(
        torch.cat((guide, alpha, guide * alpha, guide * guide), dim=1)
    )
    mean_guide, mean_alpha, correlation, second_moment = statistics.split(
        1, dim=1
    )
    covariance = correlation - mean_guide * mean_alpha
    variance = second_moment - mean_guide * mean_guide

    coefficient = covariance / (variance.clamp_min(0.0) + 0.0015)
    intercept = mean_alpha - coefficient * mean_guide
    mean_coefficient, mean_intercept = box_mean(
        torch.cat((coefficient, intercept), dim=1)
    ).split(1, dim=1)
    refined = mean_coefficient * guide + mean_intercept
    return refined.squeeze(1).clamp(0.0, 1.0)


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


def _resolve_torch_device(device: str) -> torch.device:
    """Resolve an explicit device or defer to ComfyUI when requested."""
    if device == "auto":
        try:
            import comfy.model_management as model_management

            return torch.device(model_management.get_torch_device())
        except ImportError:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda" and not torch.cuda.is_available():
        print("[HumanPartsUltra] CUDA is unavailable; using CPU for VITMatte.")
        return torch.device("cpu")
    return torch.device(device)


def _prepare_vitmatte_device(model, device: torch.device) -> None:
    """Give ComfyUI a chance to offload managed models before allocation."""
    if device.type == "cpu":
        return
    try:
        import comfy.model_management as model_management

        if hasattr(model_management, "free_memory") and hasattr(
            model_management, "module_size"
        ):
            model_management.free_memory(model_management.module_size(model), device)
    except ImportError:
        # The helpers are also usable in tests and outside a full ComfyUI install.
        pass


def _empty_device_cache(device: torch.device) -> None:
    """Release allocator caches through ComfyUI, with a standalone fallback."""
    if device.type == "cpu":
        return
    try:
        import comfy.model_management as model_management

        if hasattr(model_management, "soft_empty_cache"):
            model_management.soft_empty_cache()
            return
    except ImportError:
        pass

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()


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

    torch_device = _resolve_torch_device(device)
    try:
        _prepare_vitmatte_device(model, torch_device)
        model.to(torch_device)
        inputs = processor(images=image, trimaps=trimap, return_tensors="pt")
        inputs = {name: value.to(torch_device) for name, value in inputs.items()}
        with torch.inference_mode():
            prediction = model(**inputs).alphas.detach().cpu()
    finally:
        # Transformer models are large enough that retaining even CPU copies is
        # surprising. Always offload before dropping both LRU entries, including
        # when preprocessing or inference raises.
        try:
            if torch_device.type != "cpu":
                model.to(torch.device("cpu"))
        finally:
            _load_vitmatte.cache_clear()
            _empty_device_cache(torch_device)

    matte = mask_to_pil(prediction).crop((0, 0, image.width, image.height))
    if matte.size != original_size:
        matte = matte.resize(original_size, Image.Resampling.BILINEAR)
    return matte
