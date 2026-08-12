"""Lightweight ONNX face parsing for native facial-component masks."""

from __future__ import annotations

import cv2
import numpy as np
from onnxruntime import InferenceSession
from PIL import Image


FACE_PARSING_CLASSES = {
    "left_eye": 4,
    "right_eye": 5,
}

_INPUT_SIZE = (512, 512)
_INPUT_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_INPUT_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _face_component_bounds(
    class_map: np.ndarray, face_class: int
) -> list[tuple[int, int, int, int]]:
    """Return non-trivial connected face regions as exclusive bounds."""
    source = (class_map == face_class).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(source, 8)
    minimum_area = max(4, int(class_map.size * 0.00002))
    bounds = []
    for x, y, width, height, area in stats[1:count]:
        if area >= minimum_area:
            bounds.append((x, y, x + width, y + height))
    return bounds


def _square_crop_bounds(
    bounds: tuple[int, int, int, int], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Expand a face-skin box to the full head context expected by BiSeNet."""
    x0, y0, x1, y1 = bounds
    image_width, image_height = image_size
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * 1.5
    crop_x0 = max(0, int(round(center_x - side / 2.0)))
    crop_y0 = max(0, int(round(center_y - side / 2.0)))
    crop_x1 = min(image_width, int(round(center_x + side / 2.0)))
    crop_y1 = min(image_height, int(round(center_y + side / 2.0)))
    return crop_x0, crop_y0, crop_x1, crop_y1


def _parse_crop(
    crop: Image.Image,
    model: InferenceSession,
    selected_classes: tuple[int, ...],
) -> np.ndarray:
    resized = crop.convert("RGB").resize(_INPUT_SIZE, Image.Resampling.BILINEAR)
    input_array = np.asarray(resized, dtype=np.float32) / 255.0
    input_array = (input_array - _INPUT_MEAN) / _INPUT_STD
    input_array = np.expand_dims(input_array.transpose(2, 0, 1), axis=0)

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    logits = np.asarray(model.run([output_name], {input_name: input_array})[0])
    if logits.ndim != 4 or logits.shape[1] < 19:
        raise RuntimeError(
            "Unexpected face-parsing ONNX output shape "
            f"{tuple(logits.shape)}; expected [N,19,H,W]."
        )
    class_map = logits[0].argmax(axis=0)
    return np.isin(class_map, selected_classes).astype(np.uint8) * 255


def segment_face_parts(
    image: Image.Image,
    coarse_class_map: np.ndarray,
    coarse_face_class: int,
    model: InferenceSession,
    selected_classes: tuple[int, ...],
) -> np.ndarray:
    """Parse selected facial classes on CCIHP-guided face crops."""
    image_width, image_height = image.size
    map_height, map_width = coarse_class_map.shape
    component_bounds = _face_component_bounds(coarse_class_map, coarse_face_class)

    # If the coarse parser misses a close-up face, BiSeNet can still attempt the
    # complete image. This also keeps the helper useful with synthetic inputs.
    if not component_bounds:
        component_bounds = [(0, 0, map_width, map_height)]

    result = np.zeros((image_height, image_width), dtype=np.uint8)
    for map_x0, map_y0, map_x1, map_y1 in component_bounds:
        scaled = (
            int(np.floor(map_x0 * image_width / map_width)),
            int(np.floor(map_y0 * image_height / map_height)),
            int(np.ceil(map_x1 * image_width / map_width)),
            int(np.ceil(map_y1 * image_height / map_height)),
        )
        x0, y0, x1, y1 = _square_crop_bounds(scaled, image.size)
        if x1 <= x0 or y1 <= y0:
            continue
        parsed = _parse_crop(image.crop((x0, y0, x1, y1)), model, selected_classes)
        parsed_image = Image.fromarray(parsed, mode="L").resize(
            (x1 - x0, y1 - y0), Image.Resampling.NEAREST
        )
        result[y0:y1, x0:x1] = np.maximum(
            result[y0:y1, x0:x1], np.asarray(parsed_image, dtype=np.uint8)
        )
    return result
