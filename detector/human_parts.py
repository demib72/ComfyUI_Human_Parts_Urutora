from typing import Tuple

import numpy as np
import torch
from onnxruntime import InferenceSession
from PIL import Image

# Follows CCIHP => https://kalisteo.cea.fr/wp-content/uploads/2021/09/README.html
#
# Note: I prefer to use a dictionnary to be able to change the index if needed.
labels = {
    0: ("background", "Background"),
    1: (
        "hat",
        "Hat: Hat, helmet, cap, hood, veil, headscarf, part covering the skull and hair of a hood/balaclava, crown…",
    ),
    2: (
        "hair",
        "Hair",
    ),
    3: (
        "glove",
        "Glove",
    ),
    4: (
        "glasses",
        "Sunglasses/Glasses: Sunglasses, eyewear, protective glasses…",
    ),
    5: (
        "upper_clothes",
        "UpperClothes: T-shirt, shirt, tank top, sweater under a coat, top of a dress…",
    ),
    6: (
        "face_mask",
        "Face Mask: Protective mask, surgical mask, carnival mask, facial part of a balaclava, visor of a helmet…",
    ),
    7: (
        "coat",
        "Coat: Coat, jacket worn without anything on it, vest with nothing on it, a sweater with nothing on it…",
    ),
    8: (
        "socks",
        "Socks",
    ),
    9: (
        "pants",
        "Pants: Pants, shorts, tights, leggings, swimsuit bottoms… (clothing with 2 legs)",
    ),
    10: (
        "torso-skin",
        "Torso-skin",
    ),
    11: (
        "scarf",
        "Scarf: Scarf, bow tie, tie…",
    ),
    12: (
        "skirt",
        "Skirt: Skirt, kilt, bottom of a dress…",
    ),
    13: (
        "face",
        "Face",
    ),
    14: (
        "left-arm",
        "Left-arm (naked part)",
    ),
    15: (
        "right-arm",
        "Right-arm (naked part)",
    ),
    16: (
        "left-leg",
        "Left-leg (naked part)",
    ),
    17: (
        "right-leg",
        "Right-leg (naked part)",
    ),
    18: (
        "left-shoe",
        "Left-shoe",
    ),
    19: (
        "right-shoe",
        "Right-shoe",
    ),
    20: (
        "bag",
        "Bag: Backpack, shoulder bag, fanny pack… (bag carried on oneself",
    ),
    21: (
        "",
        "Others: Jewelry, tags, bibs, belts, ribbons, pins, head decorations, headphones…",
    ),
}


def get_class_index(class_name: str) -> int:
    """
    Return the index of the class name in the model.
    """
    if class_name == "":
        return -1

    for key, value in labels.items():
        if value[0] == class_name:
            return key

    return -1


def get_mask(
    image: torch.Tensor, model: InferenceSession, rotation: float, **kwargs
) -> Tuple[torch.Tensor, int]:
    """
    Return a Tensor with the mask of the human parts in the image.

    The rotation parameter is not used for now. The idea is to propose rotation to help
    the model to detect the human parts in the image if the character is not in a casual position.
    Several tests have been done, but the model seems to fail to detect the human parts in these cases,
    and the rotation does not help.
    """

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    selected_indices = [
        class_index
        for class_name, enabled in kwargs.items()
        if enabled and (class_index := get_class_index(class_name)) != -1
    ]

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(
            "HumanParts expects an IMAGE tensor shaped [B,H,W,C]; "
            f"received {tuple(image.shape)}."
        )

    masks: list[torch.Tensor] = []
    score = 0
    center = (256, 256)

    # The ONNX export has a fixed batch dimension, so process ComfyUI batches
    # one image at a time and collect standard [H,W] masks.
    for batch_image in image:
        image_np = batch_image.detach().cpu().numpy() * 255.0
        pil_image = Image.fromarray(image_np.astype(np.uint8)).convert("RGB")
        original_size = pil_image.size
        pil_image = pil_image.resize((512, 512), Image.Resampling.BILINEAR)

        if rotation != 0:
            pil_image = pil_image.rotate(rotation, center=center)

        model_input = np.asarray(pil_image).astype(np.float32) / 127.5 - 1.0
        model_input = np.expand_dims(model_input, axis=0)
        result = model.run([output_name], {input_name: model_input})[0]
        class_map = np.asarray(result).argmax(axis=3).squeeze(0)

        if selected_indices:
            mask = np.isin(class_map, selected_indices).astype(np.uint8) * 255
        else:
            mask = np.zeros_like(class_map, dtype=np.uint8)
        score += int(np.count_nonzero(mask))

        mask_image = Image.fromarray(mask, mode="L")
        if rotation != 0:
            mask_image = mask_image.rotate(-rotation, center=center)
        mask_image = mask_image.resize(original_size, Image.Resampling.NEAREST)
        mask_array = np.asarray(mask_image).astype(np.float32) / 255.0
        masks.append(torch.from_numpy(mask_array.copy()))

    return torch.stack(masks, dim=0).to(dtype=torch.float32), score
