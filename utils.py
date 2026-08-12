import os
from pathlib import Path


def _fallback_models_dir(plugin_dir: Path) -> str:
    """Locate ComfyUI/models when folder_paths is not importable.

    Supports a development checkout beside ComfyUI and a normal installation
    below ComfyUI/custom_nodes. The final fallback remains local to the plugin
    parent instead of accidentally walking into the user's home directory.
    """
    candidates = [
        plugin_dir.parent / "ComfyUI" / "models",
    ]
    if plugin_dir.parent.name == "custom_nodes":
        candidates.insert(0, plugin_dir.parent.parent / "models")

    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return str(candidates[0])

# get the model paths
try:
    from folder_paths import models_dir  # pyright: ignore
except ImportError:
    models_dir = _fallback_models_dir(Path(__file__).resolve().parent)

models_dir_path = os.path.join(models_dir, "onnx", "human-parts")
model_url = "https://huggingface.co/Metal3d/deeplabv3p-resnet50-human/resolve/main/deeplabv3p-resnet50-human.onnx"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir_path, "deeplabv3p-resnet50-human.onnx")

# Lightweight BiSeNet face parser used for native facial-component masks.
# Source: https://github.com/yakhyo/face-parsing
face_model_url = (
    "https://github.com/yakhyo/face-parsing/releases/download/weights/"
    "resnet18.onnx"
)
face_model_name = "face-parsing-resnet18.onnx"
face_model_path = os.path.join(models_dir_path, face_model_name)
