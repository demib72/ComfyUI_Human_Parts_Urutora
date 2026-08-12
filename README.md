# Human Parts Urutora

Detect human parts using the DeepLabV3+ ResNet50 model from Keras-io. You can extract hair, arms, legs, and other parts
with ease and with small memory usage.

This node aims to detect human parts using the model created by
[Keras-io](https://huggingface.co/keras-io/deeplabv3p-resnet50). Their "[Space](https://huggingface.co/spaces/keras-io/Human-Part-Segmentation)" was impressive, and I wanted to use the
model.

Unfortunately, the model uses an old Keras version, and there were no PyTorch implementation.

So I decided to convert the model to [ONNX](https://onnx.ai/) format and to create my [HugginFace
repository](https://huggingface.co/Metal3d/deeplabv3p-resnet50-human) to share the model with the community.

> Fortunately, Keras provides the model with a CC1.0 license, thank you guys to allow us to use it without any
> restriction.

## Example

You can drag and drop the following image to try:

![Example workflow](./images/HumanPartsWorkflow.png)

## DeepLabV3+ ResNet50 for Human

Actually, all the model I found was not trained to detect human parts, but to detect some objects or urban elements. The
Keras model is the only one I found that works!

## Installation

I strongly recommend to use ComfyUI-Manager to install the node. It will install the dependencies and the model.

> Note, as far as my repository isn't validated in the ComfyUI-Manager index, you must do the installation manually.
>
> If you set up ComfyUI-Manager to "middle" or "weak" security, you can use the "Install from Git URL" feature.

```bash
# ensure that you have activated the virtual environment before !!

# then...
cd /path/to/your/ComfyUI/custom_nodes
git clone https://github.com/metal3d/ComfyUI_Human_Parts.git
cd ComfyUI_Human_Parts
pip install -r requirements.txt
# or
python -m pip install -r requirements.txt

# install the model
python install.py
```

Use the same Python environment that starts your local ComfyUI installation
for dependency installation, model installation, and tests. For example, from
this repository when ComfyUI is installed beside it:

```bash
../ComfyUI/venv/bin/python -m pip install -r requirements.txt
../ComfyUI/venv/bin/python install.py
../ComfyUI/venv/bin/python -m unittest discover -s tests -v
```

Then, restart ComfyUI, refresh the UI, and you may find the "Human Parts
Urutora mask generator" node.

![The node](./images/node.png)

## Human Parts Ultra

This fork also includes a standalone port of LayerStyle Advance's
`LayerMask: HumanPartsUltra` node. Existing workflows using that exact node
identifier can load without installing the complete LayerStyle Advance node
suite.

Human Parts Ultra was originally implemented by
[chflame163](https://github.com/chflame163) as part of
[ComfyUI LayerStyle Advance](https://github.com/chflame163/ComfyUI_LayerStyle_Advance),
building on Metal3d's original Human Parts node. The port is used under the
MIT License with its copyright and permission notice preserved in
[THIRD_PARTY_NOTICES](./THIRD_PARTY_NOTICES).

In addition to the ONNX human-parts segmentation, Human Parts Ultra provides:

- Batch processing.
- An RGBA image output whose alpha channel contains the selected mask.
- A standard ComfyUI `MASK` output.
- Optional VITMatte, PyMatting, or Torch-native Guided Filter edge refinement
  without OpenCV contrib/ximgproc.
- CPU or CUDA selection for VITMatte.

VITMatte models are downloaded from Hugging Face into
`ComfyUI/models/vitmatte` on first use. Select `VITMatte(local)` to prohibit a
model download and use files already present in that directory. The original
ONNX segmentation model is still installed with `python install.py`.

The port fixes the original Ultra node's left-foot selection bug. ONNX uses an
explicit `auto` provider policy: it tries advertised TensorRT, CUDA, and CPU
providers in that order, and retries with progressively safer provider chains
if an accelerated provider cannot initialize. Both Human Parts nodes share this
behavior.

To choose a starting provider explicitly, set
`COMFYUI_HUMAN_PARTS_ONNX_PROVIDER` to `tensorrt`, `cuda`, or `cpu` before
starting ComfyUI. The default is `auto`; explicit accelerated policies still
retain lower-priority providers as execution fallbacks. If an explicitly
selected provider is not installed, the node reports the available providers
and how to return to `auto`.
See [THIRD_PARTY_NOTICES](./THIRD_PARTY_NOTICES) for attribution and licensing.
