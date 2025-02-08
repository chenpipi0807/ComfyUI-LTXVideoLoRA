# ComfyUI-LTXVideoLoRA
A set of custom nodes enabling LoRA support for LTX Video in ComfyUI.

### 07.02.2025 ⭐ NEW ⭐

- Add LoRA support as a individual  **LTXV LoRA Loader** node
- Add LoRA loader nodes that can be chained (same as **ComfyUI-HunyuanVideoWrapper**)

The main code is inspired by Lightricks **ComfyUI-LTXVideo** ([here](https://github.com/Lightricks/ComfyUI-LTXVideo)) and the LoRA nodes are based on the kijai's **ComfyUI-HunyuanVideoWrapper** ([here](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper)).

## Installation

Installation via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) is preferred. Simply search for `ComfyUI-LTXVideoLoRA` in the list of nodes and follow installation instructions.

#### Manual installation

1. Install ComfyUI
2. Clone this repository to `custom-nodes` folder in your ComfyUI installation directory.
3. Install the required packages:
```bash
cd custom_nodes/ComfyUI-LTXVideoLoRA && pip install -r requirements.txt
```
#### For portable ComfyUI installations, run
```
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-LTXVideoLoRA\requirements.txt
```

## Example workflows

#### Text-to-video with LoRA support