# ComfyUI-LTXVideoLoRA
A set of custom nodes enabling LoRA support for LTX Video in ComfyUI.

### 08.02.2025 ⭐ NEW ⭐

- Add LoRA support as a individual  **LTXV LoRA Loader** node > for Lightricks **ComfyUI-LTXVideo**
- Add LoRA support inside a **LTXV Checkpoint Loader with LoRA** node > for log(td) **ComfyUI-LTXTricks**
- Add LoRA selector node that can be chained using multiple **LTXV LoRA Selector**

The purpose of theses node are to enable using the **a-r-r-o-w's finetrainers** ([here](https://github.com/a-r-r-o-w/finetrainers)) LTXV LoRA directly inside ComfyUI.

The main code is inspired by:
- comfyanonymous **ComfyUI** ([here](https://github.com/comfyanonymous/ComfyUI))
- Lightricks **ComfyUI-LTXVideo** ([here](https://github.com/Lightricks/ComfyUI-LTXVideo)) 
- log(td) **ComfyUI-LTXTricks** ([here](https://github.com/logtd/ComfyUI-LTXTricks))
- kijai **ComfyUI-HunyuanVideoWrapper** ([here](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper)) for the LoRA Selector / Block Edit nodes.

## Installation

#### Installation via ComfyUI-Manager

Installation via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) is preferred. Simply search for `ComfyUI-LTXVideoLoRA` in the list of nodes.

#### Manual installation

Simply clone this repository to `custom-nodes` folder in your ComfyUI installation directory.

## Example workflows

#### Text-to-video with LoRA support

[Download workflow](assets/LTXV-T2V-LoRA-Workflow)
![workflow](assets/LTXV-T2V-LoRA-Workflow.mp4)