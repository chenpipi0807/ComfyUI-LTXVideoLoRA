import logging
import torch

from comfy import clip_vision
from comfy import model_detection
from comfy import model_management
from comfy import model_patcher

from comfy.sd import VAE, CLIP

import comfy.utils
import folder_paths

def standardize_lora_key_format(lora_sd):
    new_sd = {}
    for k, v in lora_sd.items():
        # Diffusers format
        if k.startswith('transformer.'):
            k = k.replace('transformer.', 'diffusion_model.')
        new_sd[k] = v
    return new_sd

def load_state_dict_guess_config_with_lora(sd, lora, model_options={}):
    vae = None
    model = None
    model_patcher = None

    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    parameters = comfy.utils.calculate_parameters(sd, diffusion_model_prefix)
    weight_dtype = comfy.utils.weight_dtype(sd, diffusion_model_prefix)
    load_device = model_management.get_torch_device()

    model_config = model_detection.model_config_from_unet(sd, diffusion_model_prefix)
    if model_config is None:
        return None

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if weight_dtype is not None and model_config.scaled_fp8 is None:
        unet_weight_dtype.append(weight_dtype)

    model_config.custom_operations = model_options.get("custom_operations", None)
    unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))

    if unet_dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype)

    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
    model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
    model.load_model_weights(sd, diffusion_model_prefix)

    vae_sd = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
    vae_sd = model_config.process_vae_state_dict(vae_sd)
    vae = VAE(sd=vae_sd)

    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device())
    if inital_load_device != torch.device("cpu"):
        logging.info("loaded diffusion model directly to GPU")
        model_management.load_models_gpu([model_patcher], force_full_load=True)

    if lora is not None:
        from comfy.sd import load_lora_for_models
        from comfy.utils import load_torch_file
        for l in lora:
            logging.info(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
            lora_path = l["path"]
            lora_strength = l["strength"]
            lora_sd = load_torch_file(lora_path, safe_load=True)
            lora_sd = standardize_lora_key_format(lora_sd)
            model_patcher, _ = load_lora_for_models(model_patcher, None, lora_sd, lora_strength, 0)

    return (model_patcher, vae)

def load_checkpoint_guess_config_with_lora(ckpt_path, lora):
    sd = comfy.utils.load_torch_file(ckpt_path)
    out = load_state_dict_guess_config_with_lora(sd, lora)
    if out is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))
    return out

class LTXVCheckpointLoaderLoRA:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            },
            "optional": {
                "lora": ("LTXVLORA", {"default": None}),
            }
        }
    RETURN_TYPES = ("MODEL", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint_with_lora"

    CATEGORY = "LTXVideoLoRA"
    DESCRIPTION = "Loads a diffusion model checkpoint with lora, diffusion models are used to denoise latents."

    def load_checkpoint_with_lora(self, ckpt_name, lora):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = load_checkpoint_guess_config_with_lora(ckpt_path, lora)
        return out[:2]