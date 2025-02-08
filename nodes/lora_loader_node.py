import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def standardize_lora_key_format(lora_sd):
    new_sd = {}
    for k, v in lora_sd.items():
        k = 'diffusion_model.' + k
        new_sd[k] = v
    return new_sd

class LTXVLoRALoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip", "The LTXV model to apply the LoRA chain."},
                )
            },
            "optional": {
                "lora": (
                    "LORA", 
                    {"default": None}
                ),
            }
        }
    
    FUNCTION = "lora_loader"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "LTXVideoLoRA"
    TITLE = "LTXV LoRA Loader"

    def lora_loader(self, model, lora):

        if lora is not None:
            from comfy.sd import load_lora_for_models
            from comfy.utils import load_torch_file
            for l in lora:
                log.info(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
                lora_path = l["path"]
                lora_strength = l["strength"]
                lora_sd = load_torch_file(lora_path, safe_load=True)
                lora_sd = standardize_lora_key_format(lora_sd)
                model, _ = load_lora_for_models(model, None, lora_sd, lora_strength, 0)

        return model