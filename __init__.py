from .nodes.lora_loader_node import LTXVLoRALoader

NODE_CLASS_MAPPINGS = {
    "LTXVLoRALoader": LTXVLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVLoRALoader": "LTXV LoRA Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]