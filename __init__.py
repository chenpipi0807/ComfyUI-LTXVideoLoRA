from .nodes.lora_loader_node import LTXVLoRALoader
from .nodes.lora_selector_node import LTXVLoRASelector
from .nodes.lora_block_node import LTXVLoRABlockEdit
from .nodes.wan21_lora_adapter_node import LTXVWan21LoRASelector, LTXVWan21LoRALoader

NODE_CLASS_MAPPINGS = {
    "LTXVLoRALoader": LTXVLoRALoader,
    "LTXVLoRASelector": LTXVLoRASelector,
    "LTXVLoRABlockEdit": LTXVLoRABlockEdit,
    "LTXVWan21LoRASelector": LTXVWan21LoRASelector,
    "LTXVWan21LoRALoader": LTXVWan21LoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVLoRALoader": "LTXV LoRA Loader",
    "LTXVLoRASelector": "LTXV LoRA Selector",
    "LTXVLoRABlockEdit": "LTXV LoRA Block Edit",
    "LTXVWan21LoRASelector": "LTXV Wan2.1 LoRA Adapter",
    "LTXVWan21LoRALoader": "LTXV Wan2.1 LoRA Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]