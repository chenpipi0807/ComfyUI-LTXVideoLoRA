import folder_paths
import logging
import re
import torch
import os
import json
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# 基于研究的Wan2.1与LTXV的架构映射关系
WAN21_TO_LTXV_KEY_MAPPING = {
    # 基本块结构映射
    r'lora_unet__blocks_(\d+)_': r'transformer.blocks.\1.',
    
    # 自注意力组件映射
    r'self_attn_k': r'attn1.k_proj',
    r'self_attn_q': r'attn1.q_proj',
    r'self_attn_v': r'attn1.v_proj',
    r'self_attn_o': r'attn1.o_proj',
    
    # 交叉注意力组件映射
    r'cross_attn_k': r'attn2.k_proj',
    r'cross_attn_q': r'attn2.q_proj',
    r'cross_attn_v': r'attn2.v_proj',
    r'cross_attn_o': r'attn2.o_proj',
    
    # 图像相关交叉注意力映射
    r'cross_attn_k_img': r'attn2.k_proj_img',
    r'cross_attn_q_img': r'attn2.q_proj_img',
    r'cross_attn_v_img': r'attn2.v_proj_img',
    r'cross_attn_o_img': r'attn2.o_proj_img',
    
    # 前馈网络映射
    r'ffn_0': r'mlp.layers.0',
    r'ffn_1': r'mlp.layers.1',
    r'ffn_2': r'mlp.layers.2',
    r'ffn_3': r'mlp.layers.3',
    
    # 特殊层映射
    r'lora_unet__head_head': r'transformer.output_blocks',
    r'lora_unet__img_emb_proj_(\d+)': r'transformer.image_projection.\1',
    r'lora_unet__text_embedding_(\d+)': r'transformer.text_embedding.\1',
    r'lora_unet__time_embedding_(\d+)': r'transformer.time_embedding.\1',
    r'lora_unet__time_projection_(\d+)': r'transformer.time_projection.\1',
}

# 提供辅助映射规则以提高适配率
AUXILIARY_MAPPINGS = [
    # 如果无法直接映射，尝试这些替代映射
    {'from': r'lora_unet__', 'to': r'transformer.'},
    {'from': r'lora_', 'to': r''},
    {'from': r'_img', 'to': r'_image'},
    {'from': r'self_attn', 'to': r'attn1'},
    {'from': r'cross_attn', 'to': r'attn2'},
]

def map_wan21_to_ltxv_keys(lora_sd, ltxv_loader_as_input=False, show_details=True):
    """将Wan2.1系列LoRA的键名映射到LTXV模型可接受的键名格式
    
    基于深入研究的Wan2.1和LTXV模型架构差异，使用多级映射策略
    """
    new_sd = {}
    unmapped_keys = []
    mapped_keys = []
    layer_stats = defaultdict(int)  # 跟踪不同类型层的映射统计
    
    # 打印原始键名组成的分析
    if show_details:
        log.info(f"LoRA文件包含 {len(lora_sd)} 个键")
        key_prefixes = defaultdict(int)
        for k in lora_sd.keys():
            # 提取前缀模式
            prefix = re.match(r'(lora_unet__[^_]+)', k)
            if prefix:
                key_prefixes[prefix.group(1)] += 1
            elif k.startswith('lora_'):
                key_prefixes['lora_other'] += 1
            else:
                key_prefixes['unknown'] += 1
        
        log.info("LoRA键名前缀分析:")
        for prefix, count in sorted(key_prefixes.items(), key=lambda x: x[1], reverse=True)[:10]:
            log.info(f"  {prefix}: {count} 个键")
    
    # 多轮映射尝试
    for k, v in lora_sd.items():
        original_key = k
        mapped = False
        
        # 第一轮: 尝试直接映射
        for pattern, replacement in WAN21_TO_LTXV_KEY_MAPPING.items():
            new_k = re.sub(pattern, replacement, k)
            if new_k != k:  # 如果匹配到并替换了
                k = new_k
                mapped = True
                # 跟踪映射了哪种类型的层
                for layer_type in ['self_attn', 'cross_attn', 'ffn', 'head', 'emb', 'proj']:
                    if layer_type in pattern:
                        layer_stats[layer_type] += 1
                        break
        
        # 第二轮: 如果不能直接映射，尝试辅助映射
        if not mapped:
            # 保存原始键，以便当所有辅助映射完成后再检查
            intermediate_key = k
            for aux_map in AUXILIARY_MAPPINGS:
                k = re.sub(aux_map['from'], aux_map['to'], k)
            
            # 如果辅助映射改变了键
            if k != intermediate_key:
                mapped = True
                layer_stats['auxiliary'] += 1
        
        # 处理LoRA特定后缀，确保保留必要的结构
        lora_parts = re.match(r'(.+?)\.(alpha|lora_up\.weight|lora_down\.weight)$', k)
        if lora_parts:
            base_key = lora_parts.group(1)
            suffix = lora_parts.group(2)
            # 保留LoRA特定的后缀结构
            k = f"{base_key}.{suffix}"
        
        # 处理前缀
        if ltxv_loader_as_input:
            # 如果使用官方LTXV Loader，确保存在diffusion_model前缀
            if not k.startswith('diffusion_model.'):
                k = 'diffusion_model.' + k
        else:
            # 其他情况下的前缀处理
            if not k.startswith('transformer.') and not k.startswith('diffusion_model.'):
                if 'transformer.' in k:
                    # 如果已有transformer但不在开头
                    parts = k.split('transformer.')
                    k = 'transformer.' + parts[-1]
                else:
                    # 添加diffusion_model前缀
                    k = 'diffusion_model.' + k
        
        # 记录映射情况
        if original_key == k:
            unmapped_keys.append(original_key)
        else:
            mapped_keys.append((original_key, k))
        
        new_sd[k] = v
    
    # 输出详细的映射报告
    if show_details:
        log.info(f"Wan2.1 LoRA适配统计:")
        log.info(f"  总计: {len(lora_sd)} 个键中成功映射 {len(mapped_keys)} 个，适配率 {len(mapped_keys)/len(lora_sd)*100:.1f}%")
        log.info(f"  未映射: {len(unmapped_keys)} 个键")
        
        # 按层类型分类报告
        log.info("  各类型层映射统计:")
        for layer_type, count in sorted(layer_stats.items(), key=lambda x: x[1], reverse=True):
            log.info(f"    {layer_type}: {count} 个键")
        
        # 打印映射样例
        if mapped_keys:
            log.info("映射样例:")
            for i, (orig, new_k) in enumerate(mapped_keys[:5]):
                log.info(f"  {orig} -> {new_k}")
            if len(mapped_keys) > 5:
                log.info(f"  ... 以及其他 {len(mapped_keys) - 5} 个映射")
        
        # 打印未映射样例
        if unmapped_keys:
            log.info("未映射键名样例:")
            for k in unmapped_keys[:5]:
                log.info(f"  {k}")
            if len(unmapped_keys) > 5:
                log.info(f"  ... 以及其他 {len(unmapped_keys) - 5} 个未映射键")
    
    return new_sd

class LTXVWan21LoRASelector:
    """专门为Wan2.1系列LoRA设计的选择器节点，解决与LTXV架构不匹配问题"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora": (folder_paths.get_filename_list("loras"), {
                   "tooltip": "选择Wan2.1系列LoRA模型，如Fun-Reward-LoRAs"
               }),
                "strength": ("FLOAT", {
                    "default": 0.6, 
                    "min": -10.0, 
                    "max": 10.0, 
                    "step": 0.0001, 
                    "tooltip": "LoRA强度，建议为Wan2.1系列LoRA使用0.5-0.7的强度"
                }),
                "force_wan21_mode": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "强制将LoRA当作Wan2.1系列处理，即使名称中不包含Wan2.1"
                }),
                "show_mapping_details": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "显示详细的键名映射信息，帮助调试"
                }),
            },
            "optional": {
                "previous_lora": ("LTXVLORA", {
                    "default": None, 
                    "tooltip": "可链接多个LoRA以应用多个效果"
                }),
                "blocks": ("SELECTEDBLOCKS", {
                    "tooltip": "指定要应用LoRA的模型块"
                }),
                "custom_mapping_rules": ("STRING", {
                    "default": "", 
                    "multiline": True, 
                    "tooltip": "可选的自定义映射规则，格式为JSON: {'from': 'pattern', 'to': 'replacement'}"
                }),
            }
        }
    
    RETURN_TYPES = ("LTXVLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "get_lora_path"
    CATEGORY = "LTXVideoLoRA"
    TITLE = "LTXV Wan2.1 LoRA Adapter"
    DESCRIPTION = "专门适配Wan2.1系列LoRA (如Fun-Reward-LoRAs)到LTXV模型架构"
    
    def get_lora_path(self, lora, strength, force_wan21_mode=False, show_mapping_details=True, 
                      blocks=None, previous_lora=None, custom_mapping_rules=""):
        loras_list = []
        
        # 检查是否为Wan2.1系列LoRA
        is_wan21 = force_wan21_mode or "Wan2.1" in lora or "wan2.1" in lora.lower() or "Fun-Reward" in lora
        
        # 处理自定义映射规则
        custom_rules = []
        if custom_mapping_rules and custom_mapping_rules.strip():
            try:
                # 尝试解析为JSON格式的规则
                if custom_mapping_rules.startswith('['):
                    custom_rules = json.loads(custom_mapping_rules)
                else:
                    # 如果用户输入的不是数组格式，尝试解析为单个规则
                    rule = json.loads(custom_mapping_rules)
                    custom_rules = [rule]
                
                log.info(f"加载了 {len(custom_rules)} 条自定义映射规则")
            except json.JSONDecodeError as e:
                log.warning(f"解析自定义映射规则时出错: {e}")
        
        if is_wan21:
            log.info(f"检测到Wan2.1系列LoRA: {lora}")
            if force_wan21_mode:
                log.info("强制Wan2.1模式已启用")
        
        # 获取LoRA路径
        lora_path = folder_paths.get_full_path("loras", lora)
        if lora_path is None:
            log.error(f"LoRA文件未找到: {lora}")
            return ([],) if not previous_lora else (previous_lora,)
            
        lora_info = {
            "path": lora_path,
            "strength": strength,
            "name": lora.split(".")[0],
            "blocks": blocks,
            "is_wan21": is_wan21,  # 标记为Wan2.1 LoRA
            "show_mapping_details": show_mapping_details,
            "custom_rules": custom_rules  # 保存自定义映射规则
        }
        
        if previous_lora is not None:
            loras_list.extend(previous_lora)
        
        loras_list.append(lora_info)
        return (loras_list,)

class LTXVWan21LoRALoader:
    """适配Wan2.1系列LoRA的加载器，扩展标准LTXV LoRA Loader功能"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "要应用LoRA的模型"},
                ),
                "ltxv_loader_as_input": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "使用官方LTXV Loader作为输入时设为true，否则设为false",
                    },
                ),
                "dump_lora_structure": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "将LoRA的完整结构写入日志，便于调试",
                    },
                ),
                "dump_model_params": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "显示模型的实际参数名，帮助调整映射规则",
                    },
                ),
                "only_keys_with_matches": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "只应用能够匹配到模型参数的LoRA键名，建议关闭此选项使用全部键名",
                    },
                ),
                "partial_match": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "使用部分匹配而非完全匹配，增大匹配率",
                    },
                ),
            },
            "optional": {
                "lora": ("LTXVLORA", {"default": None}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "lora_loader"
    CATEGORY = "LTXVideoLoRA"
    TITLE = "LTXV Wan2.1 LoRA Loader"
    
    def lora_loader(self, model, ltxv_loader_as_input, dump_lora_structure=False, dump_model_params=False, only_keys_with_matches=False, partial_match=True, lora=None):
        if lora is None:
            return (model,)
        
        from comfy.sd import load_lora_for_models
        from comfy.utils import load_torch_file
        import torch
        
        # 收集模型结构信息用于提高适配性
        if hasattr(model, "diffusion_model"):
            try:
                # 帮助用户分析模型结构
                if dump_lora_structure:
                    # 提取和打印模型结构
                    module_structure = self._analyze_model_structure(model)
                    log.info("\n==== LTXV 模型结构分析 ====\n")
                    self._print_model_structure(module_structure)
                    log.info("\n==== 模型结构分析结束 ====\n")
            except Exception as e:
                log.warning(f"分析模型结构时出错: {e}")
        
        # 处理所有LoRA
        for l in lora:
            log.info(f"\n==== 加载LoRA: {l['name']} 强度: {l['strength']} ====\n")
            lora_path = l["path"]
            lora_strength = l["strength"]
            
            # 检查LoRA文件是否存在
            if not os.path.exists(lora_path):
                log.error(f"LoRA文件不存在: {lora_path}")
                continue
                
            try:
                # 加载LoRA权重
                lora_sd = load_torch_file(lora_path, safe_load=True)
                
                # 分析LoRA结构这将帮助用户理解LoRA的组成
                if dump_lora_structure and lora_sd:
                    self._analyze_lora_structure(lora_sd)
                
                # 使用适当的映射函数处理LoRA键名
                if l.get("is_wan21", False):
                    log.info(f"使用Wan2.1专用键名映射处理: {l['name']}")
                    show_details = l.get("show_mapping_details", True)
                    
                    # 处理自定义映射规则
                    custom_rules = l.get("custom_rules", [])
                    if custom_rules:
                        global AUXILIARY_MAPPINGS
                        # 添加用户自定义的映射规则
                        temp_aux_mappings = AUXILIARY_MAPPINGS.copy()
                        AUXILIARY_MAPPINGS.extend(custom_rules)
                        log.info(f"应用 {len(custom_rules)} 条自定义映射规则")
                    
                    # 执行键名映射
                    lora_sd = map_wan21_to_ltxv_keys(lora_sd, ltxv_loader_as_input, show_details)
                    
                    # 恢复原始的辅助映射规则
                    if custom_rules:
                        AUXILIARY_MAPPINGS = temp_aux_mappings
                else:
                    # 使用标准映射函数
                    try:
                        from comfyui_ltxvideolora.nodes.lora_loader_node import standardize_lora_key_format
                        log.info("使用标准LTXV LoRA映射函数")
                        lora_sd = standardize_lora_key_format(lora_sd, ltxv_loader_as_input)
                    except ImportError:
                        # 如果标准函数不可用，使用我们自己的函数
                        log.warning("无法导入标准LTXV LoRA映射函数，使用备用映射函数")
                        lora_sd = map_wan21_to_ltxv_keys(lora_sd, ltxv_loader_as_input, False)
                
                # 如果指定了模型块，只保留那些块的LoRA键名
                blocks = l.get("blocks", None)
                if blocks is not None and len(blocks) > 0:
                    filtered_sd = {}
                    for k, v in lora_sd.items():
                        # 检查这个键是否属于用户选择的块
                        keep_key = False
                        for block_key in blocks.keys():
                            if block_key in k:
                                keep_key = True
                                break
                        if keep_key:
                            filtered_sd[k] = v
                    
                    # 更新LoRA状态字典
                    prev_count = len(lora_sd)
                    lora_sd = filtered_sd
                    log.info(f"模型块过滤: 保留 {len(lora_sd)}/{prev_count} 个键")
                
                # 提取并分析模型参数名
                model_params = []
                model_sd = {}
                if model is not None:
                    try:
                        # 首先检查是否为ComfyUI的ModelPatcher对象
                        if hasattr(model, 'model') and hasattr(model.model, 'named_parameters'):
                            # 这是ComfyUI的ModelPatcher对象
                            for name, _ in model.model.named_parameters():
                                model_sd[name] = True
                                model_params.append(name)
                            log.info(f"从ModelPatcher.model中获取到 {len(model_sd)} 个参数名")
                        elif hasattr(model, 'named_parameters'):
                            # 标准PyTorch模型
                            for name, _ in model.named_parameters():
                                model_sd[name] = True
                                model_params.append(name)
                            log.info(f"从模型中获取到 {len(model_sd)} 个参数名")
                        else:
                            log.warning("无法获取模型参数")
                    except Exception as e:
                        log.warning(f"提取模型参数时出错: {e}")
                
                # 显示模型参数名，帮助调试
                if dump_model_params and model_params:
                    log.info("\n==== 模型参数名分析 ====\n")
                    
                    # 分析模型参数的模式
                    param_patterns = defaultdict(int)
                    for name in model_params:
                        parts = name.split('.')
                        if len(parts) >= 2:
                            pattern = '.'.join(parts[:2])
                            param_patterns[pattern] += 1
                    
                    # 显示常见模式
                    log.info("参数名模式统计:")
                    for pattern, count in sorted(param_patterns.items(), key=lambda x: x[1], reverse=True)[:20]:
                        log.info(f"  {pattern}.*: {count} 个参数")
                    
                    # 显示一些具体参数名样例
                    log.info("\n参数名样例:")
                    for name in sorted(model_params)[:50]:
                        log.info(f"  {name}")
                    
                    if len(model_params) > 50:
                        log.info(f"  ... 以及其他 {len(model_params) - 50} 个参数")
                    
                    log.info("\n==== 模型参数分析结束 ====\n")
                
                # 过滤LoRA键，只保留能匹配到模型参数的键
                if only_keys_with_matches and model is not None and model_sd:
                    filtered_sd = {}
                    matched_count = 0
                    partial_matched = 0
                    
                    for k, v in lora_sd.items():
                        # 去除LoRA特有后缀
                        base_key = re.sub(r'\.(alpha|lora_up\.weight|lora_down\.weight)$', '', k)
                        
                        # 完全匹配
                        if base_key in model_sd:
                            filtered_sd[k] = v
                            matched_count += 1
                            continue
                            
                        # 部分匹配
                        if partial_match:
                            found_match = False
                            # 检查某个模型参数是否包含该键的一部分
                            base_parts = base_key.split('.')
                            if len(base_parts) >= 3:  # 至少要有一定的深度
                                # 提取最后几个组件作为匹配特征
                                significant_part = '.'.join(base_parts[-3:])  # 最后3个组件
                                
                                for param_name in model_sd.keys():
                                    if significant_part in param_name:
                                        filtered_sd[k] = v
                                        partial_matched += 1
                                        found_match = True
                                        break
                            
                            # 如果还是没有匹配到，尝试更宽松的匹配
                            if not found_match and len(base_parts) >= 2:
                                # 提取关键组件
                                for key_part in ['attn1', 'attn2', 'mlp', 'layers', 'blocks']:
                                    if key_part in base_key:
                                        for param_name in model_sd.keys():
                                            if key_part in param_name:
                                                # 确保有更多的匹配元素
                                                match_score = 0
                                                for part in base_parts:
                                                    if part in param_name:
                                                        match_score += 1
                                                if match_score >= 2:  # 至少要有两个匹配组件
                                                    filtered_sd[k] = v
                                                    partial_matched += 1
                                                    found_match = True
                                                    break
                                        if found_match:
                                            break
                    
                    # 更新过滤后的LoRA状态字典
                    prev_count = len(lora_sd)
                    lora_sd = filtered_sd
                    
                    log.info(f"模型参数匹配过滤: 全匹配 {matched_count} 个键, 部分匹配 {partial_matched} 个键, 总计 {len(filtered_sd)}/{prev_count} 个键")
                    
                    # 如果一个都没匹配到，返回原始字典
                    if len(filtered_sd) == 0:
                        log.warning("没有匹配到任何模型参数，将使用原始LoRA键")
                        lora_sd = {}  # 强制让当前的过滤不生效
                
                # 应用LoRA
                if lora_sd and len(lora_sd) > 0:
                    model, _ = load_lora_for_models(model, None, lora_sd, lora_strength, 0)
                    log.info(f"LoRA '{l['name']}' 应用成功，共 {len(lora_sd)} 个键")
                else:
                    log.warning(f"LoRA '{l['name']}' 强度 {lora_strength} 无法被应用: 没有有效的键")
                    
            except Exception as e:
                log.error(f"LoRA '{l['name']}' 应用失败: {e}")
                import traceback
                log.error(traceback.format_exc())
        
        return (model,)
    
    def _analyze_model_structure(self, model):
        """分析模型结构并提取层结构信息"""
        module_structure = {}
        
        for name, _ in model.named_parameters():
            # 分割模块路径
            parts = name.split('.')
            
            # 构建嵌套字典结构
            current = module_structure
            for i, part in enumerate(parts[:-1]):  # 不包括参数名
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        return module_structure
    
    def _print_model_structure(self, structure, depth=0, max_depth=7, prefix=''):
        """递归打印模型结构信息"""
        if depth > max_depth:
            return
            
        indent = '  ' * depth
        for key, value in sorted(structure.items()):
            if isinstance(value, dict):
                if prefix:
                    full_key = f"{prefix}.{key}"
                else:
                    full_key = key
                log.info(f"{indent}{key}/")
                self._print_model_structure(value, depth + 1, max_depth, full_key)
            else:
                log.info(f"{indent}{key}: {value}")
    
    def _analyze_lora_structure(self, lora_sd):
        """分析LoRA的结构并输出统计信息"""
        log.info("\n==== LoRA结构分析 ====\n")
        
        # 分析LoRA键名类型
        key_types = defaultdict(int)
        prefixes = defaultdict(int)
        suffixes = defaultdict(int)
        block_numbers = defaultdict(int)
        
        # 提取常见模式
        for key in lora_sd.keys():
            # 后缀统计
            if key.endswith('.weight'):
                if 'lora_up.weight' in key:
                    suffixes['lora_up.weight'] += 1
                elif 'lora_down.weight' in key:
                    suffixes['lora_down.weight'] += 1
                else:
                    suffixes['other.weight'] += 1
            elif '.alpha' in key:
                suffixes['alpha'] += 1
            
            # 前缀统计
            parts = key.split('.')
            if len(parts) > 0:
                prefixes[parts[0]] += 1
            
            # 检测块号
            block_match = re.search(r'blocks?_?(\d+)', key)
            if block_match:
                block_num = int(block_match.group(1))
                block_numbers[block_num] += 1
            
            # 检测常见层类型
            for layer_type in ['self_attn', 'cross_attn', 'ffn', 'attn1', 'attn2', 'mlp']:
                if layer_type in key:
                    key_types[layer_type] += 1
                    break
        
        # 输出统计结果
        log.info(f"LoRA总键数: {len(lora_sd)}")
        
        log.info("\n层类型统计:")
        for layer_type, count in sorted(key_types.items(), key=lambda x: x[1], reverse=True):
            log.info(f"  {layer_type}: {count}")
        
        log.info("\n前缀统计:")
        for prefix, count in sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:10]:
            log.info(f"  {prefix}: {count}")
        
        log.info("\n后缀统计:")
        for suffix, count in sorted(suffixes.items(), key=lambda x: x[1], reverse=True):
            log.info(f"  {suffix}: {count}")
        
        if block_numbers:
            log.info("\n块统计:")
            block_stats = [(f"Block {num}", count) for num, count in block_numbers.items()]
            for block, count in sorted(block_stats, key=lambda x: int(x[0].split()[1])):
                log.info(f"  {block}: {count}")
        
        # 打印部分键名样例
        log.info("\n键名样例:")
        for i, key in enumerate(list(lora_sd.keys())[:10]):
            tensor = lora_sd[key]
            shape_str = "x".join([str(s) for s in tensor.shape]) if hasattr(tensor, "shape") else "scalar"
            log.info(f"  {key} ({shape_str})")
        
        if len(lora_sd) > 10:
            log.info(f"  ... 以及其他 {len(lora_sd) - 10} 个键")
            
        log.info("\n==== LoRA结构分析结束 ====\n")
