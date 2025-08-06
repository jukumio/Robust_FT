import torch
import numpy as np
import json
import os
import shutil
from safetensors.torch import load_file, save_file
from collections import OrderedDict

# 경로 설정
base_model_path = '../stock_test_output/full_finetune/'
finetuned_path_1 = '../stock_test_output/first_model/'
finetuned_path_2 = '../stock_test_output/old_second_model/'

EPS = 1e-8

def load_base_model_weights(base_model_path):
    """분할된 base model weights를 로드합니다."""
    print("Loading base model weights...")
    
    index_path = os.path.join(base_model_path, 'model.safetensors.index.json')
    
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data.get('weight_map', {})
        files = set(weight_map.values())
        
        all_weights = {}
        for file in files:
            file_path = os.path.join(base_model_path, file)
            if os.path.exists(file_path):
                try:
                    weights = load_file(file_path, device="cpu")
                    # float16 → float32 변환
                    weights = {k: v.float() if v.dtype == torch.float16 else v for k, v in weights.items()}
                    all_weights.update(weights)
                    print(f"Loaded {len(weights)} parameters from {file}")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        print(f"Total base model parameters: {len(all_weights)}")
        return all_weights
    else:
        print("model.safetensors.index.json not found!")
        return None

def load_lora_weights(model_path):
    """LoRA adapter 가중치를 로드합니다."""
    adapter_path = os.path.join(model_path, 'adapter_model.safetensors')
    if os.path.exists(adapter_path):
        try:
            weights = load_file(adapter_path, device="cpu")
            # float16 → float32 변환
            weights = {k: v.float() if v.dtype == torch.float16 else v for k, v in weights.items()}
            print(f"Loaded LoRA adapter: {len(weights)} parameters")
            return weights
        except Exception as e:
            print(f"Error loading {adapter_path}: {e}")
            raise
    else:
        raise FileNotFoundError(f"adapter_model.safetensors not found in {model_path}")

def convert_lora_key_to_base_key(lora_key):
    """LoRA 키를 base model 키로 변환합니다."""
    base_key = lora_key.replace('base_model.model.', '')
    base_key = base_key.replace('.lora_A.weight', '.weight')
    base_key = base_key.replace('.lora_B.weight', '.weight')
    base_key = base_key.replace('.lora_A.default.weight', '.weight')
    base_key = base_key.replace('.lora_B.default.weight', '.weight')
    return base_key

def create_lora_mapping(base_weights, lora_weights):
    """LoRA와 base model 간의 매핑을 생성합니다."""
    print("Creating LoRA to base model mapping...")
    
    base_to_lora = {}
    lora_A_keys = [k for k in lora_weights.keys() if '.lora_A.' in k]
    
    successful_mappings = 0
    
    for lora_A_key in lora_A_keys:
        base_key = convert_lora_key_to_base_key(lora_A_key)
        lora_B_key = lora_A_key.replace('.lora_A.', '.lora_B.')
        
        if base_key in base_weights and lora_B_key in lora_weights:
            base_to_lora[base_key] = {
                'lora_A': lora_A_key,
                'lora_B': lora_B_key
            }
            successful_mappings += 1
    
    print(f"Successfully mapped {successful_mappings} LoRA pairs")
    return base_to_lora

def compute_lora_delta(lora_weights, lora_mapping):
    """LoRA weights로부터 delta를 계산합니다."""
    deltas = {}
    
    for base_key, lora_info in lora_mapping.items():
        try:
            lora_A = lora_weights[lora_info['lora_A']].float()
            lora_B = lora_weights[lora_info['lora_B']].float()
            
            # LoRA delta 계산: B @ A
            deltas[base_key] = lora_B @ lora_A
            
        except Exception as e:
            print(f"Error computing delta for {base_key}: {e}")
    
    return deltas

def reconstruct_full_weights(base_weights, lora_deltas):
    """Base weights와 LoRA deltas를 결합하여 full weights를 재구성합니다."""
    print("Reconstructing full weights from LoRA deltas...")
    
    full_weights = {}
    lora_applied_count = 0
    
    for base_key, base_weight in base_weights.items():
        if base_key in lora_deltas:
            try:
                # Full weight = base + delta
                full_weights[base_key] = base_weight.float() + lora_deltas[base_key]
                lora_applied_count += 1
                
            except Exception as e:
                print(f"Error applying LoRA to {base_key}: {e}")
                full_weights[base_key] = base_weight.float()
        else:
            full_weights[base_key] = base_weight.float()
    
    print(f"LoRA applied to {lora_applied_count} layers")
    return full_weights

def compute_angles_model_stock(full_weights_1, full_weights_2, base_weights):
    """Model Stock 공식으로 각도를 계산합니다."""
    print("Computing angles using Model Stock formula...")
    
    angles = {}
    
    # LoRA가 적용된 language model layers만 대상
    target_layers = [k for k in full_weights_1.keys() 
                    if k.startswith('language_model.model.layers.') 
                    and 'embed_tokens' not in k 
                    and 'lm_head' not in k
                    and 'norm' not in k]
    
    print(f"Target layers for angle computation: {len(target_layers)}")
    
    computed_count = 0
    
    for key in target_layers:
        if key in full_weights_1 and key in full_weights_2 and key in base_weights:
            try:
                w0 = base_weights[key].float()
                w1 = full_weights_1[key].float()
                w2 = full_weights_2[key].float()
                
                # Model Stock: (w1 - w0)와 (w2 - w0) 간의 각도
                vec1 = (w1 - w0).flatten()
                vec2 = (w2 - w0).flatten()
                
                # 코사인 유사도
                dot_product = torch.sum(vec1 * vec2)
                norm1 = torch.sqrt(torch.sum(vec1 ** 2))
                norm2 = torch.sqrt(torch.sum(vec2 ** 2))
                
                if norm1 > EPS and norm2 > EPS:
                    cosine_val = dot_product / (norm1 * norm2 + EPS)
                    cosine_val = torch.clamp(cosine_val, min=-1., max=1.)
                    
                    angle_rad = torch.acos(cosine_val)
                    angle_deg = np.rad2deg(angle_rad.item())
                    
                    angles[key] = angle_deg
                    computed_count += 1
                    
                    # 처음 5개만 출력
                    if computed_count <= 5:
                        print(f"   {key}: {angle_deg:.2f}°")
                
            except Exception as e:
                print(f"Error computing angle for {key}: {e}")
    
    print(f"Computed angles for {computed_count} layers")
    return angles

def compute_interpolation_ratios(angles_dict, k=2):
    """각도에서 interpolation ratio를 계산합니다."""
    ratios = {}
    for key, angle_deg in angles_dict.items():
        angle_rad = np.deg2rad(angle_deg)
        cos_theta = np.cos(angle_rad)
        
        # Model Stock 공식: t = k*cos(θ)/((k-1)*cos(θ)+1)
        ratio = k * cos_theta / ((k-1) * cos_theta + 1 + EPS)
        ratios[key] = ratio
    
    return ratios

def merge_weights_model_stock(base_weights, lora_deltas_1, lora_deltas_2, ratios):
    """Model Stock 비율을 사용해 delta들을 병합하고 full weights를 생성합니다."""
    print("Merging weights using Model Stock ratios...")
    
    merged_weights = {}
    model_stock_count = 0
    simple_average_count = 0
    
    for base_key, base_weight in base_weights.items():
        # 둘 다 LoRA delta가 있을 때
        if base_key in lora_deltas_1 and base_key in lora_deltas_2:
            if base_key in ratios:
                # Model Stock ratio 사용
                t = float(np.clip(ratios[base_key], 0.0, 1.0))
                # 올바른 방식: delta_merged = t * (delta1 + delta2) / 2
                delta_avg = (lora_deltas_1[base_key] + lora_deltas_2[base_key]) / 2.0
                merged_delta = t * delta_avg
                merged_weights[base_key] = base_weight.float() + merged_delta
                model_stock_count += 1
            else:
                # 비율 정보가 없으면 단순 평균
                delta_avg = (lora_deltas_1[base_key] + lora_deltas_2[base_key]) / 2.0
                merged_weights[base_key] = base_weight.float() + delta_avg
                simple_average_count += 1
        else:
            # LoRA 델타가 하나만 있거나 없으면 base 그대로
            merged_weights[base_key] = base_weight.float()
    
    print(f"Model Stock applied to {model_stock_count} layers")
    print(f"Simple averaged {simple_average_count} layers (no ratio)")
    print(f"Total merged layers: {len(merged_weights)}")
    return merged_weights

def save_merged_full_model(merged_weights, output_path):
    os.makedirs(output_path, exist_ok=True)

    # 1) Save safetensors for faster load if supported
    safetensors_path = os.path.join(output_path, 'pytorch_model.safetensors')
    cpu_weights = {k: v.cpu() for k, v in merged_weights.items()}
    save_file(cpu_weights, safetensors_path)
    print(f"Saved merged weights to {safetensors_path}")

    # 2) Also save as a real PyTorch checkpoint
    bin_path = os.path.join(output_path, 'pytorch_model.bin')
    torch.save(cpu_weights, bin_path)
    print(f"Saved PyTorch checkpoint to {bin_path}")

    # 3) Copy config & tokenizer files
    for fname in (
        'config.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'tokenizer.model',
    ):
        src = os.path.join(finetuned_path_1, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(output_path, fname))
            print(f"Copied {fname}")
        else:
            print(f"⚠️ {fname} not found, skipping.")



def main():
    # 1) load base + LoRA adapters → 2) compute deltas → 3) reconstruct fulls →
    # 4) compute angles → 5) compute ratios → 6) merge → 7) save
    base = load_base_model_weights(base_model_path)
    lora1 = load_lora_weights(finetuned_path_1)
    lora2 = load_lora_weights(finetuned_path_2)
    map1 = create_lora_mapping(base, lora1)
    map2 = create_lora_mapping(base, lora2)
    d1 = compute_lora_delta(lora1, map1)
    d2 = compute_lora_delta(lora2, map2)
    full1 = reconstruct_full_weights(base, d1)
    full2 = reconstruct_full_weights(base, d2)
    angles = compute_angles_model_stock(full1, full2, base)
    if not angles:
        raise RuntimeError("No angles computed, cannot proceed.")
    ratios = compute_interpolation_ratios(angles, k=2)
    merged = merge_weights_model_stock(base, d1, d2, ratios)
    save_merged_full_model(merged, '../stock_test_output/stock_with_full/')
    # optionally, save ratios/angles json
    info = {
        "angles": {k: float(v) for k, v in angles.items()},
        "ratios": ratios
    }
    with open('../stock_test_output/stock_with_full/info.json', 'w') as f:
        json.dump(info, f, indent=2)
    print("Model Stock merge complete.")

if __name__ == "__main__":
    main()
