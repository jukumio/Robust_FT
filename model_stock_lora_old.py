import torch
import numpy as np
import json
import os
from safetensors.torch import load_file, save_file
from collections import OrderedDict

# 경로 설정
base_model_path = '../stock_test_output/base/'
finetuned_path_1 = '../stock_test_output/first_model/'
finetuned_path_2 = '../stock_test_output/old_second_model/'

EPS = 1e-8

def load_base_model_weights(base_model_path):
    """분할된 base model weights를 로드합니다."""
    print("🔍 Loading base model weights...")
    
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
                    print(f"✅ Loaded {len(weights)} parameters from {file}")
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
        
        print(f"🎯 Total base model parameters: {len(all_weights)}")
        return all_weights
    else:
        print("❌ model.safetensors.index.json not found!")
        return None

def load_lora_weights(model_path):
    """LoRA adapter 가중치를 로드합니다."""
    adapter_path = os.path.join(model_path, 'adapter_model.safetensors')
    if os.path.exists(adapter_path):
        try:
            weights = load_file(adapter_path, device="cpu")
            # float16 → float32 변환
            weights = {k: v.float() if v.dtype == torch.float16 else v for k, v in weights.items()}
            print(f"✅ Loaded LoRA adapter: {len(weights)} parameters")
            return weights
        except Exception as e:
            print(f"❌ Error loading {adapter_path}: {e}")
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
    print("🔗 Creating LoRA to base model mapping...")
    
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
    
    print(f"✅ Successfully mapped {successful_mappings} LoRA pairs")
    return base_to_lora

def reconstruct_full_weights(base_weights, lora_weights, lora_mapping):
    """LoRA를 적용하여 full weights를 재구성합니다."""
    print("🔧 Reconstructing full weights from LoRA...")
    
    full_weights = {}
    lora_applied_count = 0
    
    for base_key, base_weight in base_weights.items():
        if base_key in lora_mapping:
            try:
                lora_info = lora_mapping[base_key]
                lora_A = lora_weights[lora_info['lora_A']].float()
                lora_B = lora_weights[lora_info['lora_B']].float()
                
                # LoRA delta 계산: B @ A
                delta = lora_B @ lora_A
                
                # Full weight = base + delta
                full_weights[base_key] = base_weight.float() + delta
                lora_applied_count += 1
                
            except Exception as e:
                print(f"⚠️ Error applying LoRA to {base_key}: {e}")
                full_weights[base_key] = base_weight.float()
        else:
            full_weights[base_key] = base_weight.float()
    
    print(f"✅ LoRA applied to {lora_applied_count} layers")
    return full_weights

def compute_angles_model_stock(full_weights_1, full_weights_2, base_weights):
    """Model Stock 공식으로 각도를 계산합니다."""
    print("📐 Computing angles using Model Stock formula...")
    
    angles = {}
    
    # LoRA가 적용된 language model layers만 대상
    target_layers = [k for k in full_weights_1.keys() 
                    if k.startswith('language_model.model.layers.') 
                    and 'embed_tokens' not in k 
                    and 'lm_head' not in k
                    and 'norm' not in k]
    
    print(f"🎯 Target layers for angle computation: {len(target_layers)}")
    
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
                        print(f"   ✅ {key}: {angle_deg:.2f}°")
                
            except Exception as e:
                print(f"⚠️ Error computing angle for {key}: {e}")
    
    print(f"📊 Computed angles for {computed_count} layers")
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

def merge_lora_weights_model_stock(lora_weights_1, lora_weights_2, ratios):
    """Model Stock 비율을 사용해 LoRA weights를 직접 병합합니다."""
    print("🔄 Merging LoRA weights using Model Stock ratios...")
    
    merged_lora = {}
    model_stock_count = 0
    simple_average_count = 0
    
    # 공통 키들 찾기
    common_keys = set(lora_weights_1.keys()) & set(lora_weights_2.keys())
    
    for lora_key in common_keys:
        # LoRA 키를 base 키로 변환하여 ratio 찾기
        base_key = convert_lora_key_to_base_key(lora_key)
        
        if base_key in ratios:
            # Model Stock ratio 사용
            t = np.clip(ratios[base_key], 0.0, 1.0)
            
            # LoRA weights 병합: w_merged = t * (w1 + w2)/2 + (1-t) * 0
            # LoRA 특성상 초기값이 0이므로 단순화: w_merged = t * (w1 + w2)/2
            w_avg = (lora_weights_1[lora_key] + lora_weights_2[lora_key]) / 2.0
            merged_lora[lora_key] = t * w_avg
            
            model_stock_count += 1
        else:
            # ratio가 없는 경우 단순 평균
            merged_lora[lora_key] = (lora_weights_1[lora_key] + lora_weights_2[lora_key]) / 2.0
            simple_average_count += 1
    
    # 한쪽에만 있는 키들 처리
    for key in lora_weights_1.keys():
        if key not in common_keys:
            merged_lora[key] = lora_weights_1[key]
    
    print(f"✅ LoRA merge completed:")
    print(f"   - Model Stock applied: {model_stock_count} parameters")
    print(f"   - Simple average: {simple_average_count} parameters")
    print(f"   - Total merged: {len(merged_lora)} parameters")
    
    return merged_lora

def save_merged_lora(merged_lora_weights, output_path):
    """병합된 LoRA를 저장합니다."""
    print(f"💾 Saving merged LoRA to {output_path}...")
    
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # CPU로 이동 후 저장
        cpu_weights = {k: v.cpu() for k, v in merged_lora_weights.items()}
        
        # adapter_model.safetensors 저장
        adapter_path = os.path.join(output_path, 'adapter_model.safetensors')
        save_file(cpu_weights, adapter_path)
        print(f"✅ Saved: {adapter_path}")
        
        # 설정 파일들 복사
        import shutil
        config_files = ['adapter_config.json', 'config.json']
        
        for config_file in config_files:
            source_file = os.path.join(finetuned_path_1, config_file)
            target_file = os.path.join(output_path, config_file)
            
            if os.path.exists(source_file):
                shutil.copy2(source_file, target_file)
                print(f"✅ Copied: {config_file}")
        
        print(f"🎉 Model saved successfully!")
        
    except Exception as e:
        print(f"❌ Error saving: {e}")
        raise

def main():
    print("🚀 Model Stock for LoRA - Simple Save Version")
    print("=" * 50)
    
    try:
        # 1. 모든 weights 로드
        print("\n📥 STEP 1: Loading all weights")
        base_weights = load_base_model_weights(base_model_path)
        if base_weights is None:
            return None
        
        lora_weights_1 = load_lora_weights(finetuned_path_1)
        lora_weights_2 = load_lora_weights(finetuned_path_2)
        
        # 2. LoRA 매핑 생성
        print("\n🔗 STEP 2: Creating LoRA mappings")
        lora_mapping_1 = create_lora_mapping(base_weights, lora_weights_1)
        lora_mapping_2 = create_lora_mapping(base_weights, lora_weights_2)
        
        # 3. Full weights 재구성
        print("\n🔧 STEP 3: Reconstructing full weights")
        full_weights_1 = reconstruct_full_weights(base_weights, lora_weights_1, lora_mapping_1)
        full_weights_2 = reconstruct_full_weights(base_weights, lora_weights_2, lora_mapping_2)
        
        # 4. 각도 계산
        print("\n📐 STEP 4: Computing angles")
        angles = compute_angles_model_stock(full_weights_1, full_weights_2, base_weights)
        
        if not angles:
            print("❌ No angles computed!")
            return None
        
        # 5. 비율 계산
        print("\n📊 STEP 5: Computing ratios")
        ratios = compute_interpolation_ratios(angles, k=2)
        
        # 통계 출력
        angle_values = list(angles.values())
        ratio_values = list(ratios.values())
        
        print(f"📊 Statistics:")
        print(f"   Angles - Mean: {np.mean(angle_values):.2f}°, Std: {np.std(angle_values):.2f}°")
        print(f"   Ratios - Mean: {np.mean(ratio_values):.4f}, Std: {np.std(ratio_values):.4f}")
        
        # 6. LoRA weights 직접 병합
        print("\n🔄 STEP 6: Merging LoRA weights")
        merged_lora = merge_lora_weights_model_stock(lora_weights_1, lora_weights_2, ratios)
        
        # 7. 저장
        output_path = '../stock_test_output/model_stock_merged_simple/'
        print(f"\n💾 STEP 7: Saving merged model")
        save_merged_lora(merged_lora, output_path)
        
        # 8. 결과 요약 저장
        results = {
            'method': 'Model Stock for LoRA - Direct Merging',
            'angles': {k: float(v) for k, v in angles.items()},
            'ratios': {k: float(v) for k, v in ratios.items()},
            'statistics': {
                'total_layers': len(angles),
                'angle_mean': float(np.mean(angle_values)),
                'angle_std': float(np.std(angle_values)),
                'ratio_mean': float(np.mean(ratio_values)),
                'ratio_std': float(np.std(ratio_values))
            },
            'output_path': output_path
        }
        
        results_path = os.path.join(output_path, 'model_stock_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Model Stock completed successfully!")
        print(f"📁 Model saved to: {output_path}")
        print(f"📄 Results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print("✅ Success!")
    else:
        print("❌ Failed!")