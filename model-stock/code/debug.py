import torch
import json
import os
from safetensors.torch import load_file

# 경로 설정
base_model_path = '../stock_test_output/base/'
finetuned_path_1 = '../stock_test_output/first_model/'
finetuned_path_2 = '../stock_test_output/second_model/'

def quick_key_analysis():
    """핵심 정보만 빠르게 분석"""
    print("🔍 Quick Key Analysis")
    print("=" * 50)
    
    # 1. Base model 키 샘플
    print("\n1️⃣ Base Model Keys:")
    try:
        index_path = os.path.join(base_model_path, 'model.safetensors.index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            weight_map = index_data.get('weight_map', {})
            
            print(f"   Total base keys: {len(weight_map)}")
            print("   Sample base keys (first 5):")
            for i, key in enumerate(list(weight_map.keys())[:5]):
                print(f"     {key}")
            print("   Sample base keys (last 5):")
            for i, key in enumerate(list(weight_map.keys())[-5:]):
                print(f"     {key}")
                
            # 특정 패턴 확인
            attention_keys = [k for k in weight_map.keys() if 'self_attn' in k]
            mlp_keys = [k for k in weight_map.keys() if 'mlp' in k]
            norm_keys = [k for k in weight_map.keys() if 'norm' in k]
            
            print(f"   - Attention layers: {len(attention_keys)}")
            print(f"   - MLP layers: {len(mlp_keys)}")
            print(f"   - Norm layers: {len(norm_keys)}")
            
        else:
            print("   ❌ No index file found")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. LoRA 키 샘플  
    print("\n2️⃣ LoRA Keys:")
    try:
        lora_path = os.path.join(finetuned_path_1, 'adapter_model.safetensors')
        lora_weights = load_file(lora_path, device="cpu")
        
        print(f"   Total LoRA keys: {len(lora_weights)}")
        print("   Sample LoRA keys (first 5):")
        for i, key in enumerate(list(lora_weights.keys())[:5]):
            print(f"     {key}")
        
        # LoRA A/B 분석
        lora_A_keys = [k for k in lora_weights.keys() if '.lora_A.' in k]
        lora_B_keys = [k for k in lora_weights.keys() if '.lora_B.' in k]
        
        print(f"   - LoRA A keys: {len(lora_A_keys)}")
        print(f"   - LoRA B keys: {len(lora_B_keys)}")
        
        if lora_A_keys:
            print(f"   Sample LoRA A: {lora_A_keys[0]}")
        if lora_B_keys:
            print(f"   Sample LoRA B: {lora_B_keys[0]}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. 키 매칭 테스트
    print("\n3️⃣ Key Matching Test:")
    try:
        # 실제 매칭 테스트
        if 'lora_weights' in locals() and 'weight_map' in locals():
            successful_mappings = 0
            failed_mappings = []
            
            for lora_key in list(lora_weights.keys())[:10]:  # 처음 10개만 테스트
                if '.lora_A.' in lora_key or '.lora_B.' in lora_key:
                    # 변환 시도
                    base_key_candidates = [
                        lora_key.replace('base_model.model.', '').replace('.lora_A.weight', '.weight').replace('.lora_B.weight', '.weight'),
                        lora_key.replace('base_model.model.', '').replace('.lora_A.default.weight', '.weight').replace('.lora_B.default.weight', '.weight'),
                        lora_key.replace('base_model.', '').replace('.lora_A.weight', '.weight').replace('.lora_B.weight', '.weight')
                    ]
                    
                    found = False
                    for candidate in base_key_candidates:
                        if candidate in weight_map:
                            successful_mappings += 1
                            found = True
                            break
                    
                    if not found:
                        failed_mappings.append(lora_key)
            
            print(f"   Successful mappings: {successful_mappings}")
            print(f"   Failed mappings: {len(failed_mappings)}")
            
            if failed_mappings:
                print(f"   Sample failed: {failed_mappings[0]}")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 4. 제외 패턴 테스트
    print("\n4️⃣ Exclude Pattern Test:")
    if 'weight_map' in locals():
        exclude_patterns = [
            'embed_tokens.weight',
            'model.norm.weight', 
            'lm_head.weight',
            'norm.weight'
        ]
        
        total_keys = len(weight_map)
        excluded_count = 0
        included_count = 0
        
        for key in weight_map.keys():
            should_exclude = any(pattern in key or key.endswith(pattern) for pattern in exclude_patterns)
            if should_exclude:
                excluded_count += 1
            else:
                included_count += 1
        
        print(f"   Total keys: {total_keys}")
        print(f"   Would exclude: {excluded_count}")
        print(f"   Would include: {included_count}")
        
        # 포함될 키들 샘플
        included_keys = [k for k in list(weight_map.keys())[:20] 
                        if not any(pattern in k or k.endswith(pattern) for pattern in exclude_patterns)]
        
        print(f"   Sample included keys:")
        for key in included_keys[:5]:
            print(f"     {key}")

if __name__ == "__main__":
    quick_key_analysis()