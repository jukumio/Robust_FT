import torch
import numpy as np
import json
import os
import shutil
from safetensors.torch import load_file, save_file
from collections import OrderedDict

# 경로 설정
base_model_path = '../stock_test_output/base/'
finetuned_path_1 = '../stock_test_output/full_finetune/'
finetuned_path_2 = '../stock_test_output/second_full_converted/'

EPS = 1e-8

def load_model_weights(model_path):
    """분할된 model weights를 로드합니다."""
    print(f"Loading model weights from {model_path}...")
    
    # safetensors index 파일 확인
    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    
    if os.path.exists(index_path):
        # 분할된 safetensors 파일들 로드
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data.get('weight_map', {})
        files = set(weight_map.values())
        
        all_weights = {}
        for file in files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                try:
                    weights = load_file(file_path, device="cpu")
                    # float16 → float32 변환
                    weights = {k: v.float() if v.dtype == torch.float16 else v for k, v in weights.items()}
                    all_weights.update(weights)
                    print(f"Loaded {len(weights)} parameters from {file}")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        print(f"Total parameters: {len(all_weights)}")
        return all_weights
    
    # 단일 safetensors 파일 확인
    single_safetensors_path = os.path.join(model_path, 'model.safetensors')
    if os.path.exists(single_safetensors_path):
        try:
            weights = load_file(single_safetensors_path, device="cpu")
            weights = {k: v.float() if v.dtype == torch.float16 else v for k, v in weights.items()}
            print(f"Loaded {len(weights)} parameters from single safetensors file")
            return weights
        except Exception as e:
            print(f"Error loading single safetensors: {e}")
    
    # pytorch_model.bin 파일 확인
    bin_path = os.path.join(model_path, 'pytorch_model.bin')
    if os.path.exists(bin_path):
        try:
            weights = torch.load(bin_path, map_location='cpu')
            weights = {k: v.float() if v.dtype == torch.float16 else v for k, v in weights.items()}
            print(f"Loaded {len(weights)} parameters from pytorch_model.bin")
            return weights
        except Exception as e:
            print(f"Error loading pytorch_model.bin: {e}")
    
    raise FileNotFoundError(f"No valid model file found in {model_path}")

def compute_angles_model_stock(w1_weights, w2_weights, w0_weights):
    """Model Stock 공식으로 각도를 계산합니다."""
    print("Computing angles using Model Stock formula...")
    
    angles = {}
    
    # Language model layers만 대상 (임베딩이나 분류기 제외)
    target_layers = []
    for key in w1_weights.keys():
        # 주요 transformer layers만 선택
        if any(pattern in key for pattern in [
            'language_model.model.layers.',
            'model.layers.',
            'transformer.h.',
            'blocks.'
        ]) and not any(skip in key for skip in [
            'embed_tokens', 'lm_head', 'norm', 'embedding'
        ]):
            target_layers.append(key)
    
    print(f"Target layers for angle computation: {len(target_layers)}")
    
    computed_count = 0
    
    for key in target_layers:
        if key in w1_weights and key in w2_weights and key in w0_weights:
            try:
                w0 = w0_weights[key].float()
                w1 = w1_weights[key].float()
                w2 = w2_weights[key].float()
                
                # Model Stock: (w1 - w0)와 (w2 - w0) 간의 각도
                delta1 = (w1 - w0).flatten()
                delta2 = (w2 - w0).flatten()
                
                # 코사인 유사도 계산
                dot_product = torch.sum(delta1 * delta2)
                norm1 = torch.sqrt(torch.sum(delta1 ** 2))
                norm2 = torch.sqrt(torch.sum(delta2 ** 2))
                
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
    """각도에서 interpolation ratio를 계산합니다.
    Model Stock 논문의 공식: t = k*cos(θ)/((k-1)*cos(θ)+1)
    """
    ratios = {}
    for key, angle_deg in angles_dict.items():
        angle_rad = np.deg2rad(angle_deg)
        cos_theta = np.cos(angle_rad)
        
        # Model Stock 공식
        ratio = k * cos_theta / ((k-1) * cos_theta + 1 + EPS)
        # ratio를 0과 1 사이로 클램핑
        ratio = np.clip(ratio, 0.0, 1.0)
        ratios[key] = ratio
    
    return ratios

def merge_weights_model_stock(w0_weights, w1_weights, w2_weights, ratios):
    """Model Stock 방식으로 weights를 병합합니다.
    
    논문의 공식:
    1. w12 = (w1 + w2) / 2  (두 fine-tuned 모델의 평균)
    2. wH = (1-t) * w0 + t * w12  (pre-trained와 평균 모델의 보간)
    """
    print("Merging weights using Model Stock...")
    
    merged_weights = {}
    model_stock_count = 0
    simple_average_count = 0
    base_only_count = 0
    
    for key in w0_weights.keys():
        if key in w1_weights and key in w2_weights:
            # 두 fine-tuned 모델의 평균 계산
            w12 = (w1_weights[key].float() + w2_weights[key].float()) * 0.5
            
            if key in ratios:
                # Model Stock 공식 적용
                t = float(ratios[key])
                merged_weight = (1 - t) * w0_weights[key].float() + t * w12
                merged_weights[key] = merged_weight
                model_stock_count += 1
                
                # 처음 5개의 비율만 출력
                if model_stock_count <= 5:
                    print(f"   {key}: t={t:.4f}")
            else:
                # 비율 정보가 없으면 순수 평균 적용
                merged_weights[key] = w12
                simple_average_count += 1
        else:
            # 하나라도 없으면 base model 사용
            merged_weights[key] = w0_weights[key].float()
            base_only_count += 1
    
    print(f"Model Stock applied to {model_stock_count} layers")
    print(f"Simple averaged {simple_average_count} layers")
    print(f"Base only {base_only_count} layers")
    print(f"Total merged layers: {len(merged_weights)}")
    
    return merged_weights

def save_merged_model(merged_weights, output_path):
    """병합된 모델을 저장합니다."""
    print(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    # CPU로 이동
    cpu_weights = {k: v.cpu() for k, v in merged_weights.items()}
    
    # Safetensors 형식으로 저장
    safetensors_path = os.path.join(output_path, 'model.safetensors')
    save_file(cpu_weights, safetensors_path)
    print(f"Saved merged weights to {safetensors_path}")

    # PyTorch 형식으로도 저장
    bin_path = os.path.join(output_path, 'pytorch_model.bin')
    torch.save(cpu_weights, bin_path)
    print(f"Saved PyTorch checkpoint to {bin_path}")

    # 설정 파일들 복사
    config_files = [
        'config.json',
        'tokenizer_config.json', 
        'special_tokens_map.json',
        'tokenizer.model',
        'tokenizer.json',
        'vocab.txt',
        'merges.txt'
    ]
    
    for fname in config_files:
        # 첫 번째 모델에서 복사 시도
        src = os.path.join(finetuned_path_1, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(output_path, fname))
            print(f"Copied {fname}")
        else:
            # 베이스 모델에서 복사 시도
            src = os.path.join(base_model_path, fname)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(output_path, fname))
                print(f"Copied {fname} from base model")
            else:
                print(f"⚠️ {fname} not found, skipping.")

def analyze_weight_statistics(w0, w1, w2, angles, ratios):
    """가중치 통계 분석."""
    print("\n=== Weight Statistics Analysis ===")
    
    # 각도 통계
    angle_values = list(angles.values())
    print(f"Angles - Mean: {np.mean(angle_values):.2f}°, Std: {np.std(angle_values):.2f}°")
    print(f"Angles - Min: {np.min(angle_values):.2f}°, Max: {np.max(angle_values):.2f}°")
    
    # 비율 통계
    ratio_values = list(ratios.values())
    print(f"Ratios - Mean: {np.mean(ratio_values):.4f}, Std: {np.std(ratio_values):.4f}")
    print(f"Ratios - Min: {np.min(ratio_values):.4f}, Max: {np.max(ratio_values):.4f}")
    
    # 모델 간 유사도 분석
    total_params = 0
    total_similarity_01 = 0
    total_similarity_02 = 0
    total_similarity_12 = 0
    
    for key in w0.keys():
        if key in w1 and key in w2:
            w0_flat = w0[key].flatten()
            w1_flat = w1[key].flatten()
            w2_flat = w2[key].flatten()
            
            # 코사인 유사도
            sim_01 = torch.cosine_similarity(w0_flat.unsqueeze(0), w1_flat.unsqueeze(0)).item()
            sim_02 = torch.cosine_similarity(w0_flat.unsqueeze(0), w2_flat.unsqueeze(0)).item()
            sim_12 = torch.cosine_similarity(w1_flat.unsqueeze(0), w2_flat.unsqueeze(0)).item()
            
            num_params = w0_flat.numel()
            total_params += num_params
            total_similarity_01 += sim_01 * num_params
            total_similarity_02 += sim_02 * num_params
            total_similarity_12 += sim_12 * num_params
    
    print(f"Average cosine similarity (w0 vs w1): {total_similarity_01/total_params:.4f}")
    print(f"Average cosine similarity (w0 vs w2): {total_similarity_02/total_params:.4f}")
    print(f"Average cosine similarity (w1 vs w2): {total_similarity_12/total_params:.4f}")

def main():
    """Model Stock 알고리즘 실행"""
    print("=== Model Stock for Full Fine-tuned Models ===")
    
    # 1. 모델 가중치 로드
    print("\n1. Loading model weights...")
    w0_weights = load_model_weights(base_model_path)
    w1_weights = load_model_weights(finetuned_path_1) 
    w2_weights = load_model_weights(finetuned_path_2)
    
    # 2. 각도 계산
    print("\n2. Computing angles between fine-tuned models...")
    angles = compute_angles_model_stock(w1_weights, w2_weights, w0_weights)
    
    if not angles:
        raise RuntimeError("No angles computed, cannot proceed.")
    
    # 3. 보간 비율 계산
    print("\n3. Computing interpolation ratios...")
    ratios = compute_interpolation_ratios(angles, k=2)
    
    # 4. 가중치 병합
    print("\n4. Merging weights using Model Stock...")
    merged_weights = merge_weights_model_stock(w0_weights, w1_weights, w2_weights, ratios)
    
    # 5. 통계 분석
    analyze_weight_statistics(w0_weights, w1_weights, w2_weights, angles, ratios)
    
    # 6. 병합된 모델 저장
    print("\n5. Saving merged model...")
    output_path = '../stock_test_output/full2_model_stock/'
    save_merged_model(merged_weights, output_path)
    
    # 7. 메타데이터 저장
    print("\n6. Saving metadata...")
    metadata = {
        "method": "Model Stock",
        "base_model": base_model_path,
        "finetuned_models": [finetuned_path_1, finetuned_path_2],
        "angles": {k: float(v) for k, v in angles.items()},
        "ratios": {k: float(v) for k, v in ratios.items()},
        "statistics": {
            "num_layers_merged": len(angles),
            "mean_angle": float(np.mean(list(angles.values()))),
            "mean_ratio": float(np.mean(list(ratios.values())))
        }
    }
    
    metadata_path = os.path.join(output_path, 'model_stock_info.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    print("\n=== Model Stock merge completed successfully! ===")

if __name__ == "__main__":
    main()