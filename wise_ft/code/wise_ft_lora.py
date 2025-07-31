import torch
import numpy as np
import json
import os
from safetensors.torch import load_file, save_file
from collections import OrderedDict
import copy
import argparse
from typing import Dict, Optional, Tuple, List

# 안전한 디바이스 설정
def setup_device():
    """안전한 디바이스 설정"""
    if torch.cuda.is_available():
        try:
            test_tensor = torch.tensor([1.0]).to("cuda")
            print(f"CUDA available and working: {torch.cuda.get_device_name()}")
            return torch.device("cuda")
        except Exception as e:
            print(f"CUDA available but not working: {e}")
            print("Falling back to CPU...")
            return torch.device("cpu")
    else:
        print("ℹ️ CUDA not available, using CPU")
        return torch.device("cpu")

device = setup_device()
EPS = 1e-8

def load_lora_weights(model_path: str) -> Dict[str, torch.Tensor]:
    """LoRA adapter 가중치를 안전하게 로드합니다."""
    adapter_path = os.path.join(model_path, 'adapter_model.safetensors')
    if os.path.exists(adapter_path):
        try:
            print(f"Loading {adapter_path}...")
            weights = load_file(adapter_path, device="cpu")
            print(f"✅ Loaded to CPU: {len(weights)} parameters")
            
            # 안전하게 디바이스로 이동
            global device
            if device.type == "cuda":
                try:
                    weights = {k: v.to(device) for k, v in weights.items()}
                    print(f"✅ Moved to GPU: {device}")
                except Exception as e:
                    print(f"⚠️ Failed to move to GPU: {e}")
                    print("Keeping weights on CPU...")
                    device = torch.device("cpu")
            
            total_params = sum(t.numel() for t in weights.values())
            print(f"📊 Total parameters: {total_params:,}")
            
            return weights
        except Exception as e:
            print(f"❌ Error loading {adapter_path}: {e}")
            raise
    else:
        raise FileNotFoundError(f"adapter_model.safetensors not found in {model_path}")

def load_config(model_path: str) -> Dict:
    """LoRA adapter 설정을 로드합니다."""
    config_path = os.path.join(model_path, 'adapter_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    else:
        print(f"Warning: adapter_config.json not found in {model_path}")
        return {}

def compute_fisher_information(lora_weights: Dict[str, torch.Tensor], 
                             method: str = 'diagonal') -> Dict[str, torch.Tensor]:
    """
    LoRA adapter에 대한 Fisher Information을 계산합니다.
    (실제 데이터 없이 가중치 기반 근사)
    """
    print(f"🧮 Computing Fisher Information using {method} method")
    
    fisher_info = {}
    
    if method == 'diagonal':
        for key, weight in lora_weights.items():
            if 'lora_A' in key or 'lora_B' in key:
                # LoRA 파라미터에 대해 가중치 크기 기반 Fisher
                fisher_info[key] = torch.abs(weight) + EPS
            else:
                fisher_info[key] = torch.ones_like(weight) * 0.1
    
    elif method == 'empirical':
        for key, weight in lora_weights.items():
            fisher_info[key] = torch.abs(weight) ** 0.5 + EPS
    
    elif method == 'uniform':
        for key, weight in lora_weights.items():
            fisher_info[key] = torch.ones_like(weight)
    
    else:
        raise ValueError(f"Unknown Fisher method: {method}")
    
    print(f"✅ Fisher Information computed for {len(fisher_info)} parameters")
    return fisher_info

def _merge(alpha: float, 
          theta_0: Dict[str, torch.Tensor], 
          theta_1: Dict[str, torch.Tensor], 
          fishers: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = None,
          fisher_floor: float = 1e-8) -> Dict[str, torch.Tensor]:
    """
    원본 wise_ft의 _merge 함수를 LoRA용으로 구현
    
    Args:
        alpha: 보간 파라미터 (0: zeroshot only, 1: finetuned only)
        theta_0: Zero-shot LoRA weights
        theta_1: Fine-tuned LoRA weights  
        fishers: Fisher Information (fisher_0, fisher_1) - Optional
        fisher_floor: Fisher Information의 최소값
    
    Returns:
        Dict[str, torch.Tensor]: 병합된 LoRA weights
    """
    
    if fishers is None:
        # 원본 wise_ft의 기본 로직: 단순 선형 보간
        print(f"📐 Simple linear interpolation with alpha={alpha:.3f}")
        return {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

    # Fisher Information을 사용한 가중 병합 (wise_ft 고급 기능)
    print(f"🧮 Fisher-weighted interpolation with alpha={alpha:.3f}")
    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_0.keys():
        # Fisher Information 확인 (원본 코드와 동일한 assert)
        assert (key in fisher_0) == (key in fisher_1), f"Fisher mismatch for {key}"
        
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta

def save_merged_weights(merged_weights: Dict[str, torch.Tensor], 
                       output_path: str, 
                       source_config_path: str):
    """병합된 가중치를 저장합니다."""
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # safetensors 형식으로 저장
        adapter_path = os.path.join(output_path, 'adapter_model.safetensors')
        
        # CPU로 이동 후 저장
        cpu_weights = {k: v.cpu() for k, v in merged_weights.items()}
        save_file(cpu_weights, adapter_path)
        
        print(f"✅ Merged weights saved to {adapter_path}")
        
        # 설정 파일들도 복사
        config_files = ['adapter_config.json', 'config.json']
        for config_file in config_files:
            source_config = os.path.join(source_config_path, config_file)
            target_config = os.path.join(output_path, config_file)
            
            if os.path.exists(source_config):
                import shutil
                shutil.copy2(source_config, target_config)
                print(f"✅ Copied {config_file}")
        
    except Exception as e:
        print(f"❌ Error saving weights: {e}")
        raise

def wise_ft(args):
    """
    원본 wise_ft 함수의 LoRA 버전
    """
    print("🚀 Wise-FT for LoRA Adapters")
    print("=" * 60)
    print(f"Device: {device}")
    if args.zeroshot_path:
        print(f"Zero-shot model: {args.zeroshot_path}")
    else:
        print("Zero-shot model: Empty LoRA (all zeros)")
    print(f"Fine-tuned model: {args.finetuned_path}")
    print(f"Output directory: {args.save}")
    print(f"Alpha values: {args.alpha}")
    print(f"Use Fisher: {args.fisher is not None}")
    print("=" * 60)
    
    # Load models
    print("\n📥 STEP 1: Loading LoRA adapters")
    
    # Fine-tuned LoRA는 반드시 로드
    theta_1 = load_lora_weights(args.finetuned_path)   # finetuned LoRA
    
    # Zero-shot LoRA 처리
    if args.zeroshot_path and os.path.exists(os.path.join(args.zeroshot_path, 'adapter_model.safetensors')):
        # 실제 zero-shot LoRA adapter가 있는 경우
        theta_0 = load_lora_weights(args.zeroshot_path)
        print("✅ Loaded actual zero-shot LoRA adapter")
    else:
        # Zero-shot = 빈 LoRA (모든 가중치가 0)
        print("📝 Creating zero-shot LoRA state (all zeros)")
        theta_0 = {key: torch.zeros_like(weight) for key, weight in theta_1.items()}
        print(f"✅ Created zero-shot state with {len(theta_0)} zero-initialized parameters")
    
    # Make sure checkpoints are compatible
    print("\n🔍 STEP 2: Checking compatibility")
    assert set(theta_0.keys()) == set(theta_1.keys()), \
        "LoRA adapters have incompatible keys!"
    print(f"✅ Both adapters have {len(theta_0)} compatible parameters")
    
    # Fisher Information 처리 (원본 코드 로직)
    fishers = None
    if args.fisher is not None:
        print(f"\n🧮 STEP 3: Computing Fisher Information")
        if args.zeroshot_path and os.path.exists(os.path.join(args.zeroshot_path, 'adapter_model.safetensors')):
            fisher_0 = compute_fisher_information(theta_0, method=args.fisher_method)
        else:
            # Zero-shot은 모든 가중치가 0이므로 균등한 Fisher 사용
            fisher_0 = {key: torch.ones_like(weight) * args.fisher_floor 
                       for key, weight in theta_0.items()}
        
        fisher_1 = compute_fisher_information(theta_1, method=args.fisher_method)
        fishers = fisher_0, fisher_1
    else:
        print("\n📝 STEP 3: Skipping Fisher Information (using simple interpolation)")
    
    # Interpolate between checkpoints for each alpha (원본 코드 로직)
    print(f"\n🔄 STEP 4: Interpolating for {len(args.alpha)} alpha values")
    results = {}
    
    for alpha in args.alpha:
        print(f"\n🔄 Processing alpha = {alpha:.3f}")
        
        # 원본 wise_ft의 핵심: _merge 함수 호출
        theta = _merge(alpha, theta_0, theta_1, fishers, args.fisher_floor)
        
        # Save model (원본에서는 finetuned.save(), 여기서는 LoRA adapter 저장)
        output_path = os.path.join(args.save, f'wise_ft_alpha_{alpha:.3f}')
        save_merged_weights(theta, output_path, args.finetuned_path)
        
        # 결과 기록
        results[alpha] = {
            'param_count': len(theta),
            'output_path': output_path
        }
        
        print(f"✅ Alpha {alpha:.3f} completed: {len(theta)} parameters")
        
        # TODO: 여기에 evaluate() 함수 호출 추가 가능
        # evaluate(merged_adapter, args)
    
    # 분석 보고서 생성
    print(f"\n📋 STEP 5: Creating analysis report")
    create_analysis_report(args, results)
    
    print(f"\n🎉 Wise-FT completed successfully!")
    print(f"📁 Results saved in: {args.save}")
    
    return results

def create_analysis_report(args, results: Dict):
    """분석 보고서를 생성합니다."""
    from datetime import datetime
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'method': 'Wise-FT for LoRA Adapters',
        'zeroshot_path': args.zeroshot_path,
        'finetuned_path': args.finetuned_path,
        'use_fisher': args.fisher is not None,
        'fisher_method': args.fisher_method if args.fisher else 'none',
        'alphas_tested': args.alpha,
        'results': results
    }
    
    # JSON 보고서 저장
    report_path = os.path.join(args.save, 'wise_ft_analysis.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 텍스트 요약 저장
    summary_path = os.path.join(args.save, 'wise_ft_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Wise-FT Analysis Summary (LoRA Version)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Method: Wise-FT for LoRA Adapters\n")
        f.write(f"Zero-shot model: {args.zeroshot_path}\n")
        f.write(f"Fine-tuned model: {args.finetuned_path}\n")
        f.write(f"Use Fisher Information: {args.fisher is not None}\n")
        f.write(f"Alpha values tested: {args.alpha}\n\n")
        
        for alpha in args.alpha:
            if alpha in results:
                f.write(f"Alpha = {alpha:.3f}:\n")
                f.write(f"  Parameters merged: {results[alpha].get('param_count', 'N/A')}\n")
                f.write(f"  Output path: {results[alpha].get('output_path', 'N/A')}\n")
                f.write("\n")
        
        f.write("Notes:\n")
        f.write("- Alpha = 0.0: Pure zero-shot model\n")
        f.write("- Alpha = 1.0: Pure fine-tuned model\n")
        f.write("- Alpha = 0.5: Balanced interpolation\n")
        f.write("- Compatible with original Wise-FT methodology\n")
    
    print(f"✅ Analysis reports saved:")
    print(f"   📄 {report_path}")
    print(f"   📄 {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Wise-FT for LoRA Adapters')
    
    # 원본 wise_ft와 유사한 인자 구조
    parser.add_argument('--zeroshot_path', type=str, default=None,
                        help='Path to zero-shot LoRA adapter (optional, defaults to all zeros)')
    parser.add_argument('--finetuned_path', type=str, required=True,
                        help='Path to fine-tuned LoRA adapter')
    parser.add_argument('--save', type=str, required=True,
                        help='Output directory for merged models')
    parser.add_argument('--alpha', type=float, nargs='+', 
                        default=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
                        help='Alpha values for interpolation')
    
    # Fisher Information 관련 (원본의 --fisher 옵션과 유사)
    parser.add_argument('--fisher', type=str, nargs='*', default=None,
                        help='Enable Fisher Information weighting')
    parser.add_argument('--fisher_method', type=str, default='diagonal',
                        choices=['diagonal', 'empirical', 'uniform'],
                        help='Method for computing Fisher Information')
    parser.add_argument('--fisher_floor', type=float, default=1e-8,
                        help='Minimum Fisher Information value')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.save, exist_ok=True)
    
    try:
        results = wise_ft(args)
        print("✅ Wise-FT completed successfully!")
        return results
        
    except Exception as e:
        print(f"\n❌ Error during Wise-FT: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    if results is not None:
        print(f"📊 Generated {len(results)} interpolated models")
    else:
        print("❌ Wise-FT failed!")