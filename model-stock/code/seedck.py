import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path

def compare_model_parameters(model_path1, model_path2, output_file="model_comparison.json"):
    """
    두 모델의 파라미터를 직접 비교
    """
    print(f"Loading models...")
    model1 = AutoModelForCausalLM.from_pretrained(model_path1, trust_remote_code=True)
    model2 = AutoModelForCausalLM.from_pretrained(model_path2, trust_remote_code=True)
    
    model1.eval()
    model2.eval()
    
    comparison_results = {
        "identical_parameters": 0,
        "different_parameters": 0,
        "total_parameters": 0,
        "parameter_differences": {},
        "max_difference": 0.0,
        "mean_difference": 0.0
    }
    
    differences = []
    
    print("Comparing parameters...")
    for name, param1 in model1.named_parameters():
        if name in dict(model2.named_parameters()):
            param2 = dict(model2.named_parameters())[name]
            
            # 파라미터 차이 계산
            diff = torch.abs(param1 - param2)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            differences.append(mean_diff)
            
            comparison_results["total_parameters"] += param1.numel()
            
            if torch.allclose(param1, param2, atol=1e-8):
                comparison_results["identical_parameters"] += param1.numel()
            else:
                comparison_results["different_parameters"] += param1.numel()
                comparison_results["parameter_differences"][name] = {
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "shape": list(param1.shape)
                }
    
    comparison_results["max_difference"] = max(differences) if differences else 0.0
    comparison_results["mean_difference"] = np.mean(differences) if differences else 0.0
    
    # 결과 저장
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n=== Model Parameter Comparison Results ===")
    print(f"Total parameters: {comparison_results['total_parameters']:,}")
    print(f"Identical parameters: {comparison_results['identical_parameters']:,}")
    print(f"Different parameters: {comparison_results['different_parameters']:,}")
    print(f"Percentage identical: {comparison_results['identical_parameters']/comparison_results['total_parameters']*100:.2f}%")
    print(f"Max difference: {comparison_results['max_difference']:.8f}")
    print(f"Mean difference: {comparison_results['mean_difference']:.8f}")
    
    return comparison_results

def compare_model_outputs(model_path1, model_path2, test_prompts, output_file="output_comparison.json"):
    """
    동일한 입력에 대한 두 모델의 출력 비교
    """
    print(f"Loading models for output comparison...")
    
    try:
        model1 = AutoModelForCausalLM.from_pretrained(model_path1, trust_remote_code=True)
        model2 = AutoModelForCausalLM.from_pretrained(model_path2, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path1, trust_remote_code=True)
        
        model1.eval()
        model2.eval()
        
        # GPU 사용 가능하면 GPU로 이동
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model1.to(device)
        model2.to(device)
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None
    
    comparison_results = {
        "prompts_tested": len(test_prompts),
        "identical_outputs": 0,
        "different_outputs": 0,
        "output_comparisons": []
    }
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            print(f"Testing prompt {i+1}/{len(test_prompts)}: {prompt[:50]}...")
            
            try:
                # 입력 토큰화
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 각 모델에서 예측
                outputs1 = model1(**inputs)
                outputs2 = model2(**inputs)
                
                # 로짓 비교
                logits1 = outputs1.logits
                logits2 = outputs2.logits
                
                # 크기가 다른 경우 처리
                if logits1.shape != logits2.shape:
                    print(f"Warning: Logits shape mismatch - Model1: {logits1.shape}, Model2: {logits2.shape}")
                    comparison_results["different_outputs"] += 1
                    continue
                
                # 동일한지 확인 (더 엄격한 기준 사용)
                is_identical = torch.allclose(logits1, logits2, atol=1e-6, rtol=1e-5)
                
                if is_identical:
                    comparison_results["identical_outputs"] += 1
                else:
                    comparison_results["different_outputs"] += 1
                
                # 차이 계산
                diff = torch.abs(logits1 - logits2)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                
                # 예측된 토큰 비교
                pred_tokens1 = torch.argmax(logits1, dim=-1)
                pred_tokens2 = torch.argmax(logits2, dim=-1)
                token_identical = torch.equal(pred_tokens1, pred_tokens2)
                
                comparison_results["output_comparisons"].append({
                    "prompt": prompt,
                    "identical_logits": is_identical,
                    "identical_predictions": bool(token_identical),  # 명시적으로 bool로 변환
                    "max_logit_diff": max_diff,
                    "mean_logit_diff": mean_diff,
                    "logits_shape": list(logits1.shape)
                })
                
            except Exception as e:
                print(f"Error processing prompt {i+1}: {e}")
                comparison_results["different_outputs"] += 1
                continue
    
    # 결과 저장
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n=== Model Output Comparison Results ===")
    print(f"Prompts tested: {comparison_results['prompts_tested']}")
    print(f"Identical outputs: {comparison_results['identical_outputs']}")
    print(f"Different outputs: {comparison_results['different_outputs']}")
    print(f"Percentage identical: {comparison_results['identical_outputs']/comparison_results['prompts_tested']*100:.2f}%")
    
    return comparison_results

def compare_training_logs(log_path1, log_path2):
    """
    학습 로그 비교 (TensorBoard 로그 또는 JSON 로그)
    """
    print("Comparing training logs...")
    
    # TensorBoard 로그 파일들 비교
    # 또는 training loss, validation loss 등 비교
    
    pass

def full_model_comparison(model_path1, model_path2):
    """
    전체적인 모델 비교 수행
    """
    print("=== Starting Full Model Comparison ===\n")
    
    # 1. 파라미터 비교
    param_results = compare_model_parameters(model_path1, model_path2)
    
    # 2. 출력 비교용 테스트 프롬프트
    test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain machine learning in simple terms.",
        "Write a short story about a robot.",
        "Solve this math problem: 2 + 2 = ?"
    ]
    
    # 2. 출력 비교
    output_results = compare_model_outputs(model_path1, model_path2, test_prompts)
    
    # 3. 종합 결과
    print(f"\n=== Summary ===")
    if param_results["different_parameters"] == 0:
        print("✅ Models are IDENTICAL in parameters")
    else:
        print("❌ Models have DIFFERENT parameters")
    
    if output_results["different_outputs"] == 0:
        print("✅ Models produce IDENTICAL outputs")
    else:
        print("❌ Models produce DIFFERENT outputs")
    
    return param_results, output_results


if __name__ == "__main__":
    # 모델 경로 설정
    model1_path = "../stock_test_output/first_model"  # 첫 번째 모델
    model2_path = "../stock_test_output/old_second_model"  # 두 번째 모델
    
    # 전체 비교 실행
    param_results, output_results = full_model_comparison(model1_path, model2_path)