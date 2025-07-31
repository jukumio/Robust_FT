import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def load_and_evaluate_model(base_model_path, adapter_path, test_data_path=None):
    """
    베이스 모델과 LoRA adapter를 로드하고 평가합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading base model from: {base_model_path}")
    
    # 베이스 모델과 토크나이저 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        # 멀티모달 모델의 경우 device_map 문제 해결
        try:
            # 먼저 device_map='auto' 시도
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("✅ Loaded with device_map='auto'")
        except Exception as e1:
            print(f"⚠️ device_map='auto' failed: {e1}")
            try:
                # device_map 없이 시도
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                # 수동으로 GPU로 이동
                if torch.cuda.is_available():
                    base_model = base_model.to(device)
                print(f"✅ Loaded and moved to {device}")
            except Exception as e2:
                print(f"⚠️ GPU loading failed: {e2}")
                # CPU만 사용
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                device = torch.device("cpu")
                base_model = base_model.to(device)
                print(f"✅ Loaded to CPU")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # LoRA adapter 로드
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from: {adapter_path}")
        try:
            model = PeftModel.from_pretrained(base_model, adapter_path)
            print("✅ LoRA adapter loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading LoRA adapter: {e}")
            print("Using base model only...")
            model = base_model
    else:
        print("No adapter found, using base model only")
        model = base_model
    
    model.eval()
    return model, tokenizer

def simple_evaluation(model, tokenizer, test_prompts=None):
    """
    간단한 텍스트 생성으로 모델을 평가합니다.
    """
    if test_prompts is None:
        # 더 간단한 프롬프트들로 변경 (멀티모달 모델 고려)
        test_prompts = [
            "Hello, how are you?",
            "What is the weather like today?",
            "Tell me a joke.",
            "Explain what AI is in simple terms.",
            "What is the capital of France?"
        ]
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    device = next(model.parameters()).device
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        
        try:
            # 토큰화
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            # 디바이스 확인 및 이동
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 생성 파라미터 조정 (안전한 설정)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,  # 더 짧게
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2
                )
            
            # 디코딩 (입력 제거)
            generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"Response: {generated_text.strip()}")
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            # 더 간단한 생성 시도
            try:
                print("Trying simpler generation...")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False,  # greedy
                        pad_token_id=tokenizer.eos_token_id
                    )
                generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                print(f"Response: {generated_text.strip()}")
            except Exception as e2:
                print(f"❌ Simple generation also failed: {e2}")
    
    print("\n" + "="*50)

def compare_models(base_model_path, model_paths, test_prompts=None):
    """
    여러 모델을 비교 평가합니다.
    """
    if test_prompts is None:
        test_prompts = ["What is artificial intelligence?"]
    
    models_info = []
    
    for model_name, adapter_path in model_paths.items():
        print(f"\n{'='*20} Evaluating {model_name} {'='*20}")
        
        model, tokenizer = load_and_evaluate_model(base_model_path, adapter_path)
        if model is not None:
            models_info.append((model_name, model, tokenizer))
    
    # 각 프롬프트에 대해 모든 모델의 응답 비교
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print('='*60)
        
        for model_name, model, tokenizer in models_info:
            print(f"\n--- {model_name} ---")
            
            try:
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=80,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                print(generated_text.strip())
                
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate merged LoRA model")
    parser.add_argument("--base_model", type=str, default="../stock_test_output/base/", 
                        help="Path to base model")
    parser.add_argument("--merged_adapter", type=str, default="../stock_test_output/merged_model/", 
                       help="Path to merged adapter")
    parser.add_argument("--first_adapter", type=str, default="../stock_test_output/first_model/", 
                       help="Path to first adapter")
    parser.add_argument("--second_adapter", type=str, default="../stock_test_output/second_model/", 
                       help="Path to second adapter")
    parser.add_argument("--compare", action="store_true", help="Compare all models")
    
    args = parser.parse_args()
    
    if args.compare:
        # 모든 모델 비교
        model_paths = {
            "Original (No adapter)": None,
            "First Model": args.first_adapter,
            "Second Model": args.second_adapter,
            "Merged Model (Model Stock)": args.merged_adapter
        }
        
        compare_models(args.base_model, model_paths)
        
    else:
        # 병합된 모델만 평가
        print("Evaluating merged model...")
        result = load_and_evaluate_model(args.base_model, args.merged_adapter)
        
        if result is not None and len(result) == 2:
            model, tokenizer = result
            if model is not None and tokenizer is not None:
                simple_evaluation(model, tokenizer)
            else:
                print("❌ Failed to load model or tokenizer")
        else:
            print("❌ Failed to load model")

if __name__ == "__main__":
    # 스크립트로 실행할 때
    main()
    
    # 또는 직접 함수 호출
    # base_model_path = "your_base_model_path"  # 실제 베이스 모델 경로로 변경
    # merged_adapter_path = "../stock_test_output/merged_model/"
    # 
    # model, tokenizer = load_and_evaluate_model(base_model_path, merged_adapter_path)
    # if model is not None:
    #     simple_evaluation(model, tokenizer)
    
