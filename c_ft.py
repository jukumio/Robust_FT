import random
import numpy as np
import torch
import os

import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
from transformers import set_seed as transformers_set_seed

from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module

def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

def set_seed(seed):
    """
    완전한 재현성을 위한 개선된 seed 설정 함수
    """
    # DeepSpeed 멀티프로세스 환경 고려
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # 각 프로세스마다 약간 다른 seed 사용 (데이터 셔플링을 위해)
    effective_seed = seed + local_rank
    
    print(f"Rank {local_rank}/{world_size}: Setting seed to {effective_seed}")
    
    # HuggingFace transformers 내장 seed 함수 사용
    transformers_set_seed(effective_seed)
    
    # 추가적인 seed 설정
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
    
    # 재현성을 위한 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 환경변수로도 설정 (일부 라이브러리들이 참조)
    os.environ['PYTHONHASHSEED'] = str(effective_seed)
    
    return effective_seed

def load_tinyllava_model_from_distributed_checkpoint(checkpoint_path):
    """
    TinyLLaVA의 분산 저장된 모델을 로드하는 함수
    """
    print(f"Loading TinyLLaVA model from distributed checkpoint: {checkpoint_path}")
    
    # config.json이 있는지 확인
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {checkpoint_path}")
    
    # 각 컴포넌트 경로 확인
    connector_path = os.path.join(checkpoint_path, "connector")
    language_model_path = os.path.join(checkpoint_path, "language_model") 
    vision_tower_path = os.path.join(checkpoint_path, "vision_tower")
    
    required_paths = [connector_path, language_model_path, vision_tower_path]
    missing_paths = [p for p in required_paths if not os.path.exists(p)]
    
    if missing_paths:
        print(f"Warning: Missing components: {missing_paths}")
        # 일부 컴포넌트가 없어도 계속 진행 (freeze된 컴포넌트일 수 있음)
    
    # TinyLLaVA 모델 로더 사용
    try:
        # TinyLLaVA의 커스텀 로딩 방식 사용
        from tinyllava.model.builder import load_pretrained_model
        
        # tokenizer 먼저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            use_fast=False,
            trust_remote_code=True
        )
        
        # 모델 로드 - TinyLLaVA의 분산 구조 지원
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Failed to load with standard method: {e}")
        
        # 대안: 원본 모델에서 시작해서 Linear Probed 헤드만 로드
        print("Falling back to loading original model and transferring LP head...")
        return load_with_head_transfer(checkpoint_path)

def load_with_head_transfer(lp_checkpoint_path):
    """
    원본 모델을 로드하고 Linear Probed 헤드만 전송하는 방법
    """
    # 원본 pretrained 모델 경로 (LP 학습 전 모델)
    original_model_path = "tinyllava/TinyLLaVA-Gemma-SigLIP-2.4B"
    
    print(f"Loading original model from: {original_model_path}")
    
    # 원본 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        original_model_path,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        original_model_path,
        use_fast=False,
        trust_remote_code=True
    )
    
    # Linear Probed connector 가중치 로드 및 전송
    connector_path = os.path.join(lp_checkpoint_path, "connector")
    if os.path.exists(connector_path):
        print("Loading Linear Probed connector weights...")
        
        # connector 가중치 파일 찾기
        connector_files = []
        for file in os.listdir(connector_path):
            if file.endswith('.bin') or file.endswith('.safetensors'):
                connector_files.append(os.path.join(connector_path, file))
        
        if connector_files:
            # 첫 번째 가중치 파일 로드
            checkpoint = torch.load(connector_files[0], map_location='cpu')
            
            # 모델의 connector 부분에 가중치 적용
            if hasattr(model, 'connector'):
                model.connector.load_state_dict(checkpoint, strict=False)
                print("Successfully loaded Linear Probed connector weights")
            else:
                print("Warning: Model doesn't have connector attribute")
    else:
        print("Warning: No connector weights found in LP checkpoint")
    
    return model, tokenizer

class CustomTrainingArguments:
    """
    pickle 불가능한 객체들을 처리하기 위한 wrapper 클래스
    """
    def __init__(self, training_arguments, dataloader_seed):
        self._training_arguments = training_arguments
        self._dataloader_seed = dataloader_seed
        
    def __getattr__(self, name):
        if name == 'dataloader_generator':
            # 필요할 때마다 새로운 generator 생성
            return torch.Generator().manual_seed(self._dataloader_seed)
        return getattr(self._training_arguments, name)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            setattr(self._training_arguments, name, value)

def train():
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    
    logger_setting(getattr(training_arguments, 'output_dir', None))
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    load_settings(model_arguments, data_arguments, training_arguments)
    
    # 개선된 seed 설정
    effective_seed = set_seed(training_arguments.seed)
    
    print(f"Training with seed: {training_arguments.seed} (effective: {effective_seed})")
    
    # LP-FT를 위한 모델 로딩 로직 개선
    pretrained_path = training_arguments.pretrained_model_path
    
    # Linear Probe 결과인지 확인 (distributed 구조)
    is_lp_checkpoint = (
        os.path.exists(os.path.join(pretrained_path, "connector")) or
        os.path.exists(os.path.join(pretrained_path, "language_model")) or
        os.path.exists(os.path.join(pretrained_path, "vision_tower"))
    )
    
    if is_lp_checkpoint:
        print("Detected Linear Probe checkpoint - using custom loader")
        model, tokenizer = load_tinyllava_model_from_distributed_checkpoint(pretrained_path)
    else:
        print("Loading standard pretrained model")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_path, 
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_path, 
            use_fast=False,
            trust_remote_code=True
        )
    
    # 모델 설정
    config = model.config
    
    # tokenizer 설정 조정
    if hasattr(config, 'tokenizer_model_max_length'):
        tokenizer.model_max_length = config.tokenizer_model_max_length
    if hasattr(config, 'tokenizer_padding_side'):
        tokenizer.padding_side = config.tokenizer_padding_side
    
    model.tokenizer = tokenizer
    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    
    # vision model name 설정
    if hasattr(config, 'vision_model_name_or_path'):
        vision_model_path = config.vision_model_name_or_path
    else:
        # 기본값 설정
        vision_model_path = "google/siglip-so400m-patch14-384"
    
    data_arguments.image_processor = AutoImageProcessor.from_pretrained(vision_model_path)
    data_arguments.is_multimodal = True
    
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_arguments
    )
    
    log_trainable_params(model)  # not work well with zero3
    
    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module
    )
    
    trainer.train()
    
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()