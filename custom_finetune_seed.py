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
    
    # 방법 1: dataloader_generator 설정하지 않기 (권장)
    # Transformers가 내부적으로 seed를 관리하도록 함
    print(f"Training with seed: {training_arguments.seed} (effective: {effective_seed})")
    
    # 방법 2: 만약 반드시 필요하다면 CustomTrainingArguments 사용
    # training_arguments = CustomTrainingArguments(training_arguments, effective_seed)
    
    # load pretrained checkpoint
    model = AutoModelForCausalLM.from_pretrained(
        training_arguments.pretrained_model_path, 
        trust_remote_code=True
    )
    config = model.config
    
    tokenizer = AutoTokenizer.from_pretrained(
        training_arguments.pretrained_model_path, 
        use_fast=False, 
        model_max_length=config.tokenizer_model_max_length,
        padding_side=config.tokenizer_padding_side
    )
    
    model.tokenizer = tokenizer
    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    
    data_arguments.image_processor = AutoImageProcessor.from_pretrained(
        config.vision_model_name_or_path
    )
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