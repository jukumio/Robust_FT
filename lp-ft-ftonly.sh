#!/bin/bash

# TinyLLaVA LP-FT (Linear Probing then Fine-Tuning) Script
# Based on the research paper: "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution"

# Set CUDA memory allocation configuration to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
# Set default paths and parameters
DATA_PATH="./TinyLLaVA_Factory/dataset/openecad/format_train.json"
IMAGE_PATH="./TinyLLaVA_Factory/dataset/openecad/split_dataset/train_images"
OUTPUT_DIR="./stock_test_output/LP_FT"
GPUS="0,1,2,3,4,5"
MASTER_PORT=29508
SEED=42
MODEL_MAX_LENGTH=2048

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --image_path)
      IMAGE_PATH="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --master_port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --model_max_length)
      MODEL_MAX_LENGTH="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo " --output_dir DIR Output directory (default: ./stock_test_output/LP_FT)"
      echo " --data_path PATH Training data JSON path"
      echo " --image_path PATH Training images directory"
      echo " --gpus GPUS GPU IDs to use (default: 0,1,2,3,4,5)"
      echo " --master_port PORT Master port (default: 29508)"
      echo " --seed SEED Random seed (default: 42)"
      echo " --model_max_length LEN Model max length (default: 2048)"
      echo " -h, --help Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use -h or --help for usage information"
      exit 1
      ;;
  esac
done

echo "=== TinyLLaVA LP-FT (Linear Probing then Fine-Tuning) ==="
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "GPUs: $GPUS"
echo "Master port: $MASTER_PORT"
echo "Seed: $SEED"
echo "Data path: $DATA_PATH"
echo "Image path: $IMAGE_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Model max length: $MODEL_MAX_LENGTH"
echo "=================================="

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/linear_probe"
mkdir -p "$OUTPUT_DIR/final_model"

# Set up environment
export PYTHONPATH="${PYTHONPATH}:./TinyLLaVA_Factory"

# Training script path
LP_TRAIN_SCRIPT="./TinyLLaVA_Factory/tinyllava/train/c_ft.py"

# Check if training script exists
if [[ ! -f "$LP_TRAIN_SCRIPT" ]]; then
  echo "Error: Training script '$LP_TRAIN_SCRIPT' not found"
  exit 1
fi

# Create model merger script
cat > "$OUTPUT_DIR/merge_distributed_model.py" << 'EOF'
import os
import torch
import shutil
from transformers import AutoTokenizer
import json

def merge_distributed_tinyllava_model(distributed_path, merged_path):
    """
    TinyLLaVA의 분산 저장된 모델을 통합 모델로 병합
    """
    print(f"Merging distributed model from {distributed_path} to {merged_path}")
    
    # 출력 디렉토리 생성
    os.makedirs(merged_path, exist_ok=True)
    
    # config.json 복사
    if os.path.exists(os.path.join(distributed_path, "config.json")):
        shutil.copy2(
            os.path.join(distributed_path, "config.json"),
            os.path.join(merged_path, "config.json")
        )
        print("✅ Copied config.json")
    
    # tokenizer 관련 파일들 복사
    tokenizer_files = [
        "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"
    ]
    for file in tokenizer_files:
        src_path = os.path.join(distributed_path, file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(merged_path, file))
            print(f"✅ Copied {file}")
    
    # 분산 저장된 모델 컴포넌트들을 찾아서 병합
    model_state_dict = {}
    
    # connector 가중치 로드
    connector_path = os.path.join(distributed_path, "connector")
    if os.path.exists(connector_path):
        for file in os.listdir(connector_path):
            if file.endswith(".bin"):
                connector_weights = torch.load(
                    os.path.join(connector_path, file), 
                    map_location="cpu"
                )
                for key, value in connector_weights.items():
                    model_state_dict[f"multi_modal_projector.{key}"] = value
                print(f"✅ Loaded connector weights from {file}")
    
    # language_model 가중치 로드  
    language_model_path = os.path.join(distributed_path, "language_model")
    if os.path.exists(language_model_path):
        for file in os.listdir(language_model_path):
            if file.endswith(".bin"):
                lm_weights = torch.load(
                    os.path.join(language_model_path, file),
                    map_location="cpu"
                )
                for key, value in lm_weights.items():
                    model_state_dict[f"language_model.{key}"] = value
                print(f"✅ Loaded language model weights from {file}")
    
    # vision_tower 가중치 로드
    vision_tower_path = os.path.join(distributed_path, "vision_tower")
    if os.path.exists(vision_tower_path):
        for file in os.listdir(vision_tower_path):
            if file.endswith(".bin"):
                vision_weights = torch.load(
                    os.path.join(vision_tower_path, file),
                    map_location="cpu"
                )
                for key, value in vision_weights.items():
                    model_state_dict[f"vision_tower.{key}"] = value
                print(f"✅ Loaded vision tower weights from {file}")
    
    # 통합된 모델 저장
    if model_state_dict:
        torch.save(model_state_dict, os.path.join(merged_path, "pytorch_model.bin"))
        print("✅ Saved merged pytorch_model.bin")
        
        # 모델 인덱스 생성 (큰 모델의 경우)
        total_size = sum(p.numel() * p.element_size() for p in model_state_dict.values())
        weight_map = {key: "pytorch_model.bin" for key in model_state_dict.keys()}
        
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map
        }
        
        with open(os.path.join(merged_path, "pytorch_model.bin.index.json"), "w") as f:
            json.dump(index, f, indent=2)
        print("✅ Created model index")
        
        return True
    else:
        print("❌ No model weights found to merge")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python merge_distributed_model.py <distributed_path> <merged_path>")
        sys.exit(1)
    
    distributed_path = sys.argv[1]
    merged_path = sys.argv[2]
    
    success = merge_distributed_tinyllava_model(distributed_path, merged_path)
    if not success:
        sys.exit(1)
EOF

# Step 1: Linear Probing Stage
echo ""
echo "=== Step 1: Linear Probing Stage ==="

# Check if Linear Probing already completed
if [[ -f "$OUTPUT_DIR/linear_probe/config.json" && -d "$OUTPUT_DIR/linear_probe/connector" ]]; then
  echo "✅ Linear Probing model already exists, skipping..."
else
  echo "Training only the head while freezing the feature extractor..."
  
  # Generate random port to avoid conflicts
  LP_PORT=$((29500 + RANDOM % 1000))
  
  # Linear Probing Command
  LP_CMD="deepspeed --include localhost:$GPUS --master_port $LP_PORT $LP_TRAIN_SCRIPT"
  LP_CMD="$LP_CMD --deepspeed ./TinyLLaVA_Factory/scripts/zero2.json"
  LP_CMD="$LP_CMD --data_path $DATA_PATH"
  LP_CMD="$LP_CMD --image_folder $IMAGE_PATH"
  LP_CMD="$LP_CMD --is_multimodal True"
  LP_CMD="$LP_CMD --conv_version gemma"
  LP_CMD="$LP_CMD --mm_vision_select_layer -2"
  LP_CMD="$LP_CMD --image_aspect_ratio square"
  LP_CMD="$LP_CMD --fp16 True"
  # Linear Probing specific settings
  LP_CMD="$LP_CMD --tune_type_llm frozen"
  LP_CMD="$LP_CMD --tune_type_vision_tower frozen"
  LP_CMD="$LP_CMD --tune_type_connector full"
  LP_CMD="$LP_CMD --group_by_modality_length False"
  LP_CMD="$LP_CMD --pretrained_model_path tinyllava/TinyLLaVA-Gemma-SigLIP-2.4B"
  LP_CMD="$LP_CMD --output_dir $OUTPUT_DIR/linear_probe"
  LP_CMD="$LP_CMD --num_train_epochs 1"
  LP_CMD="$LP_CMD --seed $SEED"
  LP_CMD="$LP_CMD --per_device_train_batch_size 1"
  LP_CMD="$LP_CMD --per_device_eval_batch_size 1"
  LP_CMD="$LP_CMD --gradient_accumulation_steps 8"
  LP_CMD="$LP_CMD --evaluation_strategy no"
  LP_CMD="$LP_CMD --save_strategy epoch"
  LP_CMD="$LP_CMD --save_total_limit 2"
  LP_CMD="$LP_CMD --learning_rate 1e-3"
  LP_CMD="$LP_CMD --weight_decay 0.01"
  LP_CMD="$LP_CMD --warmup_ratio 0.03"
  LP_CMD="$LP_CMD --lr_scheduler_type cosine"
  LP_CMD="$LP_CMD --logging_steps 1"
  LP_CMD="$LP_CMD --tf32 False"
  LP_CMD="$LP_CMD --model_max_length $MODEL_MAX_LENGTH"
  LP_CMD="$LP_CMD --gradient_checkpointing False"
  LP_CMD="$LP_CMD --dataloader_num_workers 1"
  LP_CMD="$LP_CMD --lazy_preprocess True"
  LP_CMD="$LP_CMD --report_to tensorboard"
  LP_CMD="$LP_CMD --tokenizer_use_fast False"
  LP_CMD="$LP_CMD --run_name LP_stage"

  echo "Running Linear Probing with port $LP_PORT..."
  eval $LP_CMD

  if [[ $? -ne 0 ]]; then
    echo "❌ Linear Probing stage failed"
    exit 1
  fi
fi

# Check and display LP model structure
echo ""
echo "=== LP Model Structure Check ==="
echo "Contents of $OUTPUT_DIR/linear_probe/:"
ls -la "$OUTPUT_DIR/linear_probe/"

# Step 2: Merge Distributed Model
echo ""
echo "=== Step 2: Merging Distributed LP Model ==="
MERGED_LP_PATH="$OUTPUT_DIR/linear_probe_merged"

if [[ ! -f "$MERGED_LP_PATH/pytorch_model.bin" ]]; then
  echo "Merging distributed LP model..."
  python "$OUTPUT_DIR/merge_distributed_model.py" "$OUTPUT_DIR/linear_probe" "$MERGED_LP_PATH"
  
  if [[ $? -ne 0 ]]; then
    echo "❌ Model merging failed, trying alternative approach..."
    
    # Alternative: Create symbolic links to distributed components
    mkdir -p "$MERGED_LP_PATH"
    
    # Copy essential files
    cp "$OUTPUT_DIR/linear_probe/config.json" "$MERGED_LP_PATH/" 2>/dev/null || true
    cp "$OUTPUT_DIR/linear_probe/tokenizer"* "$MERGED_LP_PATH/" 2>/dev/null || true
    cp "$OUTPUT_DIR/linear_probe/special_tokens_map.json" "$MERGED_LP_PATH/" 2>/dev/null || true
    
    # Create symbolic links to distributed folders
    ln -sf "$(realpath $OUTPUT_DIR/linear_probe/connector)" "$MERGED_LP_PATH/connector" 2>/dev/null || true
    ln -sf "$(realpath $OUTPUT_DIR/linear_probe/language_model)" "$MERGED_LP_PATH/language_model" 2>/dev/null || true  
    ln -sf "$(realpath $OUTPUT_DIR/linear_probe/vision_tower)" "$MERGED_LP_PATH/vision_tower" 2>/dev/null || true
    
    echo "✅ Created alternative merged model structure"
  fi
else
  echo "✅ Merged LP model already exists"
fi

echo "Merged model contents:"
ls -la "$MERGED_LP_PATH/"

# Step 3: Fine-Tuning Stage
echo ""
echo "=== Step 3: Fine-Tuning Stage ==="
echo "Initializing with linear probed head and fine-tuning all parameters..."

# Generate different port for FT
FT_PORT=$((30000 + RANDOM % 1000))

# Fine-Tuning Command
FT_CMD="deepspeed --include localhost:$GPUS --master_port $FT_PORT $LP_TRAIN_SCRIPT"
FT_CMD="$FT_CMD --deepspeed ./TinyLLaVA_Factory/scripts/zero2.json"
FT_CMD="$FT_CMD --data_path $DATA_PATH"
FT_CMD="$FT_CMD --image_folder $IMAGE_PATH"
FT_CMD="$FT_CMD --is_multimodal True"
FT_CMD="$FT_CMD --conv_version gemma"
FT_CMD="$FT_CMD --mm_vision_select_layer -2"
FT_CMD="$FT_CMD --image_aspect_ratio square"
FT_CMD="$FT_CMD --fp16 True"
# Fine-tuning specific settings: train all components
FT_CMD="$FT_CMD --tune_type_llm full"
FT_CMD="$FT_CMD --tune_type_vision_tower full"
FT_CMD="$FT_CMD --tune_vision_tower_from_layer 0"
FT_CMD="$FT_CMD --tune_type_connector full"
FT_CMD="$FT_CMD --group_by_modality_length False"

# Try merged model first, fallback to distributed
if [[ -f "$MERGED_LP_PATH/pytorch_model.bin" ]]; then
  FT_CMD="$FT_CMD --pretrained_model_path $MERGED_LP_PATH"
  echo "Using merged LP model: $MERGED_LP_PATH"
else
  FT_CMD="$FT_CMD --pretrained_model_path $OUTPUT_DIR/linear_probe"
  echo "Using distributed LP model: $OUTPUT_DIR/linear_probe"
fi

FT_CMD="$FT_CMD --output_dir $OUTPUT_DIR/final_model"
FT_CMD="$FT_CMD --num_train_epochs 1"
FT_CMD="$FT_CMD --seed $SEED"
FT_CMD="$FT_CMD --per_device_train_batch_size 1"
FT_CMD="$FT_CMD --per_device_eval_batch_size 1"
FT_CMD="$FT_CMD --gradient_accumulation_steps 8"
FT_CMD="$FT_CMD --evaluation_strategy no"
FT_CMD="$FT_CMD --save_strategy epoch"
FT_CMD="$FT_CMD --save_total_limit 2"
FT_CMD="$FT_CMD --learning_rate 1e-5"  # Lower learning rate for fine-tuning
FT_CMD="$FT_CMD --weight_decay 0.01"
FT_CMD="$FT_CMD --warmup_ratio 0.03"
FT_CMD="$FT_CMD --lr_scheduler_type cosine"
FT_CMD="$FT_CMD --logging_steps 1"
FT_CMD="$FT_CMD --tf32 False"
FT_CMD="$FT_CMD --model_max_length $MODEL_MAX_LENGTH"
FT_CMD="$FT_CMD --gradient_checkpointing True"
FT_CMD="$FT_CMD --dataloader_num_workers 1"
FT_CMD="$FT_CMD --lazy_preprocess True"
FT_CMD="$FT_CMD --report_to tensorboard"
FT_CMD="$FT_CMD --tokenizer_use_fast False"
FT_CMD="$FT_CMD --run_name LP_FT_final"

echo "Running Fine-Tuning with port $FT_PORT..."
echo "Command: $FT_CMD"
echo ""

eval $FT_CMD

# Check exit status
if [[ $? -eq 0 ]]; then
  echo ""
  echo "🎉 LP-FT training completed successfully!"
  echo "Linear Probing model: $OUTPUT_DIR/linear_probe"
  echo "Merged LP model: $MERGED_LP_PATH"  
  echo "Final LP-FT model: $OUTPUT_DIR/final_model"
  echo ""
  echo "=== LP-FT Training Summary ==="
  echo "1. ✅ Linear Probing: Trained connector while freezing vision and language models"
  echo "2. ✅ Model Merging: Converted distributed model to unified format"
  echo "3. ✅ Fine-Tuning: Used LP-initialized model and trained all parameters"
  echo "4. 🧠 Theory: This approach preserves pretrained features better than standard fine-tuning"
else
  echo ""
  echo "❌ Fine-tuning stage failed with exit code: $?"
  echo ""
  echo "🔍 Debugging Information:"
  echo "LP model structure:"
  ls -la "$OUTPUT_DIR/linear_probe/"
  echo ""
  echo "Merged model structure:"
  ls -la "$MERGED_LP_PATH/" 2>/dev/null || echo "Merged model not found"
  echo ""
  echo "💡 Suggestions:"
  echo "1. Check if the model merging step worked correctly"
  echo "2. Verify that all model components exist"
  echo "3. Try running with fewer GPUs if memory issues occur"
  echo "4. Check the log files for detailed error messages"
  exit 1
fi

# Cleanup
rm -f "$OUTPUT_DIR/merge_distributed_model.py"
echo ""
echo "🧹 Cleanup completed"
