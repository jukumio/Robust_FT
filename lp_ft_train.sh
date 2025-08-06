#!/bin/bash

# TinyLLaVA LP-FT (Linear Probing then Fine-Tuning) Script
# Based on the research paper: "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution"

# Set CUDA memory allocation configuration to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.6

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
            echo "  --output_dir DIR         Output directory (default: ./stock_test_output/LP_FT)"
            echo "  --data_path PATH         Training data JSON path"
            echo "  --image_path PATH        Training images directory"
            echo "  --gpus GPUS              GPU IDs to use (default: 0,1,2,3)"
            echo "  --master_port PORT       Master port (default: 29508)"
            echo "  --seed SEED              Random seed (default: 42)"
            echo "  --model_max_length LEN   Model max length (default: 2048)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "LP-FT Process:"
            echo "  1. Linear Probing: Train only the head while freezing feature extractor"
            echo "  2. Fine-Tuning: Initialize with LP head and fine-tune all parameters"
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
LP_TRAIN_SCRIPT="./TinyLLaVA_Factory/tinyllava/train/custom_finetune_seed.py"

# Check if training script exists
if [[ ! -f "$LP_TRAIN_SCRIPT" ]]; then
    echo "Error: Training script '$LP_TRAIN_SCRIPT' not found"
    exit 1
fi

# Step 1: Linear Probing Stage
echo ""
echo "=== Step 1: Linear Probing Stage ==="

# Check if Linear Probing already completed
if [[ -f "$OUTPUT_DIR/linear_probe/config.json" && -d "$OUTPUT_DIR/linear_probe/connector" ]]; then
    echo "✅ Linear Probing model already exists, skipping..."
    echo "Found model components:"
    if [[ -d "$OUTPUT_DIR/linear_probe/connector" ]]; then echo "  - Connector: ✅"; fi
    if [[ -d "$OUTPUT_DIR/linear_probe/language_model" ]]; then echo "  - Language Model: ✅"; fi  
    if [[ -d "$OUTPUT_DIR/linear_probe/vision_tower" ]]; then echo "  - Vision Tower: ✅"; fi
    if [[ -f "$OUTPUT_DIR/linear_probe/config.json" ]]; then echo "  - Config: ✅"; fi
    echo "Proceeding to Fine-tuning stage..."
else
    echo "Training only the head while freezing the feature extractor..."
    echo ""

    # Linear Probing Command
    LP_CMD="deepspeed --include localhost:$GPUS --master_port $MASTER_PORT $LP_TRAIN_SCRIPT"
    LP_CMD="$LP_CMD --deepspeed ./TinyLLaVA_Factory/scripts/zero2.json"
    LP_CMD="$LP_CMD --data_path $DATA_PATH"
    LP_CMD="$LP_CMD --image_folder $IMAGE_PATH"
    LP_CMD="$LP_CMD --is_multimodal True"
    LP_CMD="$LP_CMD --conv_version gemma"
    LP_CMD="$LP_CMD --mm_vision_select_layer -2"
    LP_CMD="$LP_CMD --image_aspect_ratio square"
    LP_CMD="$LP_CMD --fp16 True"
    # Linear Probing specific settings: freeze LLM and vision tower, train only connector
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
    LP_CMD="$LP_CMD --learning_rate 1e-3"  # Higher learning rate for linear probing
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

    echo "Running Linear Probing command:"
    echo "$LP_CMD"
    echo ""

    eval $LP_CMD

    # Check if linear probing succeeded
    if [[ $? -ne 0 ]]; then
        echo ""
        echo "Linear Probing stage failed with exit code: $?"
        exit 1
    fi
fi

# Check if the model was saved properly (TinyLLaVA uses distributed saving)
if [[ ! -f "$OUTPUT_DIR/linear_probe/config.json" ]]; then
    echo ""
    echo "Error: Linear Probing model not found in $OUTPUT_DIR/linear_probe/"
    echo "Checking for alternative save formats..."
    ls -la "$OUTPUT_DIR/linear_probe/"
    exit 1
elif [[ -f "$OUTPUT_DIR/linear_probe/config.json" ]]; then
    echo "✅ Linear Probing model found with distributed components"
    echo "Model components detected:"
    if [[ -d "$OUTPUT_DIR/linear_probe/connector" ]]; then echo "  - Connector: ✅"; fi
    if [[ -d "$OUTPUT_DIR/linear_probe/language_model" ]]; then echo "  - Language Model: ✅"; fi  
    if [[ -d "$OUTPUT_DIR/linear_probe/vision_tower" ]]; then echo "  - Vision Tower: ✅"; fi
    if [[ -f "$OUTPUT_DIR/linear_probe/config.json" ]]; then echo "  - Config: ✅"; fi
fi

echo ""
echo "Linear Probing stage completed successfully!"
echo "Checking saved model files..."
echo "Contents of $OUTPUT_DIR/linear_probe/:"
ls -la "$OUTPUT_DIR/linear_probe/" || echo "Directory not found"
echo ""

# Step 2: Fine-Tuning Stage
echo "=== Step 2: Fine-Tuning Stage ==="
echo "Initializing with linear probed head and fine-tuning all parameters..."
echo ""

# Use the linear probed model as initialization for fine-tuning
LP_MODEL_PATH="$OUTPUT_DIR/linear_probe"

# Fine-Tuning Command - Use custom_finetune_seed.py (compatible with command line args)
FT_CMD="deepspeed --include localhost:$GPUS --master_port $((MASTER_PORT + 1)) $LP_TRAIN_SCRIPT"
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
# Use the linear probed model as initialization
FT_CMD="$FT_CMD --pretrained_model_path $LP_MODEL_PATH"
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

echo "Running Fine-Tuning command:"
echo "$FT_CMD"
echo ""

eval $FT_CMD

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "LP-FT training completed successfully!"
    echo "Linear Probing model saved in: $OUTPUT_DIR/linear_probe"
    echo "Final LP-FT model saved in: $OUTPUT_DIR/final_model"
    echo ""
    echo "=== LP-FT Training Summary ==="
    echo "1. Linear Probing: Trained only the connector/head while freezing vision and language models"
    echo "2. Fine-Tuning: Used LP-initialized model and trained all parameters with lower learning rate"
    echo "3. This approach combines benefits of both linear probing (preserving pretrained features) and fine-tuning (adaptation to downstream task)"
else
    echo ""
    echo "Fine-tuning stage failed with exit code: $?"
    exit 1
fi