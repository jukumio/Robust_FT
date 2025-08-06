export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

DATA_PATH="./TinyLLaVA_Factory/dataset/openecad/format_train.json"
IMAGE_PATH="./TinyLLaVA_Factory/dataset/openecad/split_dataset/train_images"
MODEL_MAX_LENGTH=2048
OUTPUT_DIR="./stock_test_output/second_full"

deepspeed --include localhost:0,1,2,3 --master_port 29504 ./TinyLLaVA_Factory/tinyllava/train/custom_finetune_seed.py \
    --deepspeed ./TinyLLaVA_Factory/scripts/zero2.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version gemma \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --fp16 True \
    --tune_type_llm full \
    --tune_type_vision_tower full \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --group_by_modality_length False \
    --pretrained_model_path "tinyllava/TinyLLaVA-Gemma-SigLIP-2.4B" \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --seed 77 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name Gemma-2.4B-full-finetune-1epoch