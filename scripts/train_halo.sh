#!/bin/bash

export WANDB_DISABLED=True
export WANDB_ENTITY=""
export WANDB_PROJECT=""
export WANDB_RUN_GROUP=""

export MODEL=meta-llama/Meta-Llama-3-8B-Instruct
export MODEL_PRECISION=bfloat16
export COMPUTE_PRECISION=amp_bf16
export NUM_EPOCHS=1
export LR=5e-5
export WARMUP=20
export BS=32
export PER_DEVICE_BS=4
export SEED=42
export DATASET=sql

export KERNEL_TYPE=base

export BASE_SAVE_PATH="../checkpoints"

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)
   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"
   export "$KEY"="$VALUE"
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" # if not set, default to 0
export MAX_DURATION=${NUM_EPOCHS}ep
export RUN_NAME="halo-${DATASET}-krn_${KERNEL_TYPE}-seed${SEED}-lr${LR}_$RANDOM"
export CONFIG="../configs/fft_${DATASET}.yaml"

export RUN_SAVE_PATH=${BASE_SAVE_PATH}/${RUN_NAME}

export NCCL_NTHREADS=64 # for faster fsdp communication on RTX

composer ../train.py \
    ${CONFIG} \
    model_name_or_path=${MODEL} \
    model.dtype=${MODEL_PRECISION} \
    precision=${COMPUTE_PRECISION} \
    max_duration=${MAX_DURATION} \
    run_name=${RUN_NAME} \
    optimizer.lr=${LR} \
    global_train_batch_size=${BS} \
    device_train_microbatch_size=${PER_DEVICE_BS} \
    device_eval_batch_size=${PER_DEVICE_BS} \
    scheduler.t_warmup=${WARMUP}ba \
    global_seed=${SEED} \
    seed=${SEED} \
    callbacks.had_hf_checkpointer.precision=${MODEL_PRECISION} \
    callbacks.had_hf_checkpointer.save_folder=${RUN_SAVE_PATH} \
    quant_config.kernel=${KERNEL_TYPE}

# move the checkpoint (saved by llm-foundry) to the correct directory
export LAST_SAVE_DIR_NAME=$(ls -t ${RUN_SAVE_PATH}/huggingface | head -n 1)
mv ${RUN_SAVE_PATH}/huggingface/${LAST_SAVE_DIR_NAME}/* ${RUN_SAVE_PATH}
rm -rf ${RUN_SAVE_PATH}/huggingface

echo "find the model at ${RUN_SAVE_PATH}"

if [ "$DATASET" != "code" ]; then
  python eval.py --dataset=${DATASET} --model_path=${RUN_SAVE_PATH} --precision=${MODEL_PRECISION}
else
  accelerate launch bigcode_main.py --model ${RUN_SAVE_PATH} --max_length_generation 512 --tasks humaneval --temperature 1.0 --do_sample False --batch_size 1 --allow_code_execution --metric_output_path ${RUN_SAVE_PATH}/evaluation_results.json --save_generations --save_generations_path ${RUN_SAVE_PATH}/generations.json --precision bf16
fi
