#!/bin/bash
#SBATCH --job-name=dschat-sft
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1         # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=32          # number of cores per tasks
#SBATCH --gres=gpu:8                # number of gpus
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --error=%x-%j.out           # error file name (same to watch just one file)
#SBATCH --account=ENT210059
#SBATCH --partition=afs_rnd

set -e

module purge
module load singularity
module load cuda/12.2

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# so processes know who to talk to
echo "NODELIST="$SLURM_JOB_NODELIST
export MASTER_ADDR=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1`
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export NCCL_CHECKS_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

export HF_HOME=/work/u4005115/alex/cache
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

export LAUNCHER="torchrun \
    --nproc_per_node 8 \
    --nnodes $SLURM_NNODES \
    --rdzv-backend c10d \
    --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
    "

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/work/u4005115/alex/dschat_test
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

MODEL_NAME_OR_PATH=/work/u4005115/models/llama/Llama-2-70b-hf
MODEL_NAME_OR_PATH=/work/u4005115/alex/llama2-70b-chat-36k-taishin-step2-ft/hf_model/
# --data_path Yukang/LongAlpaca-12k \
# --compute_fp32_loss \
# --lora_module_name layers. \
# --lora_learning_rate 3e-4 \
# --only_optimize_lora \
export CMD="training/step1_supervised_finetuning/long_alpaca.py \
   --data_path Yukang/LongAlpaca-12k \
   --data_split 8,1,1 \
   --model_name_or_path $MODEL_NAME_OR_PATH \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 16384 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 10 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --lora_dim 0 \
   --sp 64 \
   --output_dir $OUTPUT \
   --offload \
   "

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

SINGULARITY_RUN="singularity run \
    --nv \
    --no-mount home \
    --writable-tmpfs \
    --env PYTHONPATH=$(pwd) \
    --bind /work/u4005115/alex \
    --bind $(pwd) \
    --bind $MODEL_NAME_OR_PATH \
    /home/u4005115/alex/LongContext/DeepSpeedExamples/applications/DeepSpeed-Chat/dschat.sif \
    "

srun $SRUN_ARGS --jobid $SLURM_JOBID $SINGULARITY_RUN $LAUNCHER $CMD

echo "END TIME: $(date)"