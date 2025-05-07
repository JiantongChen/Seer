#!/bin/bash
module load anaconda/2024.02
source activate seer

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
export CUDA_DEVICE_MAX_CONNECTIONS=1

### nodes gpus rank master_addr job_id
NODES=$1
NPROC_PER_NODE=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT="29501"
BATCH_JOB_ID=$5

# logs
echo "$NODE_RANK,$NODES,$NPROC_PER_NODE,$MASTER_ADDR,$BATCH_JOB_ID"
OUTPUT_LOG="train_rank${NODE_RANK}_${BATCH_JOB_ID}.log"

### NEED TO CHANGE ###
calvin_dataset_path="/ailab/group/pjlab-smartbot/share/Official_Manipulation_Data/simpler_env_lerobot/fractal20220817_data_lerobot"
save_checkpoint_path="checkpoints/seer_fractal"
finetune_from_pretrained_ckpt="checkpoints/seer_calvin/official/pretrain/5.pth"
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
### NEED TO CHANGE ###

torchrun --nnodes="${NODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${NPROC_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 16 \
    --precision fp32 \
    --learning_rate 1e-3 \
    --save_checkpoint \
    --finetune_type "fractal" \
    --wandb_project seer \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name finetune_fractal_ep5 \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --phase "finetune" \
    --action_pred_steps 3 \
    --sequence_length 10 \
    --future_steps 3 \
    --window_size 13 \
    --obs_pred \
    --loss_image \
    --loss_action \
    --report_to_wandb \
    --reset_action_token \
    --reset_obs_token \
    --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \
    --offline >> "${OUTPUT_LOG}" 2>&1