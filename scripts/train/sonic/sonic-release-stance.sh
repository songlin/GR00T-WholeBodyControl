#!/bin/bash
# SONIC release recipe on the stance sample data (sample_data/stance), 5 iterations --
# a quick end-to-end check of the image and the Slurm wrapper.
#
#   bash scripts/train/sonic/sonic-release-stance.sh [hydra overrides]          # inside sonic.sqsh
#   sbatch scripts/train/slurm_job.sh scripts/train/sonic/sonic-release-stance.sh [hydra overrides]
#
# Extra args are appended as Hydra overrides and win over the ones below, e.g.
#   ++algo.config.num_learning_iterations=1000 num_envs=2048

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NUM_PROCESSES=$((NNODES * NPROC_PER_NODE))
ulimit -n 65535

# Run dir: logs_rl/hrsi/<exp_base>_<EXP_VAR>-<timestamp> (sonic_release names runs
# ${exp_base}_${exp_var}); resume matches on that name, so keep it stable across restarts.
EXP_VAR=${EXP_VAR:-stance}
# W&B group for this policy's runs (wandb.wandb_group). A resumed run keeps the group
# it was created with: train_agent_trl.py drops the group when it reattaches by run id.
WANDB_GROUP=${WANDB_GROUP:-stance}

# Requeued by slurm_job.sh after a walltime stop: continue from the newest
# logs_rl/.../<exp_base>_<EXP_VAR>-*/last.pt instead of starting a fresh run.
RESUME=()
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    RESUME=(+resume=True)
    echo "Restart ${SLURM_RESTART_COUNT}: resuming exp_var=$EXP_VAR from its latest last.pt"
fi

LAUNCH=(accelerate launch --num_processes="$NUM_PROCESSES" --num_machines="$NNODES")
if [ "$NUM_PROCESSES" -gt 1 ]; then
    LAUNCH+=(--multi_gpu --machine_rank="$NODE_RANK"
             --main_process_ip="${MASTER_ADDR:-127.0.0.1}" --main_process_port="${MASTER_PORT:-29500}")
fi

echo "Training sonic_release exp_var=$EXP_VAR (wandb group $WANDB_GROUP) on $NNODES node(s) x $NPROC_PER_NODE GPU(s), node rank $NODE_RANK"

"${LAUNCH[@]}" gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    exp_var="$EXP_VAR" \
    project_name=hrsi \
    wandb.wandb_group="$WANDB_GROUP" \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/stance/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/stance/smpl_filtered \
    "${RESUME[@]}" \
    "$@"
