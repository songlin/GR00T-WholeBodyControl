#!/usr/bin/env bash
#SBATCH --job-name=sonic
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=120
#SBATCH --time=23:30:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=.logs/slurm-train-%j.out
#SBATCH --error=.logs/slurm-train-%j.err
#
# Slurm wrapper for SONIC training in sonic.sqsh. Modeled on psi0's
# scripts/train/slurm_job.sh, minus its dataset/venv staging (nothing here needs it).
# Submit from the repo root; everything after the script is the recipe and its args:
#
#   sbatch scripts/train/slurm_job.sh scripts/train/sonic/sonic-release-stance.sh
#   sbatch --nodes=4 scripts/train/slurm_job.sh scripts/train/sonic/<recipe>.sh [hydra overrides]
#
# .logs/ must exist BEFORE sbatch: Slurm opens --output/--error before this script runs,
# so a missing dir fails the job in 0s with no log at all (job 5544). `mkdir -p .logs` once.
#
# Overridable: SONIC_SQSH (default /mnt/beegfs/containers/sonic.sqsh)

PROJECT_ROOT=$(pwd)
TRAIN_CMD="$*"
SQSH=${SONIC_SQSH:-/mnt/beegfs/containers/sonic.sqsh}

mkdir -p "$PROJECT_ROOT/.logs"
[ -f "$PROJECT_ROOT/gear_sonic/train_agent_trl.py" ] || { echo "FATAL: submit from the GR00T-WholeBodyControl root (cwd=$PROJECT_ROOT)"; exit 1; }
[ -n "$TRAIN_CMD" ] || { echo "FATAL: usage: sbatch scripts/train/slurm_job.sh <recipe.sh> [args]"; exit 1; }

# Secrets and cache paths from the repo's .env (see .env.sample), as in psi0. Loaded
# before everything below so the job's own settings win; srun passes the result into
# the container. Refused unless gitignored, so a key can't be committed by accident.
if [ -f "$PROJECT_ROOT/.env" ]; then
    if git -C "$PROJECT_ROOT" check-ignore -q .env; then
        set -a; . "$PROJECT_ROOT/.env"; set +a
        echo "Env:         loaded .env"
    else
        echo "Env:         NOT loading .env -- it isn't gitignored"
    fi
fi

# Keep JIT/compiler scratch and cache off BeeGFS. Each node sees its own /tmp.
export H100_JOB_CACHE_ROOT="/tmp/h100-job-cache/${USER:-songlinwei}/${SLURM_JOB_ID}"
export TRITON_HOME="$H100_JOB_CACHE_ROOT/triton-home"
export TRITON_CACHE_DIR="$H100_JOB_CACHE_ROOT/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$H100_JOB_CACHE_ROOT/torchinductor"
export CUDA_CACHE_PATH="$H100_JOB_CACHE_ROOT/nv"
export TORCH_EXTENSIONS_DIR="$H100_JOB_CACHE_ROOT/torch_extensions"
export XDG_CACHE_HOME="$H100_JOB_CACHE_ROOT/xdg-cache"

# Enroot runtime/data/cache paths are managed by /etc/enroot/enroot.conf.d/10-h100-defaults.conf.
# Keep them node-local to avoid BeeGFS metadata pressure during Pyxis startup.
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_NODE" | awk '{print $1; exit}')
MASTER_ADDR=${MASTER_ADDR:-$MASTER_NODE}
MASTER_PORT=$((29500 + SLURM_JOB_ID % 10000))
NNODES=$SLURM_NNODES

echo "Job ID:      $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0})"
echo "Nodes:       $SLURM_JOB_NODELIST"
echo "Master:      $MASTER_ADDR:$MASTER_PORT"
echo "Num nodes:   $NNODES"
echo "Image:       $SQSH"
echo "Train cmd:   $TRAIN_CMD"

# NCCL config. NCCL_IB_HCA is set per node by scripts/train/nccl_rails.sh in the
# container step below -- mlx5_* numbering differs across nodes, and an export
# here would apply the first node's list to all of them.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_RAS_ENABLE=0
# Not on by default; without it a NCCL hang runs out the walltime instead of
# failing into the --requeue path.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# W&B: the sonic_release exp logs to wandb, and the container has no `wandb login`, so
# hand the key over as WANDB_API_KEY: from .env (loaded above, so it beats the submit
# environment), else the submit environment, else the api.wandb.ai entry in ~/.netrc.
# No key anywhere -> log offline into the experiment dir (`wandb sync` it later).
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.netrc" ]; then
    WANDB_API_KEY=$(awk '{for (i = 1; i < NF; i++) { if ($i == "machine") m = $(i+1);
        if (m == "api.wandb.ai" && $i == "password") { print $(i+1); exit } }}' "$HOME/.netrc")
    [ -n "$WANDB_API_KEY" ] && export WANDB_API_KEY || unset WANDB_API_KEY
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "W&B:         online (key ...${WANDB_API_KEY: -4}${WANDB_ENTITY:+, entity $WANDB_ENTITY})"
elif [ -z "${WANDB_MODE:-}" ]; then
    export WANDB_MODE=offline
    echo "W&B:         offline (no WANDB_API_KEY in the environment, .env or ~/.netrc)"
fi

# --- Auto-requeue on walltime -----------------------------------------------
# `#SBATCH --signal=B:USR1@120` makes Slurm send SIGUSR1 to THIS batch shell 120s
# before the walltime kill. Trap it and requeue the SAME job so it restarts with the
# identical submission. On restart SLURM_RESTART_COUNT>0, and the recipe adds
# +resume=True so train_agent_trl.py picks up the latest last.pt for its experiment.
_REQUEUED=0
_requeue_on_walltime() {
    echo ">>> $(date '+%H:%M:%S') SIGUSR1: nearing walltime -> scontrol requeue ${SLURM_JOB_ID}"
    if scontrol requeue "${SLURM_JOB_ID}"; then
        _REQUEUED=1
    else
        echo ">>> $(date '+%H:%M:%S') WARNING: scontrol requeue failed; job will end at walltime without restart." >&2
    fi
}
trap _requeue_on_walltime USR1

srun_exit=0
# --container-writable: Isaac Sim writes its kit/shader caches inside /isaac-sim.
srun \
  --ntasks=$SLURM_NNODES \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  --container-image="$SQSH" \
  --container-writable \
  --container-mounts=/mnt/beegfs:/mnt/beegfs,"$PROJECT_ROOT":/workspace/GR00T-WholeBodyControl \
  --container-workdir=/workspace/GR00T-WholeBodyControl \
  bash -lc "
    export TMPDIR=\"$H100_JOB_CACHE_ROOT/tmp\"
    export TMP=\"\$TMPDIR\"
    export TEMP=\"\$TMPDIR\"
    mkdir -p \"\$TMPDIR\" \"$TRITON_HOME\" \"$TRITON_CACHE_DIR\" \"$TORCHINDUCTOR_CACHE_DIR\" \"$CUDA_CACHE_PATH\" \"$TORCH_EXTENSIONS_DIR\" \"$XDG_CACHE_HOME\"
    export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    # per-node rail selection (see the NCCL note above)
    . /workspace/GR00T-WholeBodyControl/scripts/train/nccl_rails.sh
    export NNODES=$NNODES
    export NODE_RANK=\$SLURM_PROCID
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    bash $TRAIN_CMD
" &
srun_pid=$!

# Run srun in the BACKGROUND and `wait` on it: bash defers a trap until the current
# *foreground* command returns, so a foreground srun would only fire the requeue trap
# at the hard walltime kill -- too late. `wait` is signal-interruptible.
wait "$srun_pid"
srun_exit=$?

if [[ "$_REQUEUED" == "1" ]]; then
    wait "$srun_pid" 2>/dev/null || true
    echo ">>> $(date '+%H:%M:%S') walltime reached: job requeued (JobID $SLURM_JOB_ID); will resume from last.pt on restart."
    exit 0
fi

if [[ $srun_exit -ne 0 ]]; then
    echo "srun exited with code $srun_exit"
    exit $srun_exit
fi
