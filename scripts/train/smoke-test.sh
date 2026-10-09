#!/usr/bin/env bash
# GPU smoke test for the sonic image: the driver inside the container, the training
# deps, a headless Isaac Sim boot, and -- if data/ has the Bones-SEED + SMPL motions --
# the docs' 16-env / 5-iteration training run.
#
#   bash scripts/train/smoke-test.sh            # submits itself via srun (1 GPU)
#
# Overridable:
#   SONIC_SQSH   image (default: /mnt/beegfs/scratch/$USER/containers/sonic.sqsh)
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
SQSH=${SONIC_SQSH:-/mnt/beegfs/scratch/$USER/containers/sonic.sqsh}

if [ -z "${SONIC_IN_CONTAINER:-}" ]; then
  exec srun -p gpu --gres=gpu:1 --cpus-per-task=16 --time=01:00:00 -J sonic-smoke \
    --container-image="$SQSH" --container-writable \
    --container-mounts="/mnt/beegfs:/mnt/beegfs,$REPO:/workspace/GR00T-WholeBodyControl" \
    --container-workdir=/workspace/GR00T-WholeBodyControl \
    --export=ALL,SONIC_IN_CONTAINER=1 \
    bash /workspace/GR00T-WholeBodyControl/scripts/train/smoke-test.sh
fi

echo "[smoke] $(hostname) $(date +%T)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "[smoke] check_environment --training"
python check_environment.py --training || echo "[smoke] check_environment reported failures (see above)"

python -c "import torch; print('[smoke] torch', torch.__version__, 'cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "[smoke] booting Isaac Sim headless (first boot compiles shaders, a few minutes)"
# Isaac Lab's create_empty.py tutorial, but bounded: 100 physics steps, then exit.
cat > /tmp/sonic_boot.py <<'PY'
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
from isaaclab.sim import SimulationCfg, SimulationContext
sim = SimulationContext(SimulationCfg(dt=0.01, device="cuda:0"))
sim.reset()
for _ in range(100):
    sim.step()
print("[smoke] ISAAC SIM BOOT OK: 100 physics steps on", sim.device)
app.close()
PY
timeout 900 python /tmp/sonic_boot.py --headless 2>&1 | tail -40
echo "[smoke] boot rc=${PIPESTATUS[0]}"

# The docs' verification run (installation_training.md), once the training data is in place.
if [ -d data/motion_lib_bones_seed/robot_filtered ] && [ -d data/bones_seed_smpl ]; then
  echo "[smoke] $(date +%T) training: 16 envs, 5 iterations, headless"
  timeout 1200 python gear_sonic/train_agent_trl.py \
      +exp=manager/universal_token/all_modes/sonic_release \
      num_envs=16 headless=True \
      ++algo.config.num_learning_iterations=5
  echo "[smoke] train rc=$?"
else
  echo "[smoke] SKIP training: data/motion_lib_bones_seed/robot_filtered or data/bones_seed_smpl missing"
fi
echo "[smoke] done"
