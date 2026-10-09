#!/usr/bin/env bash
# Runs INSIDE the enroot container as root, invoked by build-sqsh.sh.
#
# The base is nvcr.io/nvidia/isaac-lab:2.3.2: Isaac Sim 5.1 at /isaac-sim, Isaac Lab
# at /workspace/isaaclab, torch 2.7 (cu128) already in Isaac Sim's Python 3.11. Isaac Lab
# only imports under that interpreter, so there is no venv here -- gear_sonic goes into
# Isaac Sim's site-packages via /isaac-sim/python.sh, the way Isaac Lab's own docs do it.
#
# `--root` here means UID 0 inside the container's user namespace only; no host
# privileges are involved.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

REPO=${WBC_REPO:-/workspace/GR00T-WholeBodyControl}
MIRRORS=${WBC_MIRRORS:-}
GIT_DEPS=(ZhengyiLuo/SMPLSim ZhengyiLuo/smplx)
PY=/isaac-sim/python.sh

retry() {
  local n=1 max=6
  until "$@"; do
    if [ $n -ge $max ]; then echo "[build] FATAL: '$*' failed after $max attempts"; return 1; fi
    echo "[build] attempt $n/$max failed, retrying in $((n*10))s: $*"
    sleep $((n*10)); n=$((n+1))
  done
}

# The isaac-lab image already has git, cmake and build-essential; git-lfs is for the
# LFS assets in the repo, ffmpeg for the video recorders in gear_sonic/envs.
retry apt-get update
retry apt-get install -y --no-install-recommends git-lfs ffmpeg
rm -rf /var/lib/apt/lists/*

# pip shells out to git for smpl_sim and its smplx fork; point those URLs at the mirrors
# build-sqsh.sh pre-fetched, so the build never hits GitHub over git.
for dep in "${GIT_DEPS[@]}"; do
  m="$MIRRORS/${dep#*/}.git"
  if [ -n "$MIRRORS" ] && [ -d "$m" ]; then
    git config --global url."$m".insteadOf "https://github.com/$dep.git"
    echo "[build] using local mirror for $dep: $m"
  else
    echo "[build] WARNING: no mirror for $dep, falling back to GitHub (slow here)"
  fi
done
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 60
# The repo is bind-mounted and owned by the host user, not container root.
git config --global --add safe.directory "$REPO"

export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/pip-cache}
export PIP_DISABLE_PIP_VERSION_CHECK=1
export GIT_LFS_SKIP_SMUDGE=1

# Freeze the torch stack Isaac Lab was built against. gear_sonic only asks for
# torch>=2.4, but transformers/trl/accelerate would happily drag in a newer torch and
# break Isaac Sim's CUDA extensions. numpy/scipy are left out on purpose: gear_sonic
# pins its own (numpy==1.26.4, scipy==1.15.3).
CONSTRAINTS=/opt/wbc-constraints.txt
$PY -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|warp-lang)==' > "$CONSTRAINTS"
# pyarrow 26 needs NumPy 2 but doesn't declare it, so pip pairs it with numpy 1.26.4 and
# `import datasets` (pulled in by trl's PPO trainer) dies. 25.x is the last that works.
echo "pyarrow<26" >> "$CONSTRAINTS"
cat "$CONSTRAINTS"

cd "$REPO"
# wandb: gear_sonic leaves it unpinned and pip keeps whatever is around; 0.24.0 has a
# known bug where run data can silently fail to upload, so require a fixed release.
# smpl_sim/open3d/vector_quantize_pytorch: listed in main's gear_sonic[training] but
# dropped from it on some branches (hrsi) whose code still needs them -- open3d in
# motion_lib, smpl_sim in the eval callback, and vector_quantize_pytorch as a config
# `_target_` of the universal-token model (no import line to grep for). Install them
# explicitly so the image doesn't depend on the checkout.
retry $PY -m pip install -c "$CONSTRAINTS" -e "gear_sonic/[training]" huggingface_hub "wandb>=0.29" \
    "smpl_sim @ git+https://github.com/ZhengyiLuo/SMPLSim.git" open3d vector_quantize_pytorch

# Imports that work without launching Kit (isaaclab.app needs a GPU, which the build
# deliberately has none of). The full Isaac Lab check is check_environment.py on a GPU node.
$PY -c "import torch, isaaclab, gear_sonic, trl, transformers, accelerate, hydra, wandb, smpl_sim, datasets, pyarrow, open3d, vector_quantize_pytorch; from trl import PPOConfig; print('VERIFY', torch.__version__, 'cuda', torch.version.cuda, 'isaaclab', getattr(isaaclab, '__version__', '?'), 'trl', trl.__version__, 'wandb', wandb.__version__, gear_sonic.__file__)"
# Imported on the training path but not declared in gear_sonic[training]; report rather
# than fail, so a missing one shows up here instead of 10 minutes into a Slurm job.
$PY - <<'EOF'
import importlib
missing = []
for m in ["open3d", "tensordict", "peft", "pandas", "rich", "lxml", "imageio", "einops", "vector_quantize_pytorch"]:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m} ({type(e).__name__}: {e})")
print("VERIFY optional imports:", "all OK" if not missing else "MISSING " + "; ".join(missing))
EOF

# `python` and `accelerate` on PATH resolve to Isaac Sim's interpreter with its env set,
# so the README's `python gear_sonic/train_agent_trl.py ...` and `accelerate launch ...`
# work unchanged. Child processes accelerate spawns inherit python.sh's environment.
cat > /usr/local/bin/python <<'EOF'
#!/bin/bash
exec /isaac-sim/python.sh "$@"
EOF
cat > /usr/local/bin/accelerate <<'EOF'
#!/bin/bash
exec /isaac-sim/python.sh -m accelerate.commands.accelerate_cli "$@"
EOF
chmod +x /usr/local/bin/python /usr/local/bin/accelerate

# Bake the env into the image: enroot exports /etc/environment as the image ENV.
if ! grep -q '^OMNI_KIT_ACCEPT_EULA=' /etc/environment; then
  {
    echo "ACCEPT_EULA=Y"
    echo "OMNI_KIT_ACCEPT_EULA=YES"
    echo "PRIVACY_CONSENT=Y"
    echo "GIT_LFS_SKIP_SMUDGE=1"
  } >> /etc/environment
fi
grep -E '^(PATH|ISAAC|OMNI_KIT|ACCEPT_EULA|NVIDIA_DRIVER)' /etc/environment
echo "[build] inner done"
