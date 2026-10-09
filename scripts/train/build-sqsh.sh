#!/usr/bin/env bash
# Build the GR00T-WholeBodyControl (SONIC) training image *on the cluster*:
#   Isaac Lab base sqsh -> enroot container -> pip install gear_sonic[training] -> exported sqsh.
#
# Modeled on psi0's scripts/train/build-sqsh.sh. The difference is the base: SONIC
# training needs Isaac Lab 2.3.x, which runs only inside Isaac Sim's bundled Python 3.11,
# so this starts from NVIDIA's isaac-lab image instead of a plain CUDA base. Torch 2.7
# (cu128), Isaac Sim 5.1 and Isaac Lab all come from that image; inner.sh only layers
# gear_sonic and its training extras on top, with torch pinned so pip can't replace it.
#
#   bash scripts/train/build-sqsh.sh
#
# Overridable:
#   WBC_BASE_SQSH        base sqsh       (default: <containers>/nvidia-isaac-lab-2.3.2.sqsh)
#   WBC_BASE_IMAGE       base image ref  (default: nvidia/isaac-lab:2.3.2)
#   WBC_BASE_REGISTRIES  tried in order  (default: "nvcr.io")
#   WBC_SQSH_OUT         output path     (default: /mnt/beegfs/scratch/$USER/containers/sonic.sqsh)
#   WBC_MIRRORS          git mirror dir  (default: /mnt/beegfs/scratch/$USER/mirrors)
#   WBC_DRY_RUN=1        print resolved paths and exit without building
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)

NAME=${WBC_CONTAINER_NAME:-sonic}
OUT=${WBC_SQSH_OUT:-/mnt/beegfs/scratch/$USER/containers/sonic.sqsh}
MIRRORS=${WBC_MIRRORS:-/mnt/beegfs/scratch/$USER/mirrors}
# Git deps pip would otherwise clone from GitHub: smpl_sim (gear_sonic[training]) and the
# smplx fork smpl_sim itself pins. Keep in step with GIT_DEPS in inner.sh.
GIT_DEPS=(ZhengyiLuo/SMPLSim ZhengyiLuo/smplx)

# The isaac-lab images are anonymous pulls on nvcr.io (they are not on Docker Hub).
BASE_IMAGE=${WBC_BASE_IMAGE:-nvidia/isaac-lab:2.3.2}
read -r -a BASE_REGISTRIES <<< "${WBC_BASE_REGISTRIES:-nvcr.io}"
BASE=${WBC_BASE_SQSH:-/mnt/beegfs/scratch/$USER/containers/$(echo "$BASE_IMAGE" | tr '/:' '--').sqsh}

CACHE=/tmp/pip-cache-$USER
MOUNTS=(-m "/mnt/beegfs:/mnt/beegfs:none:x-create=dir,bind,rw"
        -m "$CACHE:/pip-cache:none:x-create=dir,bind,rw"
        -m "$REPO:/workspace/GR00T-WholeBodyControl:none:x-create=dir,bind,rw")

LOCAL=/tmp/enroot-$USER
export ENROOT_DATA_PATH=$LOCAL/data
export ENROOT_TEMP_PATH=$LOCAL/tmp
export ENROOT_RUNTIME_PATH=$LOCAL/runtime
# Layer cache on BeeGFS: a handful of big write-once tarballs (no small-file penalty),
# and it keeps the ~8.4G of compressed layers off /tmp.
export ENROOT_CACHE_PATH=${WBC_ENROOT_CACHE:-/mnt/beegfs/scratch/$USER/enroot-cache}
export ENROOT_SQUASH_OPTIONS=${ENROOT_SQUASH_OPTIONS:-'-comp zstd -Xcompression-level 10'}

echo "[build] repo:    $REPO -> /workspace/GR00T-WholeBodyControl"
echo "[build] baseimg: $BASE_IMAGE via ${BASE_REGISTRIES[*]}"
echo "[build] base:    $BASE"
echo "[build] out:     $OUT"
echo "[build] mirrors: $MIRRORS (${GIT_DEPS[*]})"
echo "[build] cache:   $CACHE -> /pip-cache"
[ "${WBC_DRY_RUN:-0}" = "1" ] && { echo "[build] dry run, exiting"; exit 0; }

test -f "$REPO/gear_sonic/pyproject.toml" || { echo "[build] FATAL: no GR00T-WholeBodyControl checkout at $REPO"; exit 1; }
mkdir -p "$CACHE" "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH" "$ENROOT_RUNTIME_PATH" \
         "$ENROOT_CACHE_PATH" "$(dirname "$OUT")" "$MIRRORS" "$(dirname "$BASE")"

# Drop a previous build container BEFORE the space check: a leftover Isaac rootfs from
# an aborted run is ~25G, enough to fail the check against a corpse we were about to
# delete anyway.
enroot remove -f "$NAME" 2>/dev/null || true

# Check the QUOTA, not df. /tmp is under a 50G per-user quota on some nodes, so df can
# report plenty free while the quota is exhausted -- the base extract then dies with a
# thoroughly misleading "parallel: Is the disk full?".
NEED=40
quota_headroom_gb() {
    local dev used limit
    dev=$(df --output=source /tmp 2>/dev/null | tail -1) || return 1
    [ -n "$dev" ] || return 1
    read -r used limit < <(quota 2>/dev/null | awk -v d="$dev" '$1==d {gsub(/\*/,"",$2); print $2, $4; exit}')
    [ -n "${limit:-}" ] && [ "${limit:-0}" -gt 0 ] || return 1
    echo $(( (limit - used) / 1024 / 1024 ))
}
if avail=$(quota_headroom_gb); then
    echo "[build] /tmp quota headroom: ${avail}G (need ~${NEED}G)"
    hint="free quota ('enroot list' then 'enroot remove -f <stale>'; /var/lib/h100-enroot counts too)"
else
    avail=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
    echo "[build] /tmp avail: ${avail}G (no quota set; need ~${NEED}G)"
    hint="free node-local space"
fi
[ "$avail" -lt "$NEED" ] && { echo "[build] FATAL: not enough node-local space -- $hint"; exit 1; }

# GitHub over git is slow and resets intermittently from this region, and pip does not
# retry the `git clone` it runs for each git dependency. Pre-fetch once here, where
# retrying is cheap.
for dep in "${GIT_DEPS[@]}"; do
  m="$MIRRORS/${dep#*/}.git"
  if [ ! -d "$m" ]; then
    echo "[build] $(date +%T) cloning $dep mirror (one-off)"
    for i in 1 2 3 4 5; do
      git clone --mirror "https://github.com/$dep.git" "$m" && break
      echo "[build] mirror clone attempt $i failed, retrying"; rm -rf "$m"; sleep 10
    done
  fi
  test -d "$m" || { echo "[build] FATAL: $dep mirror missing at $m"; exit 1; }
done

# One-off base import, cached as a sqsh so later rebuilds skip the pull.
if [ ! -f "$BASE" ]; then
  for reg in "${BASE_REGISTRIES[@]}"; do
    case "$reg" in
      docker.io|"") uri="docker://$BASE_IMAGE" ;;
      *)            uri="docker://$reg#$BASE_IMAGE" ;;
    esac
    echo "[build] $(date +%T) importing base $uri (one-off, ~8.4G compressed)"
    for i in 1 2 3; do
      enroot import -o "$BASE" "$uri" && break 2
      echo "[build] base import attempt $i from $reg failed"; rm -f "$BASE"; sleep 10
    done
    echo "[build] giving up on $reg"
  done
fi
test -f "$BASE" || { echo "[build] FATAL: could not obtain base image $BASE_IMAGE"; exit 1; }

# Nothing in this build touches a GPU, and enroot's 98-nvidia.sh hook hard-fails on
# hosts with a broken libnvidia-container ("nvml error: unknown error").
NO_GPU="-e NVIDIA_VISIBLE_DEVICES=void"

echo "[build] $(date +%T) creating container from $BASE"
enroot create --name "$NAME" "$BASE"

# The isaac-lab image's ENTRYPOINT is /isaac-sim/runheadless.sh, and enroot bakes it into
# /etc/rc so it wraps every `enroot start ... <cmd>` -- our build commands would be fed
# to Kit as arguments. Make rc run the given command (or a shell) instead. pyxis skips
# the entrypoint by default, so Slurm jobs behave the same either way.
RC="$ENROOT_DATA_PATH/$NAME/etc/rc"
if grep -q runheadless "$RC" 2>/dev/null; then
  echo "[build] neutralizing image entrypoint in /etc/rc, was:"; sed 's/^/[build]   /' "$RC"
  printf '%s\n' 'if [ $# -gt 0 ]; then exec "$@"; else exec /bin/bash; fi' > "$RC"
fi

echo "[build] $(date +%T) preflight: repo visible at /workspace/GR00T-WholeBodyControl"
enroot start --root --rw "${MOUNTS[@]}" $NO_GPU "$NAME" \
  bash -c 'test -f /workspace/GR00T-WholeBodyControl/gear_sonic/pyproject.toml && echo "[build] repo mount OK"'

echo "[build] $(date +%T) running build inside container"
enroot start --root --rw "${MOUNTS[@]}" $NO_GPU -e WBC_MIRRORS="$MIRRORS" "$NAME" \
  bash /workspace/GR00T-WholeBodyControl/scripts/train/inner.sh

echo "[build] $(date +%T) exporting to $OUT"
rm -f "$OUT"
enroot export --output "$OUT" "$NAME"
enroot remove -f "$NAME"
ls -lh "$OUT"
echo "[build] $(date +%T) DONE"
echo "[build] install with: sudo cp $OUT /mnt/beegfs/containers/sonic.sqsh"
