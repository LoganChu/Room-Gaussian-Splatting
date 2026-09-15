#!/usr/bin/env bash
# Build the training toolchain (PLAN.md Phase 0b) on native Ubuntu 24.04.
#
# Four CUDA extensions compile here (gsplat, fused-ssim, fused-bilagrid, ppisp),
# so this takes a while and can fail partway. Every step is idempotent and can be
# run alone:  ./scripts/setup_env.sh --step gsplat
#
# Steps 1-2 need sudo (apt). Everything after is confined to the repo's .venv.
#
#   ./scripts/setup_env.sh            # all steps, in order
#   ./scripts/setup_env.sh --check    # verify an existing install, change nothing
#   ./scripts/setup_env.sh --list     # show step names

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# --- Pins -------------------------------------------------------------------
# gsplat 1.6.0 is unreleased (newest tag and PyPI are 1.5.3), so the pin is a
# commit. This is the commit scripts/verify_gsplat_compat.py was checked against.
GSPLAT_COMMIT="28e794ca44a4c25ffc39175370c5ee7b38bfcc36"
GSPLAT_REPO="https://github.com/nerfstudio-project/gsplat"
# gsplat's examples/requirements.txt pins these two; the cu130 index is the only
# one carrying a CUDA 13 build of 2.9.1 for cp312/linux.
TORCH_VERSION="2.9.1"
TORCHVISION_VERSION="0.24.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
# 13.2 to match what the installed driver reports. A newer toolkit would likely
# work (CUDA minor-version compatibility) but there is nothing to gain from it.
CUDA_PKG="cuda-toolkit-13-2"
CUDA_HOME_DIR="/usr/local/cuda-13.2"
PYTHON_VERSION="3.12"          # matches pyproject's requires-python
ARCH_LIST="12.0"               # sm_120, Blackwell / RTX 5090

VENDOR_DIR="$REPO/third_party/gsplat_examples"
LOCKFILE="$REPO/requirements-train.lock"
# ~2 GB of RAM per nvcc job; leave headroom on a 30 GB machine.
MAX_JOBS="${MAX_JOBS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}"
# The CUDA wheels torch pulls are huge (cuBLAS alone is 400 MB) and uv's default
# 30s per-request timeout aborts the whole install partway through one of them.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

STEPS=(apt cuda venv gsplat vendor examples freeze verify)

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# uv keeps its venv at .venv; call it explicitly so nothing depends on activation.
VPY="$REPO/.venv/bin/python"
uvpip() { uv pip install --python "$VPY" "$@"; }

# --- Preflight --------------------------------------------------------------

preflight() {
  [[ -r /etc/os-release ]] && . /etc/os-release || true
  [[ "${VERSION_ID:-}" == "24.04" ]] || info "WARNING: expected Ubuntu 24.04, found ${PRETTY_NAME:-unknown}"

  have nvidia-smi || die "nvidia-smi not found. The driver is Phase 0a and must be working first."
  local drv
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  info "driver $drv, $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  # Blackwell needs r580+; anything older cannot run sm_120 at all.
  [[ "${drv%%.*}" -ge 580 ]] || die "driver $drv is older than r580, which Blackwell requires."

  have uv || die "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
  info "MAX_JOBS=$MAX_JOBS for nvcc"
}

# --- Steps ------------------------------------------------------------------

# Ubuntu splits Python's C headers into python3-dev; build-essential does NOT
# pull them in, and the venv is built on the system interpreter. Without them
# every torch CUDA extension dies on "fatal error: Python.h: No such file or
# directory" — but only after several minutes of compiling, so check up front.
has_python_headers() { compgen -G "/usr/include/python3*/Python.h" >/dev/null 2>&1; }

step_apt() {
  say "1/8  Build tools (sudo)"
  if have gcc && have g++ && have ninja && has_python_headers; then
    info "gcc $(gcc -dumpversion), g++, ninja, Python.h already present — skipping"
    return
  fi
  sudo apt-get update
  # gcc 13 is the 24.04 default and a supported host compiler for CUDA 13.
  sudo apt-get install -y build-essential ninja-build python3-dev git curl
  has_python_headers || die "python3-dev installed but no Python.h under /usr/include/python3*/"
  info "gcc $(gcc -dumpversion), Python.h OK"
}

step_cuda() {
  say "2/8  CUDA toolkit $CUDA_PKG (sudo)"
  if [[ -x "$CUDA_HOME_DIR/bin/nvcc" ]]; then
    info "$("$CUDA_HOME_DIR/bin/nvcc" --version | tail -2 | head -1)"
    info "already installed — skipping"
    return
  fi
  if ! dpkg -l cuda-keyring >/dev/null 2>&1; then
    local deb="/tmp/cuda-keyring_1.1-1_all.deb"
    curl -fsSL -o "$deb" \
      "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i "$deb"
    rm -f "$deb"
    sudo apt-get update
  fi
  # The toolkit component ONLY. The `cuda` meta-package would pull a driver and
  # could downgrade the working 595.84.
  sudo apt-get install -y "$CUDA_PKG"
  [[ -x "$CUDA_HOME_DIR/bin/nvcc" ]] || die "$CUDA_PKG installed but $CUDA_HOME_DIR/bin/nvcc is missing"
  info "$("$CUDA_HOME_DIR/bin/nvcc" --version | tail -2 | head -1)"
}

# Exported for every step that compiles.
cuda_env() {
  [[ -x "$CUDA_HOME_DIR/bin/nvcc" ]] || die "nvcc not found at $CUDA_HOME_DIR — run: $0 --step cuda"
  has_python_headers || die "Python.h not found — run: $0 --step apt"
  export CUDA_HOME="$CUDA_HOME_DIR"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
  export MAX_JOBS
}

step_venv() {
  say "3/8  Python $PYTHON_VERSION env and torch $TORCH_VERSION+cu130"
  [[ -x "$VPY" ]] || uv venv --python "$PYTHON_VERSION"
  # The SfM deps (pillow, pycolmap) stay locked in pyproject/uv.lock.
  uv sync --inexact       # --inexact: do not remove the training deps added below
  # Build backend. Every CUDA extension below installs with --no-build-isolation,
  # which means its build sees ONLY this venv — so the backend has to live here.
  # `uv venv` ships none, and without this all four extension builds fail with
  # "Cannot import setuptools.build_meta".
  uvpip setuptools wheel ninja
  uvpip "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"
  "$VPY" - <<'PY'
import torch
cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
print(f"    torch {torch.__version__}, built for CUDA {torch.version.cuda}, device capability {cap}")
assert cap == (12, 0), f"expected sm_120, got {cap}"
PY
}

step_gsplat() {
  say "4/8  gsplat @ ${GSPLAT_COMMIT:0:8} (compiles CUDA — slow)"
  cuda_env
  # --no-build-isolation: the build needs the torch already in this venv.
  uvpip --no-build-isolation "gsplat @ git+$GSPLAT_REPO@$GSPLAT_COMMIT"
  "$VPY" -c "import gsplat; print(f'    gsplat {gsplat.__version__}')"
}

step_examples() {
  say "6/8  Example dependencies (fused-ssim, fused-bilagrid, ppisp — 3 more CUDA builds)"
  cuda_env
  local req="$VENDOR_DIR/requirements.txt"
  [[ -f "$req" ]] || die "$req missing — run: $0 --step vendor"
  # torch/torchvision are already installed from the cu130 index; re-resolving
  # them here would pull the public PyPI build and break the CUDA 13 install.
  local filtered="/tmp/gsplat-examples-requirements.txt"
  grep -vE '^\s*(torch|torchvision)\s*==' "$req" > "$filtered"
  uvpip --no-build-isolation -r "$filtered"
  rm -f "$filtered"
}

step_vendor() {
  say "5/8  Vendor examples/ at the same commit"
  if [[ -d "$VENDOR_DIR" ]] && [[ -f "$VENDOR_DIR/.gsplat_commit" ]] \
     && [[ "$(cat "$VENDOR_DIR/.gsplat_commit")" == "$GSPLAT_COMMIT" ]]; then
    info "already vendored at ${GSPLAT_COMMIT:0:8} — skipping"
    return
  fi
  local tmp; tmp="$(mktemp -d)"
  git clone --quiet --filter=blob:none --no-checkout "$GSPLAT_REPO" "$tmp/gsplat"
  git -C "$tmp/gsplat" sparse-checkout set --no-cone examples
  git -C "$tmp/gsplat" checkout --quiet "$GSPLAT_COMMIT"
  mkdir -p "$(dirname "$VENDOR_DIR")"
  rm -rf "$VENDOR_DIR"
  cp -r "$tmp/gsplat/examples" "$VENDOR_DIR"
  echo "$GSPLAT_COMMIT" > "$VENDOR_DIR/.gsplat_commit"
  rm -rf "$tmp"
  info "vendored to ${VENDOR_DIR#$REPO/} — mark local edits with '# [splat]'"
}

step_freeze() {
  say "7/8  Freeze the training stack"
  # uv.lock covers only the SfM deps declared in pyproject. The training stack is
  # installed imperatively (a git commit, a custom index, --no-build-isolation),
  # so its reproducibility artifact is this freeze plus the pins at the top.
  uv pip freeze --python "$VPY" > "$LOCKFILE"
  info "wrote ${LOCKFILE#$REPO/} ($(wc -l < "$LOCKFILE") packages)"
}

step_verify() {
  say "8/8  Verify"
  "$VPY" - <<'PY'
import torch, gsplat
print(f"    torch     {torch.__version__}  cuda={torch.version.cuda}")
print(f"    gsplat    {gsplat.__version__}")
cap = torch.cuda.get_device_capability()
assert cap == (12, 0), f"expected sm_120, got {cap}"
print(f"    device    {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")

# Rasterize one Gaussian. This is the real check: it exercises the compiled
# CUDA kernel, not just the import.
import torch as t
means   = t.tensor([[0.0, 0.0, 2.0]], device="cuda")
quats   = t.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
scales  = t.tensor([[0.1, 0.1, 0.1]], device="cuda")
opacity = t.tensor([1.0], device="cuda")
colors  = t.tensor([[1.0, 0.5, 0.2]], device="cuda")
K = t.tensor([[[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]]], device="cuda")
out, alpha, _ = gsplat.rasterization(
    means, quats, scales, opacity, colors, t.eye(4, device="cuda")[None], K, 64, 64
)
assert out.shape == (1, 64, 64, 3), out.shape
assert alpha.max() > 0, "rendered nothing — the Gaussian was not rasterized"
print(f"    rasterize OK  image={tuple(out.shape)}  peak_alpha={alpha.max():.3f}")
PY
  local vend="$VENDOR_DIR/simple_trainer.py"
  [[ -f "$vend" ]] && info "vendored trainer: ${vend#$REPO/}" || info "WARNING: $vend missing"
  say "Phase 0b complete"
  cat <<EOF
    Next (PLAN.md Phase 1):
      .venv/bin/python scripts/verify_gsplat_compat.py
      .venv/bin/python ${VENDOR_DIR#$REPO/}/simple_trainer.py default \\
          --data_dir data/scenes/room-1 --data_factor 4 --max_steps 7000
EOF
}

# --- Driver -----------------------------------------------------------------

case "${1:-}" in
  --list)  printf '%s\n' "${STEPS[@]}"; exit 0 ;;
  --check) preflight; step_verify; exit 0 ;;
  --step)
    [[ -n "${2:-}" ]] || die "--step needs a name (see --list)"
    [[ " ${STEPS[*]} " == *" $2 "* ]] || die "unknown step '$2' (see --list)"
    preflight; "step_$2"; exit 0 ;;
  "") ;;
  *) die "unknown option '$1' (--list, --check, --step NAME)" ;;
esac

preflight
for s in "${STEPS[@]}"; do "step_$s"; done
