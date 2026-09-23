#!/usr/bin/env bash
# ============================================================================
# One command to rebuild llama-v100.cpp at full production performance
# (NCCL allreduce + W3 tensor-core decode + MTP).
#
#   ./v100/build-perf.sh
#
# What it does, in order:
#   1. checks the system pieces that carry the performance, and installs the
#      ones that are missing (NCCL only; the driver and CUDA toolkit are
#      reported, never upgraded, because a driver change needs a reboot);
#   2. configures a fresh build tree with the exact flags the 108 t/s
#      configuration uses;
#   3. builds llama-server + the bench binaries;
#   4. prints the startup-affecting facts and a self-check.
#
# PERFORMANCE-CRITICAL pieces (measured, see v100/RESULTS.md):
#   * NCCL >= 2.31 allreduce  -> prefill 800 -> 1236 t/s (+54%). Zero code.
#     Ubuntu's own libnccl2 (2.18.5) CRASHES with this driver: every
#     ncclAllReduce returns "unhandled cuda error", exit 134. Do not let apt
#     pick it. See RESULTS.md 11.2.
#   * GGML_CUDA_FORCE_CUBLAS=ON -> keeps prefill on the fast path. Turning it
#     off (build-mmq) measured -0.2%, and MMQ for the MTP verify batch measured
#     -29%. See RESULTS.md 14/15.
#   * The W3 kernel (ggml-cuda/mmvq-tc.cu) is already a normal part of the tree
#     at HEAD e7597da9f. It gives decode +8.6~8.8%. Nothing to enable.
#   * GGML_CUDA_GRAPHS / FA_QUANTS / F16 must stay as below; they were part of
#     the measured baseline.
#
# Usage:
#   ./v100/build-perf.sh                 # deps + configure + build
#   BUILD=.../build-foo ./v100/build-perf.sh
#   JOBS=8 ./v100/build-perf.sh
#   ./v100/build-perf.sh deps            # only check/install system packages
#   SKIP_DEPS=1 ./v100/build-perf.sh     # never touch apt
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD:-$ROOT/llama-v100.cpp/build-perf}"
JOBS="${JOBS:-10}"
MODE="${1:-build}"

# --- the exact versions that were measured -----------------------------------
CUDA_REQUIRED="${CUDA_REQUIRED:-12.8}"
NCCL_VERSION="${NCCL_VERSION:-2.31.2-1+cuda12.9}"
NCCL_UBUNTU_PKGS="libnccl2 libnccl-dev"

log()  { printf '\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

have_root() { [ "$(id -u)" -eq 0 ] || sudo -n true 2>/dev/null; }
as_root()   { if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo -n "$@"; fi; }

# ---------------------------------------------------------------------------
# 1. system dependencies
# ---------------------------------------------------------------------------
check_cuda() {
    local nvcc="${CUDA_HOME:-/usr/local/cuda-$CUDA_REQUIRED}/bin/nvcc"
    command -v "$nvcc" >/dev/null 2>&1 || nvcc="$(command -v nvcc || true)"
    [ -n "$nvcc" ] && [ -x "$nvcc" ] || die "nvcc not found. Install cuda-toolkit-$CUDA_REQUIRED."
    local ver
    ver="$("$nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
    [ -n "$ver" ] || die "cannot parse nvcc version"
    case "$ver" in
        "$CUDA_REQUIRED"*) log "CUDA $ver OK ($nvcc)";;
        *) warn "CUDA is $ver, the measured build used $CUDA_REQUIRED. Continuing anyway.";;
    esac
    export CUDA_HOME="$(dirname "$(dirname "$nvcc")")"
}

check_driver() {
    local ver=""
    [ -r /proc/driver/nvidia/version ] && \
        ver="$(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version | head -1)"
    if [ -z "$ver" ]; then
        warn "no NVIDIA kernel module loaded. Install the driver and REBOOT, then re-run."
        return
    fi
    log "NVIDIA driver $ver"
    # NCCL 2.31.2 is built against CUDA 12.9; the 2.18.5 in Ubuntu's archive is
    # incompatible with the 580.x driver. 550 is a conservative lower bound.
    local major="${ver%%.*}"
    if [ "${major:-0}" -lt 550 ]; then
        warn "driver $ver looks too old for NCCL $NCCL_VERSION."
        warn "Measured with nvidia-driver-580 (580.173.02). Upgrading the driver needs a reboot,"
        warn "so this script will not do it for you."
    fi
}

check_nccl() {
    # The dev package must provide both the header and the .so that FindNCCL
    # uses, and the runtime must be the measured version.
    local need_install=0 installed=""
    installed="$(dpkg-query -W -f='${Version}' libnccl2 2>/dev/null || true)"
    if [ "$installed" != "$NCCL_VERSION" ]; then
        need_install=1
    fi
    if [ ! -r /usr/include/nccl.h ] || [ ! -e /usr/lib/x86_64-linux-gnu/libnccl.so ]; then
        need_install=1
    fi

    if [ "$need_install" -eq 0 ]; then
        log "NCCL $installed OK"
        return
    fi

    warn "NCCL $NCCL_VERSION not fully installed (found: '${installed:-none}')"
    if [ "${SKIP_DEPS:-0}" = "1" ]; then
        die "SKIP_DEPS=1 is set, so I will not install it. Run without SKIP_DEPS."
    fi
    have_root || die "need root to install NCCL. Re-run with sudo, or install $NCCL_UBUNTU_PKGS=$NCCL_VERSION yourself."

    log "adding NVIDIA's CUDA apt source (for NCCL only)"
    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN
    curl -fsSL -o "$tmp/cuda-keyring.deb" \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
    as_root dpkg -i "$tmp/cuda-keyring.deb" >/dev/null

    # Pin NCCL to NVIDIA's repo. Ubuntu's multiverse carries an older libnccl2
    # that would win on priority and then crash at runtime (RESULTS.md 11.2).
    as_root tee /etc/apt/preferences.d/nccl-pin >/dev/null <<EOF
Package: libnccl2 libnccl-dev
Pin: origin developer.download.nvidia.com
Pin-Priority: 1001
EOF

    log "installing $NCCL_UBUNTU_PKGS=$NCCL_VERSION (large download, ~470 MB)"
    as_root apt-get update -qq
    as_root apt-get install -y --no-install-recommends \
        "${NCCL_UBUNTU_PKGS%% *}=$NCCL_VERSION" "${NCCL_UBUNTU_PKGS##* }=$NCCL_VERSION"

    installed="$(dpkg-query -W -f='${Version}' libnccl2 2>/dev/null || true)"
    [ "$installed" = "$NCCL_VERSION" ] || die "install finished but libnccl2 is '$installed'"
    log "NCCL $installed installed"
}

check_gcc() {
    if command -v gcc >/dev/null 2>&1; then
        log "gcc $(gcc -dumpversion)"
    fi
}

log "checking system dependencies (driver / CUDA / NCCL)"
check_cuda
check_driver
check_nccl
check_gcc

if [ "$MODE" = deps ]; then
    log "deps only, nothing to build"
    exit 0
fi

# ---------------------------------------------------------------------------
# 2+3. configure and build
# ---------------------------------------------------------------------------
log "building into $BUILD (-j$JOBS)"

# build-v100.sh needs nvcc on PATH when CUDA_HOME is not the default location
if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    case ":$PATH:" in *":$CUDA_HOME/bin:"*) ;; *) PATH="$CUDA_HOME/bin:$PATH";; esac
    export PATH
fi

# The two flags that carry the measured performance. Everything else
# (arch=70, FA_QUANTS, graphs, shared libs, Release) is already in
# build-v100.sh and must not be changed.
BUILD="$BUILD" JOBS="$JOBS" "$ROOT/v100/build-v100.sh" \
    -DGGML_CUDA_NCCL=ON \
    -DGGML_CUDA_FORCE_CUBLAS=ON

# ---------------------------------------------------------------------------
# 4. verify the build actually has the performance-critical pieces
# ---------------------------------------------------------------------------
log "self-check"
SERVER="$BUILD/bin/llama-server"
[ -x "$SERVER" ] || die "$SERVER was not produced"

fail=0
if ldd "$SERVER" | grep -q "libnccl.so.2"; then
    echo "  NCCL linked      : $(ldd "$SERVER" | awk '/libnccl.so.2/{print $3}')"
else
    echo "  NCCL linked      : NO  <-- prefill will stay at ~800 t/s"; fail=1
fi
if nm -D "$BUILD/bin/libggml-cuda.so.0" 2>/dev/null | grep -q "ncclAllReduce"; then
    echo "  ncclAllReduce    : present in libggml-cuda"
else
    echo "  ncclAllReduce    : MISSING  <-- CMake did not enable GGML_USE_NCCL"; fail=1
fi
echo "  FORCE_CUBLAS     : $(grep -E '^GGML_CUDA_FORCE_CUBLAS:' "$BUILD/CMakeCache.txt" | cut -d= -f2)"
echo "  CUDA arch        : $(grep -E '^CMAKE_CUDA_ARCHITECTURES' "$BUILD/CMakeCache.txt" | cut -d= -f2)"
echo "  W3 kernel (mmvq-tc): $([ -f "$ROOT/llama-v100.cpp/ggml/src/ggml-cuda/mmvq-tc.cu" ] && echo present || echo MISSING)"
RP="$(readelf -d "$SERVER" | sed -n 's/.*RUNPATH.*\[\(.*\)\]/\1/p')"
echo "  RUNPATH          : ${RP:-none}"
case "$RP" in
    *llama.cpp/build*) echo "  ^^ WARNING: RUNPATH points into the official llama.cpp tree"; fail=1;;
esac

if [ "$fail" -ne 0 ]; then
    warn "self-check found problems; this build will NOT reach the measured numbers"
    exit 1
fi

log "done: $BUILD/bin"
cat <<EOF

Run the server with:

  $BUILD/bin/llama-server \\
      --parallel 1 --ctx-size 200000 --kv-unified \\
      --cache-type-k q8_0 --cache-type-v q8_0 \\
      --batch-size 8192 --ubatch-size 1024 --flash-attn on \\
      -ngl 999 --tensor-split 1,1 --split-mode tensor \\
      --model  models/Qwen3.8-27B-UD-Q4_K_XL.gguf \\
      --model-draft models/mtp-Qwen3.8-27B-Q4_0.gguf \\
      --spec-draft-n-max 7 --spec-draft-ngl all \\
      --host 127.0.0.1 --port 8080 --metrics

Measured on that recipe (30K depth, greedy, MTP n-max 7):
  prefill  1236 t/s
  decode    98.4 t/s at n_predict=64, 103.5 t/s at n_predict=512   (steady state)
  draft      445/455 accepted (98%)

When you measure decode, discard the FIRST generation after the prefill:
it pays a fixed ~2.2 ms/step, which is 22% at n_predict=64 and 0.4% at 512.
RESULTS.md 36 has the data; v100/warmseq.sh reproduces it.
EOF
