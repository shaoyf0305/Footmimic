#!/usr/bin/env bash
#
# Video -> GVHMR -> GMR retarget -> Footmimic compatible pkl -> npz
#
# All intermediates live under:  Footmimic/pipeline/<batch>/
#   videos/           input .mp4
#   gvhmr/            per-video GVHMR outputs (*/hmr4d_results.pt)
#   gmr/              retargeted .pkl  ({stem}_{robot}.pkl)
#   pkl_compatible/   soccer-format .pkl
#   npz/              final .npz for Isaac training / replay
#
# One-liner (4 steps: GVHMR -> GMR -> compatible pkl -> npz via pkl_to_npz.py):
#   bash shell/video_to_npz_pipeline.sh --batch 0522
#
# T1 retarget, videos already in GVHMR usage folder:
#   bash shell/video_to_npz_pipeline.sh --batch 0522 \
#     --robot booster_t1 --video-dir /path/to/videos
#
# Re-run from GMR only (GVHMR already done):
#   bash shell/video_to_npz_pipeline.sh --batch 0522 --from gmr
#
# Copy final npz into motions/ for training:
#   bash shell/video_to_npz_pipeline.sh --batch 0522 --publish motions/my_batch
#

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FOOTMIMIC_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

GVHMR_ROOT="${GVHMR_ROOT:-/data/GVHMR}"
GMR_ROOT="${GMR_ROOT:-/data/GMR}"
ISAAC_SETUP="${ISAAC_SETUP:-/data/isaacsim/setup_conda_env.sh}"

CONDA_SH="${CONDA_SH:-/data/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

ENV_GVHMR="${ENV_GVHMR:-gvhmr}"
ENV_GMR="${ENV_GMR:-gmr}"
ENV_ISAAC="${ENV_ISAAC:-isaaclab_211}"

ROBOT="unitree_g1"
BATCH=""
VIDEO_DIR=""
FROM_STEP="gvhmr"
STATIC_CAM=true
NORMALIZE_YAW=false
TARGET_YAW="-90.0"
RECORD_VIDEO=false
PUBLISH_DIR=""
SKIP_EXISTING=false

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --batch NAME          Required. Workspace: pipeline/NAME/
  --robot NAME          GMR --robot (default: unitree_g1). e.g. booster_t1, unitree_h1
  --video-dir PATH      Folder of .mp4 inputs (default: pipeline/BATCH/videos)
  --from STEP           Start at: gvhmr | gmr | convert | npz (default: gvhmr)
  --no-static-cam       Do not pass -s to GVHMR (enable DPVO)
  --normalize-yaw       Pass --normalize_yaw to convert_gmr_to_soccer
  --target-yaw DEG      With --normalize-yaw (default: -90)
  --record-video        GMR --record_video
  --publish DIR         Copy npz/*.npz -> FOOTMIMIC/motions/DIR/
  --skip-existing       Skip outputs that already exist
  -h, --help            Show this help

Environment overrides:
  GVHMR_ROOT, GMR_ROOT, CONDA_SH, ENV_GVHMR, ENV_GMR, ENV_ISAAC, ISAAC_SETUP
EOF
}

step_ge() {
  local want="$1" current="$2"
  case "${current}" in
    gvhmr)   [[ "${want}" == gvhmr ]] && return 0 ;;
    gmr)     [[ "${want}" == gvhmr || "${want}" == gmr ]] && return 0 ;;
    convert) [[ "${want}" == gvhmr || "${want}" == gmr || "${want}" == convert ]] && return 0 ;;
    npz)     return 0 ;;
    *) echo "[ERROR] Unknown step: ${want}" >&2; return 1 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch) BATCH="$2"; shift 2 ;;
    --robot) ROBOT="$2"; shift 2 ;;
    --video-dir) VIDEO_DIR="$2"; shift 2 ;;
    --from) FROM_STEP="$2"; shift 2 ;;
    --no-static-cam) STATIC_CAM=false; shift ;;
    --normalize-yaw) NORMALIZE_YAW=true; shift ;;
    --target-yaw) TARGET_YAW="$2"; shift 2 ;;
    --record-video) RECORD_VIDEO=true; shift ;;
    --publish) PUBLISH_DIR="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${BATCH}" ]]; then
  echo "[ERROR] --batch is required." >&2
  usage
  exit 1
fi

case "${FROM_STEP}" in
  gvhmr|gmr|convert|npz) ;;
  *) echo "[ERROR] --from must be gvhmr, gmr, convert, or npz" >&2; exit 1 ;;
esac

BATCH_ROOT="${FOOTMIMIC_ROOT}/pipeline/${BATCH}"
DIR_VIDEOS="${BATCH_ROOT}/videos"
DIR_GVHMR="${BATCH_ROOT}/gvhmr"
DIR_GMR="${BATCH_ROOT}/gmr"
DIR_COMPAT="${BATCH_ROOT}/pkl_compatible"
DIR_NPZ="${BATCH_ROOT}/npz"

if [[ -z "${VIDEO_DIR}" ]]; then
  VIDEO_DIR="${DIR_VIDEOS}"
fi

mkdir -p "${DIR_GVHMR}" "${DIR_GMR}" "${DIR_COMPAT}" "${DIR_NPZ}"
if [[ "${VIDEO_DIR}" == "${DIR_VIDEOS}" ]]; then
  mkdir -p "${DIR_VIDEOS}"
fi

if step_ge "${FROM_STEP}" gvhmr; then
  if ! compgen -G "${VIDEO_DIR}"/*.mp4 >/dev/null 2>&1 && \
     ! compgen -G "${VIDEO_DIR}"/*.MP4 >/dev/null 2>&1; then
    echo "[ERROR] No .mp4 files in ${VIDEO_DIR}" >&2
    echo "        Put videos in pipeline/${BATCH}/videos or pass --video-dir" >&2
    exit 1
  fi
fi

[[ -f "${CONDA_SH}" ]] || { echo "[ERROR] conda.sh not found: ${CONDA_SH}" >&2; exit 1; }
# shellcheck source=/dev/null
source "${CONDA_SH}"

log() { echo "[pipeline] $*"; }

# Isaac's setup_conda_env.sh tests $ZSH_VERSION; with set -u that aborts the pipeline.
source_isaac_env() {
  if [[ ! -f "${ISAAC_SETUP}" ]]; then
    echo "[WARN] ISAAC_SETUP not found (${ISAAC_SETUP})" >&2
    return 0
  fi
  set +u
  # shellcheck source=/dev/null
  source "${ISAAC_SETUP}"
  set -u
}

run_gvhmr() {
  log "Step 1/4: GVHMR (env=${ENV_GVHMR})"
  conda activate "${ENV_GVHMR}"
  cd "${GVHMR_ROOT}"
  local extra=()
  if ${STATIC_CAM}; then extra+=(-s); fi
  python tools/demo/demo_folder.py \
    -f "${VIDEO_DIR}" \
    -d "${DIR_GVHMR}" \
    "${extra[@]}"
}

run_gmr() {
  log "Step 2/4: GMR retarget -> ${DIR_GMR} (robot=${ROBOT}, env=${ENV_GMR})"
  conda activate "${ENV_GMR}"
  cd "${GMR_ROOT}"

  local pt_files=()
  mapfile -t pt_files < <(find "${DIR_GVHMR}" -name "hmr4d_results.pt" | sort)
  if [[ ${#pt_files[@]} -eq 0 ]]; then
    echo "[ERROR] No hmr4d_results.pt under ${DIR_GVHMR}. Run GVHMR first." >&2
    exit 1
  fi

  local f dir stem out extra=()
  if ${RECORD_VIDEO}; then extra+=(--record_video); fi

  for f in "${pt_files[@]}"; do
    dir="$(dirname "${f}")"
    stem="$(basename "${dir}")"
    out="${DIR_GMR}/${stem}_${ROBOT}.pkl"
    if ${SKIP_EXISTING} && [[ -f "${out}" ]]; then
      log "  skip (exists): ${out}"
      continue
    fi
    log "  ${stem}: ${f} -> ${out}"
    python scripts/gvhmr_to_robot.py \
      --gvhmr_pred_file "${f}" \
      --robot "${ROBOT}" \
      --save_path "${out}" \
      "${extra[@]}"
  done
}

run_convert() {
  log "Step 3/4: convert_gmr_to_soccer -> ${DIR_COMPAT} (env=${ENV_ISAAC}, no Isaac Sim launch)"
  conda activate "${ENV_ISAAC}"
  cd "${FOOTMIMIC_ROOT}"

  local pkl_files=()
  mapfile -t pkl_files < <(find "${DIR_GMR}" -maxdepth 1 -name "*.pkl" | sort)
  if [[ ${#pkl_files[@]} -eq 0 ]]; then
    echo "[ERROR] No .pkl in ${DIR_GMR}. Run GMR step first." >&2
    exit 1
  fi

  local f base out yaw_args=()
  if ${NORMALIZE_YAW}; then
    yaw_args=(--normalize_yaw --target_yaw "${TARGET_YAW}")
  fi

  for f in "${pkl_files[@]}"; do
    base="$(basename "${f}")"
    out="${DIR_COMPAT}/${base}"
    if ${SKIP_EXISTING} && [[ -f "${out}" ]]; then
      log "  skip (exists): ${out}"
      continue
    fi
    log "  ${base}"
    python scripts/convert_gmr_to_soccer.py \
      --input "${f}" \
      --output "${out}" \
      "${yaw_args[@]}"
  done
}

run_npz() {
  log "Step 4/4: pkl_to_npz -> ${DIR_NPZ} (env=${ENV_ISAAC}, headless)"
  conda activate "${ENV_ISAAC}"
  source_isaac_env
  cd "${FOOTMIMIC_ROOT}"

  if [[ -z "$(find "${DIR_COMPAT}" -maxdepth 1 -name '*.pkl' -print -quit 2>/dev/null)" ]]; then
    echo "[ERROR] No .pkl in ${DIR_COMPAT}. Run convert step first." >&2
    exit 1
  fi

  python scripts/pkl_to_npz.py \
    --input_dir "${DIR_COMPAT}" \
    --output_dir "${DIR_NPZ}" \
    --headless
}

publish_motions() {
  local dest="${FOOTMIMIC_ROOT}/${PUBLISH_DIR}"
  mkdir -p "${dest}"
  log "Publishing npz -> ${dest}"
  cp -v "${DIR_NPZ}"/*.npz "${dest}/"
}

log "Batch workspace: ${BATCH_ROOT}"
log "Videos: ${VIDEO_DIR}"
log "Robot: ${ROBOT} | From step: ${FROM_STEP}"

if step_ge "${FROM_STEP}" gvhmr; then run_gvhmr; fi
if step_ge "${FROM_STEP}" gmr; then run_gmr; fi
if step_ge "${FROM_STEP}" convert; then run_convert; fi
if step_ge "${FROM_STEP}" npz; then run_npz; fi

if [[ -n "${PUBLISH_DIR}" ]]; then
  publish_motions
fi

n_compat="$(find "${DIR_COMPAT}" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l)"
n_npz="$(find "${DIR_NPZ}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)"
log "Done. pkl_compatible=${n_compat}  npz=${n_npz}"
if step_ge "${FROM_STEP}" convert && [[ "${n_compat}" -eq 0 ]]; then
  echo "[ERROR] Step 3 produced no pkl_compatible files." >&2
  exit 1
fi
if step_ge "${FROM_STEP}" npz && [[ "${n_npz}" -eq 0 ]]; then
  echo "[ERROR] Step 4 produced no npz files (check Isaac Sim / pkl_to_npz logs above)." >&2
  exit 1
fi

log "Paths:"
log "  gvhmr:          ${DIR_GVHMR}"
log "  gmr pkl:        ${DIR_GMR}"
log "  compatible pkl: ${DIR_COMPAT}"
log "  npz:            ${DIR_NPZ}"
