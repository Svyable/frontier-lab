#!/usr/bin/env bash
# Explicit, reversible bootstrap for a 24 GB Apple Silicon Mac.
# No model download or package installation happens with the default "check".
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-check}"
TEACHER_ENV="$ROOT/.venv-teacher"
MODEL_DIR="$ROOT/models/clef-flash-4bit"
REVISION_FILE="$ROOT/data/teacher_revision.txt"
REPO_ID="mlx-community/clef-flash-4bit"

usage() {
  cat <<'EOF'
Usage: bash scripts/mac_bootstrap.sh [check|prepare|serve|collect]
  check    Validate Apple Silicon and show memory/disk/uv status (default).
  prepare  Create a Python 3.12 environment, install teacher dependencies,
           download exactly one revision of the Clef-Flash MLX model.
  serve    Run the downloaded SystemOne teacher on 127.0.0.1:8001.
  collect  Probe a running teacher, store genuine predictions for 12 synthetic
           seed cases, and export group-held-out MLX chat SFT records.
None of these commands trains or publishes a model.
EOF
}

is_mac() {
  if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "ERROR: this runner targets macOS on Apple Silicon." >&2
    exit 2
  fi
}

check() {
  is_mac
  local bytes disk_kb disk_gib
  bytes="$(sysctl -n hw.memsize)"
  disk_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  disk_gib="$((disk_kb / 1024 / 1024))"
  echo "Device: $(uname -m) / $(sw_vers -productVersion)"
  echo "Physical unified memory: $((bytes / 1024 / 1024 / 1024)) GiB (rounded down)"
  echo "Available filesystem space: ~${disk_gib} GiB"
  echo "uv: $(command -v uv || echo 'NOT INSTALLED')"
  echo "Python: $(command -v python3 || echo 'NOT INSTALLED')"
  echo "Current model directory: $MODEL_DIR"
  if (( disk_gib < 12 )); then
    echo "WARNING: less than 12 GiB free; the ~6.2 GB teacher download is not advised." >&2
  fi
  if (( bytes < 17179869184 )); then
    echo "WARNING: <16 GiB unified memory; this MLX teacher may not fit." >&2
  fi
}

require_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: install uv first (e.g. 'brew install uv'), then rerun prepare." >&2
    exit 2
  fi
}

prepare() {
  check
  require_uv
  local free_kb revision
  free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  if (( free_kb < 12 * 1024 * 1024 )); then
    echo "ERROR: fewer than 12 GiB free; make space before downloading weights." >&2
    exit 2
  fi
  uv venv --python 3.12 "$TEACHER_ENV"
  uv pip install --python "$TEACHER_ENV/bin/python" \
    'mlx-vlm==0.7.4' 'huggingface_hub[hf_xet]'
  revision="$("$TEACHER_ENV/bin/python" - <<'PY'
from huggingface_hub import HfApi
info = HfApi().model_info("mlx-community/clef-flash-4bit")
assert info.sha and len(info.sha) == 40, "Hugging Face did not return a revision SHA"
print(info.sha)
PY
)"
  echo "Downloading $REPO_ID pinned at $revision"
  "$TEACHER_ENV/bin/hf" download "$REPO_ID" \
    --revision "$revision" --local-dir "$MODEL_DIR"
  test -f "$MODEL_DIR/clef_mlx.py" || {
    echo "ERROR: missing clef_mlx.py after download" >&2
    exit 1
  }
  mkdir -p "$ROOT/data"
  printf '%s\n' "$revision" > "$REVISION_FILE"
  echo "Teacher ready. Revision saved under gitignored data/teacher_revision.txt."
  echo "Next: bash scripts/mac_bootstrap.sh serve"
}

serve() {
  is_mac
  if [[ ! -f "$MODEL_DIR/clef_mlx.py" || ! -x "$TEACHER_ENV/bin/python" ]]; then
    echo "ERROR: run 'bash scripts/mac_bootstrap.sh prepare' first." >&2
    exit 2
  fi
  echo "Starting local teacher at 127.0.0.1:8001; Ctrl-C to stop."
  (cd "$MODEL_DIR" && "$TEACHER_ENV/bin/python" clef_mlx.py serve --port 8001)
}

collect() {
  is_mac
  if [[ ! -f "$REVISION_FILE" || ! -x "$TEACHER_ENV/bin/python" ]]; then
    echo "ERROR: run 'prepare' first and 'serve' in a separate terminal." >&2
    exit 2
  fi
  local revision
  revision="$(< "$REVISION_FILE")"
  if [[ ! "$revision" =~ ^[a-f0-9]{40}$ ]]; then
    echo "ERROR: revision file is invalid; rerun prepare." >&2
    exit 2
  fi
  "$TEACHER_ENV/bin/python" -m orinth_clef probe \
    --url http://127.0.0.1:8001
  "$TEACHER_ENV/bin/python" -m orinth_clef collect \
    --url http://127.0.0.1:8001 \
    --revision "$revision" \
    --cases examples/cases.jsonl \
    --output data/raw/clef_12.jsonl
  "$TEACHER_ENV/bin/python" -m orinth_clef export \
    --raw data/raw/clef_12.jsonl \
    --output data/mlx
  echo "Collected 12 teacher predictions. They are NOT human gold labels."
  echo "Saved dataset splits at data/mlx/{train,valid,test}.jsonl."
}

case "$MODE" in
  check) check ;;
  prepare) prepare ;;
  serve) serve ;;
  collect) collect ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
