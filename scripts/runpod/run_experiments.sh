#!/usr/bin/env bash
# Train and evaluate a set of configs x seeds on one GPU pod, sequentially.
#
# Run from the repo root inside a RunPod PyTorch pod (see docs/runpod.md):
#   bash scripts/runpod/run_experiments.sh                       # baseline, seeds 0 1 2
#   CONFIGS="baseline rgb_only no_synthetic" bash scripts/runpod/run_experiments.sh
#
# Safe to re-run after a stop or spot interruption: finished runs (eval files
# present) are skipped and unfinished ones resume from last.pt, which
# reproduces an uninterrupted run.
#
# Environment (all optional):
#   CONFIGS      config names from configs/ (default: baseline)
#   SEEDS        default: "0 1 2"
#   DATA_DIR     extracted dataset with manifest.json (default /workspace/colab_data_v2)
#   DATA_ZIP     zip to extract when DATA_DIR is missing (default /workspace/colab_data_v2.zip)
#   OUT_DIR      run folders + results; keep it on the volume (default /workspace/runs)
#   NUM_WORKERS  data loader workers (default 8)
#   EXTRA_SET    extra --set overrides, e.g. "num_epochs=60"
#   SPLITS       test splits to evaluate (default "test_real test_roboflow")
#   AUTO_STOP=1  stop the pod when the script ends (success or failure) to stop GPU billing
#   PIP_INSTALL=0 / REQUIRE_CUDA=0  skip dependency install / GPU check (local smoke tests)

set -uo pipefail

CONFIGS=${CONFIGS:-baseline}
SEEDS=${SEEDS:-"0 1 2"}
DATA_DIR=${DATA_DIR:-/workspace/colab_data_v2}
DATA_ZIP=${DATA_ZIP:-/workspace/colab_data_v2.zip}
OUT_DIR=${OUT_DIR:-/workspace/runs}
NUM_WORKERS=${NUM_WORKERS:-8}
EXTRA_SET=${EXTRA_SET:-}
SPLITS=${SPLITS:-"test_real test_roboflow"}
AUTO_STOP=${AUTO_STOP:-0}
PIP_INSTALL=${PIP_INSTALL:-1}
REQUIRE_CUDA=${REQUIRE_CUDA:-1}

stop_pod() {
    if [[ "$AUTO_STOP" == "1" && -n "${RUNPOD_POD_ID:-}" ]]; then
        echo "[run_experiments] stopping pod $RUNPOD_POD_ID"
        runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"
    fi
}
trap stop_pod EXIT

if [[ ! -f src/training/train.py ]]; then
    echo "run from the repo root" >&2
    exit 1
fi

if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
    if [[ ! -f "$DATA_ZIP" ]]; then
        echo "no dataset: put colab_data_v2.zip at $DATA_ZIP or set DATA_DIR" >&2
        exit 1
    fi
    echo "[run_experiments] extracting $DATA_ZIP"
    python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
        "$DATA_ZIP" "$(dirname "$DATA_DIR")" || exit 1
fi
MANIFEST="$DATA_DIR/manifest.json"

if [[ "$PIP_INSTALL" == "1" ]]; then
    python -m pip install -q -r requirements.txt || exit 1
    # opencv-python needs libGL, which slim CUDA images may lack (pods run as root).
    if ! python -c "import cv2" 2>/dev/null; then
        apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null || exit 1
    fi
fi
if [[ "$REQUIRE_CUDA" == "1" ]]; then
    python - <<'EOF' || exit 1
import torch
assert torch.cuda.is_available(), "no CUDA device"
print(f"[run_experiments] torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}")
EOF
fi

mkdir -p "$OUT_DIR"
SHA=$(git rev-parse HEAD)
echo "[run_experiments] commit $SHA, configs: $CONFIGS, seeds: $SEEDS"
FAILED=()

for cfg in $CONFIGS; do
    if [[ ! -f "configs/$cfg.json" ]]; then
        echo "unknown config configs/$cfg.json" >&2
        FAILED+=("$cfg")
        continue
    fi
    for seed in $SEEDS; do
        run="$OUT_DIR/${cfg}_s${seed}"
        done_marker="$run/eval_done"
        if [[ -f "$done_marker" ]]; then
            echo "[run_experiments] $cfg seed $seed already done"
            continue
        fi
        resume=()
        if [[ -f "$run/last.pt" ]]; then
            resume=(--resume-from "$run/last.pt")
            echo "[run_experiments] resuming $cfg seed $seed"
        fi
        echo "===== $cfg seed=$seed ====="
        # shellcheck disable=SC2086
        if ! python -m src.training.train --config "configs/$cfg.json" "${resume[@]}" \
                --set seed="$seed" manifest="$MANIFEST" name="$cfg" checkpoint_dir="$run" \
                num_workers="$NUM_WORKERS" $EXTRA_SET 2>&1 | tee -a "$run.log"; then
            echo "[run_experiments] training failed: $cfg seed $seed" >&2
            FAILED+=("${cfg}_s${seed}")
            continue
        fi
        ok=1
        for split in $SPLITS; do
            python -m src.evaluation.evaluate --checkpoint "$run/best_model.pt" \
                --manifest "$MANIFEST" --split "$split" --out "$run/eval_${split}.json" \
                2>&1 | tee -a "$run.log" || ok=0
        done
        if [[ "$ok" == "1" ]]; then
            echo "$SHA" > "$done_marker"
        else
            FAILED+=("${cfg}_s${seed}_eval")
        fi
    done
done

for split in $SPLITS; do
    if [[ ! -f "$OUT_DIR/nn_baseline_${split}.json" ]]; then
        python scripts/nn_baseline.py --manifest "$MANIFEST" --train-split train \
            --eval-split "$split" --out "$OUT_DIR/nn_baseline_${split}.json" || true
    fi
done
python scripts/summarize_runs.py --runs "$OUT_DIR" --out "$OUT_DIR/summary.md"

if (( ${#FAILED[@]} )); then
    echo "[run_experiments] FAILED: ${FAILED[*]}" >&2
    exit 1
fi
echo "[run_experiments] all done; results in $OUT_DIR/summary.md"
