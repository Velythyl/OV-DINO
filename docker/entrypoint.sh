#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--" ]]; then
    shift
fi

# Allow the image to be used for training, evaluation, or debugging as well.
if [[ $# -gt 0 ]]; then
    exec "$@"
fi

: "${MODEL_ROOT:=/weights}"
: "${MODEL_CHECKPOINT:=}"

if [[ -z "$MODEL_CHECKPOINT" ]]; then
    mapfile -t checkpoints < <(find "$MODEL_ROOT" -maxdepth 3 -type f -name '*.pth' | sort)
    if [[ ${#checkpoints[@]} -eq 1 ]]; then
        MODEL_CHECKPOINT="${checkpoints[0]}"
    else
        echo "Set MODEL_CHECKPOINT to the mounted OV-DINO .pth file." >&2
        echo "Found ${#checkpoints[@]} checkpoint candidates under $MODEL_ROOT." >&2
        exit 2
    fi
fi

if [[ ! -f "$MODEL_CHECKPOINT" ]]; then
    echo "MODEL_CHECKPOINT does not exist: $MODEL_CHECKPOINT" >&2
    exit 2
fi

export MODEL_ROOT
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-/workspace/datas}"

exec python3 demo/app.py \
    --config-file projects/ovdino/configs/ovdino_swin_tiny224_bert_base_infer_demo.py \
    --opts train.init_checkpoint="$MODEL_CHECKPOINT" model.app_mode=True
