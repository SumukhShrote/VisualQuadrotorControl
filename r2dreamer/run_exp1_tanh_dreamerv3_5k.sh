#!/usr/bin/env bash
set -euo pipefail

cd /home/sumukh/AerialCatching/r2dreamer

unset CUDA_VISIBLE_DEVICES

export PYTHONPATH="/home/sumukh/AerialCatching:/home/sumukh/AerialCatching/r2dreamer${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

if ! grep -q "class TanhNormal" distributions.py; then
    echo "[ERROR] TanhNormal class is missing."
    exit 1
fi

if ! grep -A 15 "def bounded_normal" distributions.py \
    | grep -q "return TanhNormal"; then
    echo "[ERROR] bounded_normal does not return TanhNormal."
    exit 1
fi

python -m py_compile distributions.py networks.py dreamer.py

GPU_INDEX="$(
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits \
    | awk -F',' '
        {
            gsub(/ /, "", $1)
            gsub(/ /, "", $2)
            gsub(/ /, "", $3)
            score = $2 + 100 * $3
            print score, $1
        }
      ' \
    | sort -n \
    | head -n 1 \
    | awk '{print $2}'
)"

if [[ -z "$GPU_INDEX" ]]; then
    echo "[ERROR] Could not choose a GPU."
    exit 1
fi

DEVICE="cuda:${GPU_INDEX}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RUN_DIR="/home/sumukh/AerialCatching/logs/r2dreamer/exp1_tanh_dreamerv3_5k_${TIMESTAMP}"

mkdir -p "$RUN_DIR"

printf '%s\n' "$RUN_DIR" \
  > /home/sumukh/AerialCatching/logs/r2dreamer/latest_exp1_tanh_run.txt

cp distributions.py "$RUN_DIR/distributions.py"
cp dreamer.py "$RUN_DIR/dreamer.py"
cp networks.py "$RUN_DIR/networks.py"
cp "$0" "$RUN_DIR/run_script.sh"

echo "============================================================"
echo "EXPERIMENT 1: TANH-SQUASHED DREAMERV3 ACTOR"
echo "GPU:       ${GPU_INDEX}"
echo "Device:    ${DEVICE}"
echo "Run dir:   ${RUN_DIR}"
echo "Changes:   action distribution + metrics only"
echo "============================================================"

python -u train_checkpointed.py \
  env=isaaclab_quadrotor_rgb \
  model=size12M \
  model.rep_loss=dreamer \
  model.compile=false \
  seed=42 \
  device="$DEVICE" \
  env.device="$DEVICE" \
  env.env_num=4 \
  env.eval_episode_num=0 \
  env.steps=5000 \
  env.train_ratio=512 \
  trainer.update_log_every=250 \
  trainer.video_pred_log=false \
  +trainer.checkpoint_every=500 \
  buffer.device="$DEVICE" \
  buffer.storage_device="$DEVICE" \
  logdir="$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/terminal.log"

TRAIN_RC=${PIPESTATUS[0]}

echo
echo "Training return code: $TRAIN_RC"
echo "Run directory: $RUN_DIR"

find "$RUN_DIR" -maxdepth 1 -type f \
  \( -name 'step_*.pt' -o -name 'latest.pt' \) \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS  %12s bytes  %f\n' \
  | sort

exit "$TRAIN_RC"
