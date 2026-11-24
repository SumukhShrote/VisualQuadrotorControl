#!/usr/bin/env bash

set +e

ROOT=/home/sumukh/AerialCatching
BASE="$ROOT/experiments/corners400_reproduction_v1"

RUN="$ROOT/logs/rsl_rl/CTBR_Hover_Corners_Asymmetric_PPO/2026-07-17_15-08-49_corners_tcn_anchored_one_update"

REPORT="$BASE/reports/001_successful_lineage_reconstruction.txt"

mkdir -p \
  "$BASE/reports" \
  "$BASE/manifests" \
  "$BASE/docs"

{
  echo "SUCCESSFUL CORNERS-TCN LINEAGE RECONSTRUCTION"
  echo "============================================================================================================"
  echo
  echo "RULE: historical checkpoints are read-only references and must never initialize the reproduction."
  echo

  echo "===== AUTHORITATIVE HISTORICAL RESULT ====="
  if [ -f "$ROOT/experiments/corners256_v2/reports/180_final_head_to_head.txt" ]; then
    cat "$ROOT/experiments/corners256_v2/reports/180_final_head_to_head.txt"
  else
    echo "[WARN] Missing final head-to-head report."
  fi

  echo
  echo "===== HISTORICAL RUN CONTENTS ====="
  if [ -d "$RUN" ]; then
    find "$RUN" \
      -maxdepth 3 \
      -type f \
      -printf '%P\n' \
      | sort
  else
    echo "[WARN] Missing historical run directory: $RUN"
  fi

  echo
  echo "===== RUN COMMAND AND PARENT REFERENCES ====="
  grep -R -n -A15 -B15 \
    -E \
    'python|corners_ppo|train_dagger|resume|checkpoint|load_run|load_checkpoint|student_checkpoint|anchor|anchored|model_28|model_29|model_30|teacher_checkpoint|max_iterations|learning_rate' \
    "$RUN" \
    "$ROOT/experiments/corners256_v2" \
    "$ROOT/experiments/corners256_sequential_v2" \
    "$ROOT/eval_logs/corners_tcn_anchored_one_update" \
    "$ROOT/eval_logs/corners_tcn_anchored_one_update_multiseed" \
    2>/dev/null \
    | head -n 2500

  echo
  echo "===== RELEVANT EXPERIMENT COMMAND FILES ====="
  find \
    "$ROOT/experiments/corners256_v2" \
    "$ROOT/experiments/corners256_sequential_v2" \
    -type f \
    \( -name '*.sh' -o -name '*.py' -o -name '*.md' -o -name '*.txt' \) \
    -print 2>/dev/null \
    | sort \
    | while IFS= read -r path; do
        if grep -Eq \
          'corners_tcn_anchored_one_update|corners_ppo|model_29|model_30|anchor|historical_final|763/2000|393\.07' \
          "$path"
        then
          echo "$path"
        fi
      done

  echo
  echo "===== SAVED HYDRA / RSL-RL CONFIGURATION ====="
  for path in \
    "$RUN/params/env.yaml" \
    "$RUN/params/agent.yaml" \
    "$RUN/.hydra/config.yaml" \
    "$RUN/.hydra/overrides.yaml" \
    "$RUN/terminal.log"
  do
    if [ -f "$path" ]; then
      echo
      echo "----- $path -----"
      cat "$path"
    else
      echo "[WARN] Missing: $path"
    fi
  done

  echo
  echo "===== HISTORICAL CHECKPOINT METADATA ====="
  python - \
    "$RUN/model_29.pt" \
    "$RUN/model_30.pt" <<'PY'
from pathlib import Path
import hashlib
import sys

import torch


for value in sys.argv[1:]:
    path = Path(value)

    print()
    print("=" * 108)
    print(f"path: {path}")

    if not path.is_file():
        print("status: MISSING")
        continue

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "sha256:",
        hashlib.sha256(
            path.read_bytes()
        ).hexdigest(),
    )

    print("checkpoint_type:", type(checkpoint).__name__)

    if not isinstance(checkpoint, dict):
        continue

    print("keys:", sorted(checkpoint))

    for key in sorted(checkpoint):
        value = checkpoint[key]

        lowered = key.lower()

        if not any(
            term in lowered
            for term in (
                "epoch",
                "iteration",
                "checkpoint",
                "resume",
                "load",
                "run",
                "experiment",
                "anchor",
                "learning",
                "seed",
                "optimizer",
                "model",
                "student",
            )
        ):
            continue

        if isinstance(value, dict):
            print(
                f"{key}: dict[{len(value)}]"
            )

            for nested_key in sorted(value):
                nested_value = value[nested_key]

                if isinstance(
                    nested_value,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    ),
                ):
                    print(
                        f"  {nested_key}: "
                        f"{nested_value}"
                    )

        elif isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                type(None),
            ),
        ):
            print(f"{key}: {value}")

        elif isinstance(value, torch.Tensor):
            print(
                f"{key}: tensor "
                f"shape={tuple(value.shape)}"
            )
PY

  echo
  echo "===== SOURCE CODE AROUND ANCHORED PPO ====="
  for path in \
    "$ROOT/rl/corners_ppo.py" \
    "$ROOT/envs/aruco_hover/agents/rsl_rl_distillation_cfg.py" \
    "$ROOT/envs/aruco_hover/agents/vision_student_teacher.py"
  do
    if [ -f "$path" ]; then
      echo
      echo "----- $path -----"

      grep -n -A35 -B20 \
        -E \
        'anchor|reference|initial|resume|checkpoint|student|actor|CornersTCN|learning_rate|optimizer|loss' \
        "$path" \
        | head -n 700
    else
      echo "[WARN] Missing: $path"
    fi
  done

  echo
  echo "===== GIT STATE AROUND SUCCESSFUL RUN ====="
  git -C "$ROOT" log \
    --all \
    --since='2026-07-16 00:00:00' \
    --until='2026-07-18 23:59:59' \
    --date=iso \
    --pretty=format:'%H | %ad | %s'

  echo
  echo
  echo "===== RELEVANT HISTORICAL SOURCE VERSIONS ====="
  for commit in \
    b4e22d9dced34541be5479e065379a361f9b5c42 \
    76e5a8debc8be7d0fd5e545941614a5b641aef9b \
    6400d81159f948f5af1643023a137c12b1695552 \
    81aaf975093aa84c96b9a90a858265bb8d790981
  do
    echo
    echo "----- $commit -----"

    git -C "$ROOT" show \
      --stat \
      --oneline \
      "$commit" \
      2>/dev/null
  done

  echo
  echo "SUCCESSFUL_LINEAGE_RECONSTRUCTION_READY: True"
} | tee "$REPORT"

sha256sum \
  "$REPORT" \
  > "$BASE/manifests/001_successful_lineage_reconstruction.sha256"

cat > "$BASE/docs/CURRENT_STATUS.md" <<EOF
# Corners-400 Reproduction Status

## Objective

Reproduce the historical corners-only TCN result from its earliest reproducible
training root.

## Historical benchmark

- Success: 763/2000
- Success rate: 38.15%
- Mean episode length: 393.07
- Mean return: 27.6692

## Restrictions

Historical model files are read-only references.

They may not be used as:

- initialization weights
- resume checkpoints
- optimizer state
- PPO anchors
- copied final policies

## Current phase

Recovering the exact training dependency chain.

No reproduction training has started.
EOF

echo
echo "============================================================"
echo "LINEAGE RECONSTRUCTION FINISHED"
echo "============================================================"
echo "report=$REPORT"
