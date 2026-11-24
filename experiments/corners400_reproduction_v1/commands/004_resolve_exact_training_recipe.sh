#!/usr/bin/env bash

set +e

ROOT="/home/sumukh/AerialCatching"
ARCHIVE="$ROOT/dagger_data/corners_stage2_final_dataset.tar.gz"
REPORT="$ROOT/experiments/corners400_reproduction_v1/reports/004_exact_training_recipe_inputs.txt"
HISTORICAL_COMMIT="b4e22d9dced34541be5479e065379a361f9b5c42"

cd "$ROOT" || {
    echo "[ERROR] Could not enter $ROOT"
}

{
    echo "EXACT HISTORICAL TRAINING RECIPE INPUTS"
    echo "================================================================================"

    echo
    echo "===== ARCHIVED SHARD CONTENTS ====="

    python - "$ARCHIVE" <<'PY'
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import gc
import io
import re
import sys
import tarfile

import torch


archive_path = Path(sys.argv[1])
actual_optimizer_steps = 215_465
batch_sizes = (128, 256, 512, 1024)

if not archive_path.is_file():
    print("[ERROR] Missing archive:", archive_path)
    raise SystemExit(0)


def summarize_value(value):
    if torch.is_tensor(value):
        if value.numel() <= 16:
            return value.detach().cpu().tolist()

        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, dict):
        result = {}

        for key, child in value.items():
            result[str(key)] = summarize_value(child)

        return result

    if isinstance(value, (list, tuple)):
        if len(value) <= 16:
            return [
                summarize_value(child)
                for child in value
            ]

        return {
            "type": type(value).__name__,
            "length": len(value),
        }

    if isinstance(
        value,
        (str, int, float, bool, type(None)),
    ):
        return value

    return repr(value)


records = []

with tarfile.open(archive_path, "r:gz") as archive:
    for member in archive:
        if not member.isfile():
            continue

        match = re.search(
            r"round_(\d+)_shard_(\d+)\.pt$",
            member.name,
        )

        if match is None:
            continue

        extracted = archive.extractfile(member)

        if extracted is None:
            print("[WARN] Could not read:", member.name)
            continue

        payload = extracted.read()

        try:
            shard = torch.load(
                io.BytesIO(payload),
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            print(
                "[ERROR] Failed to load",
                member.name,
                repr(exc),
            )
            continue

        observations = shard.get("observations")
        teacher_actions = shard.get("teacher_actions")

        observation_rows = (
            int(observations.shape[0])
            if torch.is_tensor(observations)
            else None
        )

        observation_dim = (
            int(observations.shape[-1])
            if torch.is_tensor(observations)
            and observations.ndim >= 2
            else None
        )

        action_rows = (
            int(teacher_actions.shape[0])
            if torch.is_tensor(teacher_actions)
            else None
        )

        action_dim = (
            int(teacher_actions.shape[-1])
            if torch.is_tensor(teacher_actions)
            and teacher_actions.ndim >= 2
            else None
        )

        metadata = {
            str(key): summarize_value(value)
            for key, value in shard.items()
            if key not in {
                "observations",
                "teacher_actions",
            }
        }

        records.append(
            {
                "round": int(match.group(1)),
                "shard": int(match.group(2)),
                "name": member.name,
                "tar_size": int(member.size),
                "tar_mtime": datetime.fromtimestamp(
                    member.mtime
                ).isoformat(),
                "rows": observation_rows,
                "observation_dim": observation_dim,
                "action_rows": action_rows,
                "action_dim": action_dim,
                "metadata": metadata,
            }
        )

        del shard
        del observations
        del teacher_actions
        del payload
        gc.collect()


records.sort(
    key=lambda record: (
        record["round"],
        record["shard"],
    )
)

print("archive:", archive_path)
print("shards:", len(records))
print()

for record in records:
    rows = record["rows"]

    batch_counts = {
        batch_size: (
            rows // batch_size
            if rows is not None
            else None
        )
        for batch_size in batch_sizes
    }

    print(
        f"round={record['round']:02d} "
        f"shard={record['shard']:04d} "
        f"rows={rows} "
        f"obs_dim={record['observation_dim']} "
        f"action_dim={record['action_dim']} "
        f"tar_bytes={record['tar_size']} "
        f"mtime={record['tar_mtime']} "
        f"batches={batch_counts}"
    )

    if record["metadata"]:
        print("    metadata:", record["metadata"])


groups = {
    "rounds_00_04": range(0, 5),
    "rounds_00_09": range(0, 10),
    "rounds_00_14": range(0, 15),
    "rounds_20_24": range(20, 25),
    "rounds_20_29": range(20, 30),
    "base_plus_20_24": (
        list(range(0, 15))
        + list(range(20, 25))
    ),
    "base_plus_20_29": (
        list(range(0, 15))
        + list(range(20, 30))
    ),
}

print()
print("DATASET-PASS UPDATE COUNTS")
print("=" * 80)

record_by_round = defaultdict(list)

for record in records:
    record_by_round[record["round"]].append(record)

for group_name, rounds in groups.items():
    selected = [
        record
        for round_index in rounds
        for record in record_by_round.get(
            round_index,
            [],
        )
    ]

    print()
    print(
        group_name,
        "rounds=",
        [record["round"] for record in selected],
        "rows=",
        sum(
            record["rows"] or 0
            for record in selected
        ),
    )

    for batch_size in batch_sizes:
        updates = sum(
            (record["rows"] or 0) // batch_size
            for record in selected
        )

        print(
            f"    batch_size={batch_size}: "
            f"updates_per_epoch={updates}"
        )


print()
print("OPTIMIZER-STEP TARGET")
print("=" * 80)
print("historical optimizer steps:", actual_optimizer_steps)

for batch_size in batch_sizes:
    all_updates = sum(
        (record["rows"] or 0) // batch_size
        for record in records
    )

    print(
        f"batch_size={batch_size}: "
        f"all_25_shards_updates_per_epoch={all_updates}, "
        f"equivalent_full_epochs="
        f"{actual_optimizer_steps / max(all_updates, 1):.6f}"
    )
PY

    echo
    echo "===== HISTORICAL TRAINER: ARGUMENTS AND LR SCHEDULE ====="

    if git cat-file -e \
        "${HISTORICAL_COMMIT}:rl/train_dagger.py" \
        2>/dev/null
    then
        git show \
            "${HISTORICAL_COMMIT}:rl/train_dagger.py" \
            | nl -ba \
            | sed -n '1,180p'
    else
        echo "[ERROR] Historical trainer unavailable at $HISTORICAL_COMMIT"
    fi

    echo
    echo "===== HISTORICAL TRAINER: SAVE AND RESUME LOGIC ====="

    if git cat-file -e \
        "${HISTORICAL_COMMIT}:rl/train_dagger.py" \
        2>/dev/null
    then
        git show \
            "${HISTORICAL_COMMIT}:rl/train_dagger.py" \
            | nl -ba \
            | sed -n '250,380p'
    fi

    echo
    echo "===== BASH HISTORY AROUND ORIGINAL TRAIN_DAGGER COMMAND ====="

    history -a 2>/dev/null || true

    if [[ -s "$HOME/.bash_history" ]]; then
        nl -ba "$HOME/.bash_history" \
            | sed -n '1180,1305p'
    else
        echo "[MISSING] $HOME/.bash_history"
    fi

    echo
    echo "===== EXACT HISTORICAL STAGE-NAME HITS ====="

    grep -n -C 25 -E \
        'corners_paper_stage2_pilot|corners_stage2_best_bc|corners_stage2_best_dagger1m|corners_stage2_best_dagger1p5m|corners_stage2_dagger2m_conservative|corners_stage2_beta50_refine|corners_stage2_final_dagger|python -m rl\.train_dagger' \
        "$HOME/.bash_history" \
        2>/dev/null

    echo
    echo "EXACT_RECIPE_INPUTS_READY: True"
} | tee "$REPORT"

echo
echo "============================================================"
echo "REPORT CREATED"
echo "============================================================"
echo "$REPORT"
