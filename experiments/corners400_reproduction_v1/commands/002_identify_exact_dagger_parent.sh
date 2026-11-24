#!/usr/bin/env bash

set +e

ROOT=/home/sumukh/AerialCatching
BASE="$ROOT/experiments/corners400_reproduction_v1"

RUN="$ROOT/logs/rsl_rl/CTBR_Hover_Corners_Asymmetric_PPO/2026-07-17_15-08-49_corners_tcn_anchored_one_update"

MODEL29="$RUN/model_29.pt"
MODEL30="$RUN/model_30.pt"
AGENT_YAML="$RUN/params/agent.yaml"
TERMINAL_LOG="$RUN/terminal.log"

REPORT="$BASE/reports/002_exact_dagger_parent.txt"
JSON_REPORT="$BASE/reports/002_exact_dagger_parent.json"

python - \
  "$ROOT" \
  "$MODEL29" \
  "$MODEL30" \
  "$AGENT_YAML" \
  "$TERMINAL_LOG" \
  "$REPORT" \
  "$JSON_REPORT" <<'PY'
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import re
import sys

import torch


root = Path(sys.argv[1])
model29_path = Path(sys.argv[2])
model30_path = Path(sys.argv[3])
agent_yaml_path = Path(sys.argv[4])
terminal_log_path = Path(sys.argv[5])
report_path = Path(sys.argv[6])
json_path = Path(sys.argv[7])


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint is not a dictionary: {path}"
        )

    return checkpoint


def extract_rsl_actor(
    checkpoint: dict,
) -> dict[str, torch.Tensor]:
    state = checkpoint.get(
        "model_state_dict",
    )

    if not isinstance(state, dict):
        raise KeyError(
            "RSL-RL checkpoint lacks model_state_dict."
        )

    actor = {}

    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        normalized = None

        if key.startswith("student."):
            normalized = key[len("student."):]

        elif key.startswith("tcn."):
            normalized = (
                "0."
                + key[len("tcn."):]
            )

        elif key.startswith("actor."):
            normalized = (
                "1."
                + key[len("actor."):]
            )

        if normalized is not None:
            actor[normalized] = tensor.detach().cpu()

    if not actor:
        raise RuntimeError(
            "Could not extract TCN actor tensors."
        )

    return actor


def normalize_offline_student(
    checkpoint: dict,
) -> dict[str, torch.Tensor] | None:
    state = checkpoint.get(
        "student_state_dict",
    )

    if not isinstance(state, dict):
        return None

    normalized = {}

    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        new_key = key

        if new_key.startswith("student."):
            new_key = new_key[len("student."):]

        if new_key.startswith("tcn."):
            new_key = (
                "0."
                + new_key[len("tcn."):]
            )

        elif new_key.startswith("vision_student."):
            new_key = (
                "1."
                + new_key[len("vision_student."):]
            )

        elif new_key.startswith("actor."):
            new_key = (
                "1."
                + new_key[len("actor."):]
            )

        normalized[new_key] = tensor.detach().cpu()

    return normalized or None


def compare_states(
    candidate: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> dict:
    candidate_keys = set(candidate)
    reference_keys = set(reference)

    common = sorted(
        candidate_keys & reference_keys
    )

    missing = sorted(
        reference_keys - candidate_keys
    )

    extra = sorted(
        candidate_keys - reference_keys
    )

    squared_error = 0.0
    maximum_error = 0.0
    exact = (
        not missing
        and not extra
    )

    compared_elements = 0

    for key in common:
        candidate_tensor = candidate[key]
        reference_tensor = reference[key]

        if candidate_tensor.shape != reference_tensor.shape:
            exact = False
            maximum_error = math.inf
            continue

        difference = (
            candidate_tensor.float()
            - reference_tensor.float()
        )

        squared_error += float(
            difference.square().sum().item()
        )

        maximum_error = max(
            maximum_error,
            float(
                difference.abs().max().item()
            ),
        )

        compared_elements += int(
            difference.numel()
        )

        if not torch.equal(
            candidate_tensor,
            reference_tensor,
        ):
            exact = False

    return {
        "exact": exact,
        "common_keys": len(common),
        "missing_keys": missing,
        "extra_keys": extra,
        "compared_elements":
            compared_elements,
        "l2_error":
            math.sqrt(squared_error),
        "maximum_error":
            maximum_error,
    }


model29 = load_checkpoint(
    model29_path
)

model30 = load_checkpoint(
    model30_path
)

actor29 = extract_rsl_actor(
    model29
)

actor30 = extract_rsl_actor(
    model30
)

actor_update = compare_states(
    actor30,
    actor29,
)

candidate_paths = []

for search_root in (
    root / "checkpoints",
    root / "logs",
):
    if not search_root.exists():
        continue

    for path in search_root.rglob("*.pt"):
        if path in {
            model29_path,
            model30_path,
        }:
            continue

        try:
            size = path.stat().st_size
        except OSError:
            continue

        if size > 2 * 1024**3:
            continue

        candidate_paths.append(path)

matches = []
errors = []

for index, path in enumerate(
    sorted(set(candidate_paths)),
    start=1,
):
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            continue

        student = normalize_offline_student(
            checkpoint
        )

        if student is None:
            continue

        comparison = compare_states(
            student,
            actor29,
        )

        matches.append(
            {
                "path": str(
                    path.relative_to(root)
                ),
                "sha256": sha256(path),
                "epoch": checkpoint.get(
                    "global_epoch",
                    checkpoint.get("epoch"),
                ),
                "training_seed":
                    checkpoint.get("training_seed"),
                "data_dir":
                    checkpoint.get("data_dir"),
                **comparison,
            }
        )

    except Exception as exc:
        errors.append(
            {
                "path": str(path),
                "error": repr(exc),
            }
        )

matches.sort(
    key=lambda item: (
        not item["exact"],
        item["l2_error"],
        item["maximum_error"],
        item["path"],
    )
)

exact_matches = [
    item
    for item in matches
    if item["exact"]
]

agent_yaml_text = (
    agent_yaml_path.read_text(
        errors="replace"
    )
    if agent_yaml_path.is_file()
    else ""
)

terminal_text = (
    terminal_log_path.read_text(
        errors="replace"
    )
    if terminal_log_path.is_file()
    else ""
)

important_patterns = (
    "student_checkpoint",
    "finetune_checkpoint",
    "dagger_anchor_coef",
    "anchor_coef",
    "critic_warmup_iterations",
    "ppo_actor_lr",
    "ppo_critic_lr",
    "max_iterations",
    "num_steps_per_env",
    "num_learning_epochs",
    "num_mini_batches",
    "clip_param",
    "entropy_coef",
    "init_noise_std",
    "run_name",
    "experiment_name",
)

config_lines = []

for source_name, text in (
    ("agent.yaml", agent_yaml_text),
    ("terminal.log", terminal_text),
):
    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        if any(
            pattern.lower()
            in line.lower()
            for pattern in important_patterns
        ):
            config_lines.append(
                {
                    "source": source_name,
                    "line": line_number,
                    "text": line.strip(),
                }
            )

path_references = sorted(
    set(
        match.group(0)
        for match in re.finditer(
            r"/home/sumukh/AerialCatching/"
            r"[^\s\"']+\.pt",
            terminal_text,
        )
    )
)

result = {
    "model29": {
        "path": str(model29_path),
        "sha256": sha256(model29_path),
        "iter": model29.get("iter"),
        "actor_tensors": len(actor29),
    },
    "model30": {
        "path": str(model30_path),
        "sha256": sha256(model30_path),
        "iter": model30.get("iter"),
        "actor_tensors": len(actor30),
    },
    "model30_minus_model29_actor":
        actor_update,
    "exact_offline_parent_matches":
        exact_matches,
    "closest_offline_matches":
        matches[:20],
    "config_lines":
        config_lines,
    "checkpoint_path_references":
        path_references,
    "scan_errors": errors,
}

lines = [
    "EXACT HISTORICAL DAGGER PARENT IDENTIFICATION",
    "=" * 116,
    "",
    "MODEL 29",
    "-" * 116,
    f"path: {model29_path}",
    f"sha256: {result['model29']['sha256']}",
    f"iter: {result['model29']['iter']}",
    f"actor tensors: {len(actor29)}",
    "",
    "MODEL 30",
    "-" * 116,
    f"path: {model30_path}",
    f"sha256: {result['model30']['sha256']}",
    f"iter: {result['model30']['iter']}",
    f"actor tensors: {len(actor30)}",
    "",
    "MODEL 30 ACTOR MINUS MODEL 29 ACTOR",
    "-" * 116,
    (
        "exactly identical: "
        f"{actor_update['exact']}"
    ),
    (
        "actor L2 change: "
        f"{actor_update['l2_error']:.10f}"
    ),
    (
        "actor maximum change: "
        f"{actor_update['maximum_error']:.10f}"
    ),
    "",
    "EXACT OFFLINE DAGGER PARENT MATCHES",
    "=" * 116,
]

if exact_matches:
    for item in exact_matches:
        lines.extend(
            [
                f"path: {item['path']}",
                f"sha256: {item['sha256']}",
                f"epoch: {item['epoch']}",
                (
                    "training seed: "
                    f"{item['training_seed']}"
                ),
                f"data directory: {item['data_dir']}",
                "",
            ]
        )
else:
    lines.append(
        "No byte-exact offline student match was found."
    )

lines.extend(
    [
        "CLOSEST OFFLINE STUDENT MATCHES",
        "=" * 116,
    ]
)

for item in matches[:20]:
    lines.append(
        f"exact={item['exact']} "
        f"L2={item['l2_error']:.10f} "
        f"max={item['maximum_error']:.10f} "
        f"epoch={item['epoch']} "
        f"path={item['path']}"
    )

lines.extend(
    [
        "",
        "SAVED CONFIGURATION LINES",
        "=" * 116,
    ]
)

for item in config_lines:
    lines.append(
        f"{item['source']}:{item['line']}: "
        f"{item['text']}"
    )

lines.extend(
    [
        "",
        "CHECKPOINT PATHS RECORDED IN TERMINAL LOG",
        "=" * 116,
    ]
)

if path_references:
    lines.extend(path_references)
else:
    lines.append(
        "No absolute .pt path was recovered from terminal.log."
    )

lineage_ready = bool(
    exact_matches
    or path_references
)

lines.extend(
    [
        "",
        "SUMMARY",
        "=" * 116,
        (
            "offline checkpoints compared: "
            f"{len(matches)}"
        ),
        (
            "exact parent matches: "
            f"{len(exact_matches)}"
        ),
        (
            "LINEAGE_PARENT_IDENTIFIED: "
            f"{bool(exact_matches)}"
        ),
        (
            "LINEAGE_PARENT_EVIDENCE_READY: "
            f"{lineage_ready}"
        ),
    ]
)

text = "\n".join(lines) + "\n"

print(text)
report_path.write_text(text)

json_path.write_text(
    json.dumps(
        result,
        indent=2,
        default=str,
    )
    + "\n"
)
PY

sha256sum \
  "$MODEL29" \
  "$MODEL30" \
  "$REPORT" \
  "$JSON_REPORT" \
  > "$BASE/manifests/002_exact_dagger_parent.sha256"

echo
echo "============================================================"
echo "EXACT DAGGER-PARENT MATCHING FINISHED"
echo "============================================================"

grep -E \
  'actor L2 change:|path: .*\.pt|exact parent matches:|LINEAGE_PARENT_' \
  "$REPORT"
