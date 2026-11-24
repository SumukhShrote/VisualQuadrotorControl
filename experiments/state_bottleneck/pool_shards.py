from pathlib import Path
import torch

ROOT = Path("experiments/state_bottleneck")
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "pooled_student_epoch70_all_levels.pt"

levels = [0.00, 0.25, 0.50, 0.75, 1.00]
required = {
    "teacher_observations",
    "teacher_actions",
    "environment_ids",
    "episode_ids",
    "episode_steps",
}

shards = []
sample_keys = None

for round_idx, level in enumerate(levels):
    path = DATA_DIR / f"round_{round_idx:02d}_shard_0000.pt"

    if not path.is_file():
        raise RuntimeError(f"Missing shard: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a dictionary")

    if not required.issubset(data):
        raise KeyError(
            f"{path}: missing {sorted(required - set(data))}"
        )

    n = len(data["teacher_actions"])

    if n != 20000:
        raise ValueError(f"{path}: expected 20000, found {n}")

    keys = {
        k for k, v in data.items()
        if isinstance(v, torch.Tensor)
        and v.ndim >= 1
        and len(v) == n
    }

    if sample_keys is None:
        sample_keys = keys
        print("[DATASET KEYS]", sorted(sample_keys))
    elif keys != sample_keys:
        raise ValueError(
            f"Shard schema differs: {path}\n"
            f"Missing: {sorted(sample_keys - keys)}\n"
            f"Extra: {sorted(keys - sample_keys)}"
        )

    data["environment_ids"] = (
        data["environment_ids"].long() + round_idx * 10000
    )

    if "curriculum_levels" in keys:
        recorded = data["curriculum_levels"].float()
        if not torch.allclose(
            recorded,
            torch.full_like(recorded, level),
            atol=1e-4,
        ):
            raise ValueError(
                f"Unexpected curriculum labels: {path}"
            )

    shards.append({k: data[k] for k in sample_keys})

    print(
        f"[PASS] round={round_idx} "
        f"level={level:.2f} "
        f"samples={n}"
    )

pooled = {
    k: torch.cat([shard[k] for shard in shards], dim=0)
    for k in sorted(sample_keys)
}

if "curriculum_levels" not in pooled:
    pooled["curriculum_levels"] = torch.cat([
        torch.full((20000,), level)
        for level in levels
    ])

corners = [
    (k, v)
    for k, v in pooled.items()
    if isinstance(v, torch.Tensor)
    and v.ndim == 2
    and v.shape == (100000, 256)
]

if not corners:
    raise RuntimeError("No 256-D corner-history tensor found")

if pooled["teacher_observations"].shape != (100000, 20):
    raise ValueError("Unexpected teacher observation shape")

if pooled["teacher_actions"].shape != (100000, 4):
    raise ValueError("Unexpected teacher action shape")

if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists; preserving it: {OUTPUT}"
    )

torch.save(pooled, OUTPUT)

print("\n========== POOLED DATASET ==========")
print("Total samples:", len(pooled["teacher_actions"]))
print("Corner tensors:", [(k, tuple(v.shape)) for k, v in corners])
print("Teacher observations:", pooled["teacher_observations"].shape)
print("Teacher actions:", pooled["teacher_actions"].shape)
print("Unique environment/episode pairs:", len(torch.unique(
    torch.stack([
        pooled["environment_ids"].long(),
        pooled["episode_ids"].long()
    ], dim=1),
    dim=0
)))
print("[SAVED]", OUTPUT)
