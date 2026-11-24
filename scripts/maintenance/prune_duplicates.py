"""Remove verified redundant files from temporary historical directories."""

from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import csv
import os
import stat

ROOT = Path(__file__).resolve().parents[2]

ARCHIVE = ROOT.parent / "VisualQuadrotorLearning_archive"

RECOVERY = (
    ROOT.parent
    / "VisualQuadrotorLearning_recovery_20261124T193142Z"
)

CLEANUP = ROOT / "docs/cleanup"

STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

HISTORICAL = [ARCHIVE, RECOVERY]

SKIP = {
    ".git",
    ".cache",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}

if ROOT != Path("/home/sumukh/AerialCatching"):
    raise RuntimeError(f"Unexpected repository: {ROOT}")

for directory in HISTORICAL:
    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError(f"Unexpected historical directory: {directory}")

# ------------------------------------------------------------
# File utilities
# ------------------------------------------------------------

def files_under(directory):

    for current, dirs, names in os.walk(
        directory, followlinks=False
    ):

        current = Path(current)

        dirs[:] = [
            d for d in dirs
            if d not in SKIP
            and not (current / d).is_symlink()
        ]

        for name in names:

            path = current / name

            if path.is_symlink():
                continue

            if stat.S_ISREG(path.lstat().st_mode):
                yield path


def sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            digest.update(chunk)

    return digest.hexdigest()


def signature(path):

    s = path.stat()

    return (
        s.st_dev,
        s.st_ino,
        s.st_size,
        s.st_mtime_ns,
        s.st_ctime_ns,
    )


hash_cache = {}


def cached_hash(path):

    key = signature(path)

    if key not in hash_cache:
        hash_cache[key] = sha256(path)

    return hash_cache[key]


# ------------------------------------------------------------
# Verify reference checkpoints and essential dependencies
# ------------------------------------------------------------

manifest = json.loads(
    (ROOT / "checkpoints/reference/manifest.json").read_text()
)

def verify_references():

    for item in manifest:

        path = ROOT / item["canonical_path"]

        if not path.is_file():
            raise FileNotFoundError(path)

        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Reference hash mismatch: {path}")


verify_references()

print("[PASS] All reference artifacts verified:", len(manifest))

required = [
    "experiments/state_bottleneck/data/"
    "pooled_student_epoch70_all_levels.pt",

    "experiments/state_bottleneck/"
    "state_reconstruction_20261119T195844Z/protocol.json",

    "experiments/state_bottleneck/"
    "supervised_multiscale_transformer.py",

    "experiments/state_bottleneck/"
    "train_two_gpu_visual_policies.py",

    "experiments/state_bottleneck/"
    "supervised_observability_probe.py",
]

dependencies = []

for rel in required:

    path = ROOT / rel

    if not path.is_file():
        raise FileNotFoundError(path)

    dependencies.append({
        "path": rel,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    })

    print("[RETAIN]", rel)

(CLEANUP / "training_dependencies.json").write_text(
    json.dumps(dependencies, indent=2) + "\n"
)

# ------------------------------------------------------------
# Check for active training/evaluation processes
# ------------------------------------------------------------

for proc in Path("/proc").glob("[0-9]*"):

    if int(proc.name) == os.getpid():
        continue

    try:

        cwd = (proc / "cwd").resolve(strict=True)

        args = (
            (proc / "cmdline").read_bytes()
            .decode(errors="replace")
            .split("\0")
        )

    except OSError:
        continue

    executable = Path(args[0]).name.lower() if args else ""

    if (
        cwd == ROOT or ROOT in cwd.parents
    ) and any(
        word in executable
        for word in ("python", "isaac", "ffmpeg")
    ):
        raise RuntimeError(
            f"Repository process still running: "
            f"{proc.name} {args[:5]}"
        )

print("[PASS] No detected repository compute processes")

# ------------------------------------------------------------
# Protect symbolic-link targets
# ------------------------------------------------------------

protected_targets = set()

for directory in [ROOT] + HISTORICAL:

    for current, dirs, names in os.walk(
        directory, followlinks=False
    ):

        current = Path(current)

        dirs[:] = [d for d in dirs if d not in SKIP]

        for name in dirs + names:

            path = current / name

            if path.is_symlink():
                protected_targets.add(path.resolve())

# ------------------------------------------------------------
# Index retained files
# ------------------------------------------------------------

active = defaultdict(list)

for path in files_under(ROOT):
    active[path.stat().st_size].append(path)

print(
    "[PASS] Active files indexed:",
    sum(len(v) for v in active.values()),
)

# ------------------------------------------------------------
# Permanent duplicate removal
# ------------------------------------------------------------

journal = CLEANUP / f"duplicate_deletions_{STAMP}.jsonl"

deleted = 0
deleted_bytes = 0


def log(event):

    with journal.open("a") as f:

        f.write(json.dumps({
            "timestamp_utc":
                datetime.now(timezone.utc).isoformat(),
            **event,
        }) + "\n")

        f.flush()
        os.fsync(f.fileno())


for directory in HISTORICAL:

    print("\n[PROCESSING]", directory.name)

    for path in files_under(directory):

        if path.name == "archive_manifest.jsonl":
            continue

        resolved = path.resolve()

        if any(
            resolved == target or target in resolved.parents
            for target in protected_targets
        ):
            continue

        candidates = active.get(path.stat().st_size, [])

        if not candidates:
            continue

        source_sig = signature(path)
        source_hash = cached_hash(path)

        retained = None

        for candidate in candidates:

            if (
                candidate.is_file()
                and not candidate.is_symlink()
                and cached_hash(candidate) == source_hash
            ):

                retained = candidate
                break

        if retained is None:
            continue

        if directory not in path.resolve().parents:
            raise RuntimeError(f"Unsafe deletion path: {path}")

        if ROOT not in retained.resolve().parents:
            raise RuntimeError(f"Unsafe retained path: {retained}")

        if path.is_symlink() or retained.is_symlink():
            raise RuntimeError("Unexpected symbolic link")

        retained_sig = signature(retained)

        # Recheck actual content before irreversible deletion.
        if sha256(path) != sha256(retained):
            raise RuntimeError(f"Duplicate check failed: {path}")

        if (
            source_sig != signature(path)
            or retained_sig != signature(retained)
        ):
            raise RuntimeError("File changed during verification")

        event = {
            "operation": "delete_verified_duplicate",
            "deleted_path": str(path),
            "retained_path": str(retained.relative_to(ROOT)),
            "sha256": source_hash,
            "bytes": source_sig[2],
        }

        log({**event, "status": "planned"})

        path.unlink()

        log({**event, "status": "completed"})

        deleted += 1
        deleted_bytes += source_sig[2]

        if deleted % 500 == 0:

            print(
                "[PROGRESS]",
                deleted,
                "files deleted;",
                f"{deleted_bytes / 1024**3:.3f} GiB",
            )

# ------------------------------------------------------------
# Remove empty directories
# ------------------------------------------------------------

for directory in HISTORICAL:

    for current, dirs, names in os.walk(
        directory, topdown=False, followlinks=False
    ):

        path = Path(current)

        if path == directory or path.is_symlink():
            continue

        if not any(path.iterdir()):
            path.rmdir()

# ------------------------------------------------------------
# Inventory remaining unique historical artifacts
# ------------------------------------------------------------

groups = defaultdict(lambda: {
    "files": 0,
    "bytes": 0,
    "metadata": 0,
    "binaries": 0,
})

metadata_extensions = {
    ".json", ".jsonl", ".csv", ".yaml", ".yml",
    ".md", ".txt", ".log", ".toml",
}

binary_extensions = {
    ".pt", ".pth", ".ts", ".ckpt", ".npz",
    ".npy", ".pkl", ".h5", ".safetensors",
}

report = CLEANUP / "remaining_historical_artifacts.csv"

with report.open("w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "location",
        "relative_path",
        "bytes",
        "extension",
    ])

    for directory in HISTORICAL:

        for path in files_under(directory):

            rel = path.relative_to(directory)
            size = path.stat().st_size

            parts = rel.parts

            if parts[0] == "files":
                parts = parts[1:]

            if not parts:
                continue

            depth = (
                4 if parts[0] == "logs"
                and len(parts) > 3
                and parts[1] == "rsl_rl"
                else 3
            )

            group = str(Path(*parts[:depth]))

            info = groups[(directory.name, group)]

            info["files"] += 1
            info["bytes"] += size

            if path.suffix.lower() in metadata_extensions:
                info["metadata"] += 1

            if path.suffix.lower() in binary_extensions:
                info["binaries"] += 1

            writer.writerow([
                directory.name,
                str(rel),
                size,
                path.suffix,
            ])

verify_references()

summary = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "operation": "permanent_duplicate_cleanup",
    "deleted_files": deleted,
    "deleted_bytes": deleted_bytes,
    "reference_artifacts_verified": len(manifest),
    "unique_artifacts_deleted": 0,
}

with (CLEANUP / "operations.jsonl").open("a") as f:
    f.write(json.dumps(summary) + "\n")

print("\n" + "=" * 80)
print("PERMANENT DUPLICATE CLEANUP COMPLETE")
print("=" * 80)

print("Files permanently deleted:", deleted)

print(
    "Logical duplicate storage removed:",
    f"{deleted_bytes / 1024**3:.3f} GiB",
)

print("\nLargest remaining historical groups:")

for (location, group), info in sorted(
    groups.items(),
    key=lambda x: x[1]["bytes"],
    reverse=True,
)[:25]:

    print(
        f"{info['bytes'] / 1024**3:8.3f} GiB",
        f"files={info['files']:5}",
        f"metadata={info['metadata']:4}",
        f"binary={info['binaries']:4}",
        location,
        group,
    )

print("\nDeletion journal:", journal)
print("Remaining inventory:", report)

print("[PASS] Reference artifacts unchanged")
print("[PASS] No active repository files deleted")
print("[PASS] No unique research files deleted")
print("[PASS] No new backups or archives created")
