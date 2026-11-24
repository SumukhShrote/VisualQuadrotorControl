#!/usr/bin/env python3

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path.cwd()
BASE = ROOT / "experiments/state_bottleneck"

STUDY = BASE / "action_history_study_20261120T200813Z"

COLLECTOR = BASE / "collect_transformer_teacher_probe.py"
TRAINER = BASE / "launch_action_history_study.py"
EVAL = STUDY / "evaluate_action_history.py"

ROUND1 = (
    BASE / "k8_onpolicy_dagger_round1_20261120T214738Z"
)

BASE_CKPT = ROUND1 / "finetuned_k8/policy.pt"
BASE_TS = ROUND1 / "finetuned_k8/policy.ts"

TEACHER = (
    ROOT / "logs/rsl_rl/CTBR_Hover_Teacher/"
    "2026-06-25_12-41-52/model_749.pt"
)

PYTHON = sys.executable
SEED = 4400
N_PER_LEVEL = 50000


def check_files():
    for p in [
        COLLECTOR, TRAINER, EVAL,
        BASE_CKPT, BASE_TS, TEACHER
    ]:
        assert p.is_file(), f"Missing required file: {p}"

    expected = (
        "1e47ed23f54c83acfc060ef8daa69"
        "f00f28a9db29b5284cc99c5462899e56bd6"
    )

    assert hashlib.sha256(
        TEACHER.read_bytes()
    ).hexdigest() == expected, "Wrong privileged PPO checkpoint"

    torch.set_num_threads(4)

    policy = torch.jit.load(
        str(BASE_TS),
        map_location="cpu"
    ).eval()

    with torch.inference_mode():
        out = policy(
            torch.zeros(2, 256),
            torch.zeros(2, 8, 4)
        )

    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()

    print(
        "[PASS] PPO teacher, K8 checkpoint, "
        "and TorchScript interface",
        flush=True
    )


def replace_exact(source, old, new):
    n = source.count(old)

    if n != 1:
        raise RuntimeError(
            f"Collector interface changed: "
            f"expected one occurrence, found {n}: "
            f"{old[:100]!r}"
        )

    return source.replace(old, new, 1)


def make_collector(path):
    code = COLLECTOR.read_text()

    code = replace_exact(
        code,
        "    transformer_device = None\n",
        '''    transformer_device = None

    action_history = torch.zeros(
        args_cli.num_envs,
        8,
        4,
        device=agent_cfg.device,
        dtype=torch.float32,
    )

    previous_episode_steps = None
'''
    )

    code = replace_exact(
        code,
        '''            policy_obs = obs_dict["policy"]
            teacher_obs = obs_dict["teacher"]''',
        '''            policy_obs = obs_dict["policy"]
            teacher_obs = obs_dict["teacher"]

            # The buffer contains ONLY previously executed commands.
            # The privileged observation is used exclusively
            # to verify alignment, never as an actor input.

            current_episode_steps = (
                env.unwrapped.episode_length_buf
                .detach()
                .clone()
            )

            if previous_episode_steps is not None:
                reset_mask = (
                    current_episode_steps
                    <= previous_episode_steps
                )

                if bool(reset_mask.any()):
                    action_history[reset_mask] = 0.0

            previous_episode_steps = current_episode_steps

            if action_history.device != policy_obs.device:
                action_history = action_history.to(
                    policy_obs.device
                )

            if not torch.allclose(
                action_history[:, -1, :],
                teacher_obs[:, 15:19],
                atol=1e-5,
                rtol=0.0,
            ):
                err = (
                    action_history[:, -1, :]
                    - teacher_obs[:, 15:19]
                ).abs().max().item()

                raise RuntimeError(
                    f"Action-history mismatch at collection "
                    f"step {controller_step}: "
                    f"maximum difference={err}"
                )'''
    )

    code = replace_exact(
        code,
        "transformer_policy(policy_obs)",
        "transformer_policy(policy_obs, action_history)"
    )

    code = replace_exact(
        code,
        '''            executed_action = torch.where(
                teacher_mask,
                teacher_action,
                student_action,
            )''',
        '''            executed_action = torch.where(
                teacher_mask,
                teacher_action,
                student_action,
            )

            # Advance only AFTER computing the current action.
            # Reset detection happens before the next inference.

            action_history = torch.roll(
                action_history,
                shifts=-1,
                dims=1
            )

            action_history[:, -1, :] = (
                executed_action.detach()
            )'''
    )

    assert "env_cfg.eval_mode = True" in code
    assert "not args_cli.torchscript_policy" in code

    compile(code, str(path), "exec")
    path.write_text(code)

    print(
        "[PASS] K8 collector created",
        "[PASS] Causal action history enforced",
        "[PASS] Reset handling enabled",
        "[PASS] Alignment assertions enabled",
        sep="\n",
        flush=True
    )


def run_logged(cmd, log):
    env = os.environ.copy()

    env.pop("CUDA_VISIBLE_DEVICES", None)

    env["PYTHONPATH"] = (
        str(ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    with log.open("w") as f:
        process = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed: {process.returncode}\n"
            f"Log: {log}\n"
            + "\n".join(
                log.read_text(
                    errors="replace"
                ).splitlines()[-45:]
            )
        )


def collect_one(
    collector, run, level, gpu,
    samples, envs, suffix
):
    name = f"k8_level{level}_{suffix}"

    out = run / name
    log = run / f"{name}.log"

    assert not out.exists(), (
        f"Output already exists: {out}"
    )

    cmd = [
        PYTHON,
        "-u",
        str(collector),

        "--task",
        "Isaac-QuadrotorHoverStudentDistill-v0",

        "--teacher_checkpoint",
        str(TEACHER),

        "--torchscript_policy",
        str(BASE_TS),

        "--output_dir",
        str(out),

        "--round_index",
        "0",

        "--curriculum_level",
        str(level),

        "--teacher_probability",
        "0.0",

        "--mixing_mode",
        "episode",

        "--num_envs",
        str(envs),

        "--num_samples",
        str(samples),

        "--shard_size",
        str(samples),

        "--seed",
        str(SEED),

        "--device",
        f"cuda:{gpu}",
    ]

    print(
        f"[COLLECT START] GPU={gpu}, "
        f"level={level}, samples={samples}",
        flush=True
    )

    run_logged(cmd, log)

    shard = out / "round_00_shard_0000.pt"

    assert shard.is_file(), (
        f"Missing output shard: {shard}"
    )

    d = torch.load(
        shard,
        map_location="cpu",
        weights_only=False
    )

    assert len(d["teacher_actions"]) == samples

    assert (
        torch.as_tensor(
            d["controller_is_teacher"]
        ).float() == 0
    ).all(), "Student did not control all sampled trajectories"

    assert (
        torch.as_tensor(d["student_actions"])
        - torch.as_tensor(d["executed_actions"])
    ).abs().max() < 1e-6

    print(
        f"[COLLECT PASS] level={level}, "
        f"{samples} student-controlled transitions",
        flush=True
    )

    return shard


def original_data():
    spec = importlib.util.spec_from_file_location(
        "k8_dagger_study",
        str(TRAINER)
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    x, h, y, steps, split, valid = module.read_data()

    return module, (
        x, h, y, steps, split, valid
    )


def reconstruct_onpolicy(shards):
    datasets = []

    for level, path in shards:
        d = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        def as_np(key):
            v = d[key]

            return (
                v.detach().cpu().numpy()
                if torch.is_tensor(v)
                else np.asarray(v)
            )

        n = len(d["teacher_actions"])

        item = {
            "x": as_np("observations")
                .reshape(n, 256)
                .astype("float32"),

            "y": as_np("teacher_actions")
                .reshape(n, 4)
                .astype("float32"),

            "prev": as_np("teacher_observations")
                .reshape(n, 20)[:, 15:19]
                .astype("float32"),

            "exec": as_np("executed_actions")
                .reshape(n, 4)
                .astype("float32"),

            "step": as_np("episode_steps")
                .reshape(-1)
                .astype("int64"),

            "env": as_np("environment_ids")
                .reshape(-1)
                .astype("int64"),

            "episode": as_np("episode_ids")
                .reshape(-1)
                .astype("int64"),

            "level": np.full(
                n, float(level), "float32"
            ),
        }

        assert all(
            len(v) == n
            for v in item.values()
        )

        datasets.append(item)

    merged = {
        k: np.concatenate([
            d[k] for d in datasets
        ])
        for k in datasets[0]
    }

    n = len(merged["x"])

    lookup = {}

    for i in range(n):
        key = (
            float(merged["level"][i]),
            int(merged["env"][i]),
            int(merged["episode"][i]),
            int(merged["step"][i]),
        )

        lookup.setdefault(key, []).append(i)

    h = np.zeros(
        (n, 8, 4),
        dtype="float32"
    )

    valid = np.ones(n, dtype=bool)

    for i in range(n):
        level = float(merged["level"][i])
        env = int(merged["env"][i])
        epi = int(merged["episode"][i])
        step = int(merged["step"][i])

        for lag in range(8):
            target_step = step - lag

            if target_step < 0:
                continue

            if lag == 0:
                h[i, 7] = merged["prev"][i]
                continue

            predecessors = lookup.get(
                (level, env, epi, target_step),
                []
            )

            if len(predecessors) != 1:
                valid[i] = False
                break

            p = predecessors[0]

            h[i, 7 - lag] = merged["prev"][p]

            if lag == 1:
                assert np.allclose(
                    merged["prev"][i],
                    merged["exec"][p],
                    atol=1e-6,
                    rtol=0
                ), (
                    "Offline action-history "
                    "reconstruction mismatch"
                )

    assert np.isfinite(merged["x"]).all()
    assert np.isfinite(merged["y"]).all()

    assert (
        np.abs(merged["y"]) <= 1.0001
    ).all()

    assert (
        h[merged["step"] == 0] == 0
    ).all(), "Episode-reset padding invalid"

    print(
        f"[ONPOLICY HISTORY] "
        f"valid={int(valid.sum())}/{n} "
        f"({valid.mean() * 100:.1f}%)",
        flush=True
    )

    if valid.mean() < 0.80:
        raise RuntimeError(
            "Fewer than 80% of samples have "
            "verified 8-action histories."
        )

    # Episode-disjoint holdout of approximately 10%.
    is_val = np.zeros(n, dtype=bool)

    for i in range(n):
        key = (
            float(merged["level"][i]),
            int(merged["env"][i]),
            int(merged["episode"][i]),
        )

        digest = hashlib.sha256(
            ("dagger-round0-" + repr(key)).encode()
        ).digest()

        is_val[i] = (
            int.from_bytes(digest[:8], "big") % 10 == 0
        )

    assert (valid & is_val).sum() > 100
    assert (valid & ~is_val).sum() > 100

    return (
        torch.from_numpy(merged["x"]),
        torch.from_numpy(h),
        torch.from_numpy(merged["y"]),
        torch.from_numpy(merged["step"]),
        valid & ~is_val,
        valid & is_val,
    )


def make_train(
    ckpt_dir,
    shards,
    epochs=12,
    batch=512
):
    torch.set_num_threads(4)

    random.seed(20261120)
    np.random.seed(20261120)
    torch.manual_seed(20261120)

    module, old = original_data()

    xo, ho, yo, so, split, valid = old

    (
        xnew, hnew, ynew, snew,
        train_new, val_new
    ) = reconstruct_onpolicy(shards)

    # Retain the entire previous on-policy collection.
    prior_shards = [
        (
            0,
            ROUND1
            / "k8_level0_full"
            / "round_00_shard_0000.pt",
        ),
        (
            1,
            ROUND1
            / "k8_level1_full"
            / "round_00_shard_0000.pt",
        ),
    ]

    (
        xprior, hprior, yprior, sprior,
        train_prior, val_prior
    ) = reconstruct_onpolicy(prior_shards)

    xnew = torch.cat([xnew, xprior], dim=0)
    hnew = torch.cat([hnew, hprior], dim=0)
    ynew = torch.cat([ynew, yprior], dim=0)
    snew = torch.cat([snew, sprior], dim=0)

    train_new = np.concatenate([
        train_new,
        train_prior
    ])

    val_new = np.concatenate([
        val_new,
        val_prior
    ])

    print(
        "[AGGREGATED DAGGER DATA] "
        f"new_and_prior={len(xnew)}, "
        f"train={train_new.sum()}, "
        f"val={val_new.sum()}",
        flush=True
    )

    iot = np.flatnonzero(
        valid & (split == 0)
    )

    iov = np.flatnonzero(
        valid & (split == 1)
    )

    int_ = np.flatnonzero(train_new)
    inv = np.flatnonzero(val_new)

    print(
        f"[FINE-TUNE DATA] "
        f"old_train={len(iot)}, "
        f"old_val={len(iov)}, "
        f"new_train={len(int_)}, "
        f"new_val={len(inv)}",
        flush=True
    )

    device = torch.device("cuda:0")

    model = module.Policy(8).to(device)

    initial = torch.load(
        BASE_CKPT,
        map_location="cpu",
        weights_only=False
    )

    assert initial["k"] == 8

    model.load_state_dict(
        initial["model_state_dict"],
        strict=True
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.estimator.parameters(),
                "lr": 1e-5,
            },
            {
                "params": model.head.parameters(),
                "lr": 5e-5,
            },
        ],
        weight_decay=1e-4
    )

    old_t = (
        xo[iot].to(device),
        ho[iot].to(device),
        yo[iot].to(device),
        so[iot].to(device),
    )

    new_t = (
        xnew[int_].to(device),
        hnew[int_].to(device),
        ynew[int_].to(device),
        snew[int_].to(device),
    )

    old_v = (
        xo[iov].to(device),
        ho[iov].to(device),
        yo[iov].to(device),
        so[iov].to(device),
    )

    new_v = (
        xnew[inv].to(device),
        hnew[inv].to(device),
        ynew[inv].to(device),
        snew[inv].to(device),
    )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    def error(model, data, max_batch):
        X, H, Y, S = data

        loss_sum = 0.0
        weight_sum = 0.0
        mse_sum = 0.0

        with torch.inference_mode():
            for i in range(0, len(X), max_batch):
                j = slice(i, i + max_batch)

                e = (
                    model(X[j], H[j]) - Y[j]
                ).square().mean(dim=1)

                w = torch.where(
                    S[j] < 31,
                    2.0,
                    1.0
                )

                loss_sum += (e * w).sum().item()
                weight_sum += w.sum().item()
                mse_sum += e.sum().item()

        return (
            loss_sum / weight_sum,
            math.sqrt(mse_sum / len(X))
        )

    model.eval()

    baseline = {}

    for name, ds in [
        ("original_val", old_v),
        ("onpolicy_val", new_v),
    ]:
        baseline[name] = error(
            model, ds, batch
        )

    print(
        "[BASELINE VALIDATION]",
        baseline,
        flush=True
    )

    history = []
    best = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()

        rng = torch.Generator(device=device)
        rng.manual_seed(20261120 + epoch)

        iters = max(
            math.ceil(
                len(old_t[0]) / (batch // 2)
            ),
            math.ceil(
                len(new_t[0]) / (batch // 2)
            )
        )

        for _ in range(iters):
            oi = torch.randint(
                len(old_t[0]),
                (batch // 2,),
                generator=rng,
                device=device
            )

            ni = torch.randint(
                len(new_t[0]),
                (batch // 2,),
                generator=rng,
                device=device
            )

            X = torch.cat([
                old_t[0][oi],
                new_t[0][ni]
            ])

            H = torch.cat([
                old_t[1][oi],
                new_t[1][ni]
            ])

            Y = torch.cat([
                old_t[2][oi],
                new_t[2][ni]
            ])

            S = torch.cat([
                old_t[3][oi],
                new_t[3][ni]
            ])

            pred = model(X, H)

            e = (
                pred - Y
            ).square().mean(dim=1)

            w = torch.where(
                S < 31,
                2.0,
                1.0
            )

            loss = (
                (e * w).sum() / w.sum()
            )

            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"Nonfinite loss at epoch {epoch}"
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

        model.eval()

        ov, ormse = error(
            model, old_v, batch
        )

        nv, nrmse = error(
            model, new_v, batch
        )

        combined = 0.5 * ov + 0.5 * nv

        row = {
            "epoch": epoch,
            "original_val_mse": ov,
            "original_val_rmse": ormse,
            "onpolicy_val_mse": nv,
            "onpolicy_val_rmse": nrmse,
            "combined_val": combined,
        }

        history.append(row)

        print(
            f"[EPOCH {epoch:02d}] "
            f"old_rmse={ormse:.4f} "
            f"new_rmse={nrmse:.4f} "
            f"combined={combined:.6f}",
            flush=True
        )

        if combined < best:
            best = combined

            torch.save(
                {
                    "model_state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v
                        in model.state_dict().items()
                    },
                    "k": 8,
                    "epoch": epoch,
                    "combined_val": combined,
                    "original_val_rmse": ormse,
                    "onpolicy_val_rmse": nrmse,
                },
                ckpt_dir / "policy.pt"
            )

    with (
        ckpt_dir / "training.csv"
    ).open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0])
        )

        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(
        ckpt_dir / "policy.pt",
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model = model.cpu().eval()

    with torch.inference_mode():
        example = (
            torch.zeros(2, 256),
            torch.zeros(2, 8, 4)
        )

        traced = torch.jit.trace(
            model,
            example,
            check_trace=True
        )

        torch.testing.assert_close(
            traced(*example),
            model(*example),
            atol=1e-5,
            rtol=1e-5
        )

        traced.save(
            str(ckpt_dir / "policy.ts")
        )

    (
        ckpt_dir / "training_summary.json"
    ).write_text(
        json.dumps(
            {
                "baseline_val": baseline,
                "best_epoch": checkpoint["epoch"],
                "best_combined_val": best,
            },
            indent=2
        )
    )

    print(
        "[FINETUNE PASS] Exported",
        ckpt_dir / "policy.ts",
        flush=True
    )


def evaluate_one(
    run, label, policy,
    level, gpu, seeds
):
    results = []

    for seed in seeds:
        tag = (
            f"{label}_L{level}_seed{seed}"
        )

        out = run / tag

        cmd = [
            PYTHON,
            "-u",
            str(EVAL),

            "--task",
            "Isaac-QuadrotorHoverStudentDistill-v0",

            "--policy",
            str(policy),

            "--variant",
            label,

            "--level",
            str(level),

            "--num-envs",
            "64",

            "--seed",
            str(seed),

            "--steps",
            "500",

            "--device",
            f"cuda:{gpu}",

            "--out",
            str(out),
        ]

        print(
            f"[EVAL START] {tag}",
            flush=True
        )

        run_logged(
            cmd,
            run / f"{tag}.log"
        )

        summary = json.loads(
            (out / "summary.json").read_text()
        )

        assert (
            summary["completed"]
            + summary["censored"]
            == 64
        )

        assert (
            summary["expected_full_horizon_steps"]
            == 499
        )

        assert summary["num_envs"] == 64

        with (
            out / "episodes.csv"
        ).open() as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 64

        assert all(
            int(r["length"]) >= 499
            for r in rows
            if int(r["timeout"]) == 1
        ), "Premature timeout detected"

        results.append(summary)

        print(
            f"[EVAL PASS] {tag} "
            f"survival="
            f"{summary['full_horizon_survival_count']}/64 "
            f"length={summary['mean_length']:.1f} "
            f"pos={summary['mean_position_rms_m']:.3f} "
            f"speed={summary['mean_speed_rms_mps']:.3f}",
            flush=True
        )

    return results


def main(args):
    check_files()

    assert torch.cuda.device_count() >= 2, (
        "Two visible CUDA GPUs required"
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    run = BASE / (
        f"k8_onpolicy_dagger_round2_{timestamp}"
    )

    run.mkdir(
        parents=True,
        exist_ok=False
    )

    manifest = {
        "base_checkpoint": str(BASE_CKPT),
        "base_torchscript": str(BASE_TS),
        "teacher": str(TEACHER),
        "collection_per_level": N_PER_LEVEL,
        "curriculum_levels": [0, 1],
        "teacher_probability": 0.0,
        "train_old_new_ratio": "50% demonstrations, "
        "25% Round 1, 25% Round 2",
        "lr_encoder": 1e-5,
        "lr_head": 5e-5,
        "epochs": args.epochs,
    }

    (
        run / "manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2)
    )

    print(
        "[RUN DIRECTORY]",
        run,
        flush=True
    )

    collector = run / "collect_k8_dagger.py"

    make_collector(collector)

    # Small integration test before full collection.
    collect_one(
        collector,
        run,
        0,
        0,
        256,
        16,
        "smoke"
    )

    print(
        "[SMOKE PASS] Starting full collection",
        flush=True
    )

    with ThreadPoolExecutor(
        max_workers=2
    ) as pool:

        f0 = pool.submit(
            collect_one,
            collector,
            run,
            0,
            0,
            N_PER_LEVEL,
            64,
            "full"
        )

        f1 = pool.submit(
            collect_one,
            collector,
            run,
            1,
            1,
            N_PER_LEVEL,
            64,
            "full"
        )

        path0 = f0.result()
        path1 = f1.result()

    print(
        "[DATA READY] 100,000 student-controlled "
        "transitions with PPO teacher labels",
        flush=True
    )

    ckpt_dir = run / "finetuned_k8"

    make_train(
        ckpt_dir,
        [
            (0, path0),
            (1, path1)
        ],
        args.epochs,
        args.batch
    )

    # Verify the trained policy in a short real rollout.
    test = run / "finetuned_smoke"

    cmd = [
        PYTHON,
        "-u",
        str(EVAL),

        "--task",
        "Isaac-QuadrotorHoverStudentDistill-v0",

        "--policy",
        str(ckpt_dir / "policy.ts"),

        "--variant",
        "finetuned_smoke",

        "--level",
        "0",

        "--num-envs",
        "4",

        "--seed",
        "5300",

        "--steps",
        "25",

        "--device",
        "cuda:0",

        "--out",
        str(test),
    ]

    run_logged(
        cmd,
        run / "finetuned_smoke.log"
    )

    smoke_summary = json.loads(
        (test / "summary.json").read_text()
    )

    assert smoke_summary["num_envs"] == 4
    assert smoke_summary["max_steps"] == 25

    assert (
        smoke_summary["completed"]
        + smoke_summary["censored"]
        == 4
    )

    print(
        "[SMOKE PASS] Fine-tuned K8 executed "
        "in closed-loop simulation",
        flush=True
    )

    with ThreadPoolExecutor(
        max_workers=2
    ) as pool:

        f0 = pool.submit(
            evaluate_one,
            run,
            "dagger_round2",
            ckpt_dir / "policy.ts",
            0,
            0,
            (5300, 5301, 5302)
        )

        f1 = pool.submit(
            evaluate_one,
            run,
            "dagger_round2",
            ckpt_dir / "policy.ts",
            1,
            1,
            (5300, 5301, 5302)
        )

        results = {
            0: f0.result(),
            1: f1.result(),
        }

    print(
        "\n========== K8: BASELINE VS DAGGER ROUND 1 ==========",
        flush=True
    )

    summary = {}

    for level in (0, 1):
        r = results[level]
        n = 64 * len(r)

        row = {
            "episodes": n,

            "survival": sum(
                v["full_horizon_survival_count"]
                for v in r
            ),

            "mean_length": sum(
                v["mean_length"] for v in r
            ) / len(r),

            "mean_position_rms_m": sum(
                v["mean_position_rms_m"]
                for v in r
            ) / len(r),

            "mean_speed_rms_mps": sum(
                v["mean_speed_rms_mps"]
                for v in r
            ) / len(r),

            "tight_holds": sum(
                v["final_two_second_tight_hold_count"]
                for v in r
            ),
        }

        summary[level] = row

        baseline_survival = (
            98 if level == 0 else 109
        )

        baseline_length = (
            403.3 if level == 0 else 425.5
        )

        baseline_pos = (
            0.441 if level == 0 else 0.416
        )

        print(
            f"L{level} ROUND1 K8 "
            f"survival={baseline_survival}/192 "
            f"length={baseline_length:.1f} "
            f"pos={baseline_pos:.3f} "
            f"holds=0/192"
        )

        print(
            f"L{level} DAGGER "
            f"survival={row['survival']}/192 "
            f"length={row['mean_length']:.1f} "
            f"pos={row['mean_position_rms_m']:.3f} "
            f"speed={row['mean_speed_rms_mps']:.3f} "
            f"holds={row['tight_holds']}/192",
            flush=True
        )

    (
        run / "comparison.json"
    ).write_text(
        json.dumps(summary, indent=2)
    )

    print(
        "[RESULT DIRECTORY]",
        run,
        flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=12
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=512
    )

    args = parser.parse_args()

    assert args.epochs > 0
    assert args.batch >= 64
    assert args.batch % 2 == 0

    main(args)
