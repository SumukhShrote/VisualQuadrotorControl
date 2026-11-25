"""Reference pose-attention training and simulation runtime.

Extracted from the validated November 24, 2026 implementation.
Importing this module does not launch Isaac Sim.

configure() must be called after AppLauncher initialization.
One simulation context is supported per process.
"""

import ast
import copy
import inspect
import json
import math
import os
import random
import re
import sys
import textwrap

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from networks.spatial_pose_attention import PoseAttention
from controllers.pose_attention_policy import Augmented
from utils.marker_projection import rotations, project
from utils.checkpoint_utils import load_visual_baseline

ROOT = Path(__file__).resolve().parents[1]

# Retained only for any historical dependency identified
# during extraction. Remove after verifying those dependencies.

REF = ROOT / "checkpoints/reference"

BASE = REF / "visual_baseline/ppo_checkpoint.pt"
REFERENCE = REF / "visual_baseline/policy.ts"
TEACHER = REF / "teacher/teacher.pt"

TASK = "Isaac-QuadrotorHoverStudentDistill-v0"

RESET = {
    "enabled": False,
    "scale": 1.0,
}

a = None
OUT = None
DEV = None


def configure(args):
    global a, OUT, DEV

    a = args

    OUT = Path(a.out).resolve()
    OUT.mkdir(parents=True, exist_ok=False)

    DEV = torch.device(a.device)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    RESET["enabled"] = False
    RESET["scale"] = 1.0


def load_base():
    model = load_visual_baseline(
        checkpoint=BASE,
        device=DEV,
        frozen=True,
    )

    log("[PASS] K=8 baseline loaded", BASE)

    return model

def log(*x):
    print(*x, flush=True)

def save(obj, name):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(p)

def report(obj):
    log(json.dumps(obj))
    with (OUT / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(obj) + "\n")

def uniform(n, scale):
    return (
        2 * torch.rand(n, len(scale), device=DEV) - 1
    ) * torch.tensor(scale, device=DEV)

def poses(n, mode="mixed"):
    off = uniform(n, [.65, .65, .45])
    rpy = uniform(n, [.25, .25, .42])

    k = torch.rand(n, device=DEV)

    # Permanent near-hover population.
    near = k < .35
    off[near] *= .25
    rpy[near] *= .35

    # Deliberate lateral-yaw compensated configurations.
    compensated = (k >= .35) & (k < .65)

    if mode == "compensated":
        compensated[:] = True

    rpy[compensated, 2] = torch.atan2(
        -off[compensated, 1],
        1.47 - off[compensated, 0],
    )

    if mode != "compensated":
        rpy[compensated, 2] += uniform(
            int(compensated.sum()), [.035]
        )[:, 0]

    uv, valid = project(off, rpy)

    target = torch.cat((
        -off,
        torch.sin(rpy[:, 2:3]),
        torch.cos(rpy[:, 2:3]) - 1,
    ), -1)

    return uv[valid], target[valid]

def pose_metrics(model, uv, target):
    _, p = model(uv)

    err = (
        (p[:, :3] - target[:, :3])
        .square()
        .mean(0)
        .sqrt()
    )

    yp = torch.atan2(p[:, 3], p[:, 4] + 1)
    yt = torch.atan2(target[:, 3], target[:, 4] + 1)

    yaw = torch.atan2(
        torch.sin(yp - yt),
        torch.cos(yp - yt),
    ).square().mean().sqrt() * 180 / math.pi

    return {
        "x_rmse": float(err[0]),
        "y_rmse": float(err[1]),
        "z_rmse": float(err[2]),
        "yaw_rmse_deg": float(yaw),
    }

def pose_gate(m):
    return (
        m["y_rmse"] <= .035
        and m["yaw_rmse_deg"] <= 2.0
    )

def validate_archived_pose(model):
    import cv2
    archive = ROOT / 'results/diagnostics/frozen_representation/seed_9513_distill/frozen_features.npz'
    with np.load(archive) as data:
        corners = data['corners'][::8, -8:].reshape(-1, 4, 2)
        truth = data['teacher_state'][::8, :3]
    obj = np.array([[-0.05, 0.05, 0], [0.05, 0.05, 0], [0.05, -0.05, 0], [-0.05, -0.05, 0]], np.float64)
    K = np.array([[128 * 24 / 20.955, 0, 64], [0, 128 * 24 / 20.955, 64], [0, 0, 1]], np.float64)
    Rwo = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], np.float64)
    Rcb = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], np.float64)
    inputs = []
    labels = []
    check = []
    for (c, t) in zip(corners, truth):
        if not np.isfinite(c).all():
            continue
        if not ((c >= 0) & (c <= 1)).all():
            continue
        if cv2.contourArea(c.astype(np.float32)) < 1e-05:
            continue
        (n, rv, tv, reproj) = cv2.solvePnPGeneric(obj, c.astype(np.float64) * 128, K, None, flags=cv2.SOLVEPNP_IPPE)
        if not n:
            continue
        j = int(np.argmin(reproj))
        Rco = cv2.Rodrigues(rv[j])[0]
        Rwb = Rwo @ Rco.T @ Rcb
        p = -Rwo @ Rco.T @ tv[j].reshape(3)
        e = np.array([-1.47, 0, 0]) - p
        y = np.arctan2(Rwb[1, 0], Rwb[0, 0])
        check.append(Rwb.T @ e - t)
        inputs.append(c)
        labels.append(np.r_[e, np.sin(y), np.cos(y) - 1])
    if len(inputs) < 500:
        raise RuntimeError('Insufficient archived pose samples')
    convention_rmse = np.sqrt(np.mean(np.asarray(check) ** 2))
    if convention_rmse > 0.003:
        raise RuntimeError('Archived pose convention check failed')
    metrics = pose_metrics(model, torch.tensor(np.asarray(inputs), device=DEV, dtype=torch.float32), torch.tensor(np.asarray(labels), device=DEV, dtype=torch.float32))
    report({'stage': 'archived_pose_validation', 'samples': len(inputs), 'convention_rmse': float(convention_rmse), **metrics})
    if not pose_gate(metrics):
        raise RuntimeError('Pose branch did not generalize to simulator observations')

def pretrain():
    log("=" * 70)
    log("STAGE 1: SUPERVISED POSE ATTENTION")
    log("=" * 70)

    model = PoseAttention().to(DEV)

    import os

    resume_pose = (
        getattr(a, "pose_checkpoint", None)
        or os.environ.get("POSE_RESUME")
    )

    if resume_pose:
        ck = torch.load(
            resume_pose,
            map_location=DEV,
            weights_only=False,
        )

        model.load_state_dict(ck["model"], strict=True)

        model.eval().requires_grad_(False)

        for key in ("mixed", "compensated"):
            if not pose_gate(ck[key]):
                raise RuntimeError(
                    f"Saved pose checkpoint failed {key} gate"
                )

        save(ck, "pose_best.pt")

        log("[PASS] Loaded pretrained pose attention")
        log("[POSE CHECKPOINT]", resume_pose)
        log("[POSE TRAINING STEP]", ck["step"])

        report({
            "stage": "pose_resumed",
            "step": ck["step"],
            "mixed": ck["mixed"],
            "compensated": ck["compensated"],
        })

        validate_archived_pose(model)

        log("[PASS] Archived simulator pose gate")
        log("[SKIP] No repeated pose pretraining required")

        return model

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
    )

    # Separate validation and training RNG streams.
    torch.manual_seed(a.seed + 1000)

    val = [
        poses(24000, m)
        for m in ("mixed", "compensated")
    ]

    torch.manual_seed(a.seed + 2000)
    best = float("inf")

    for i in range(1, a.pose_updates + 1):
        uv, t = poses(2048)

        # Small measurement perturbations. This is primarily
        # a clean analytic-corner experiment, not yet a
        # full real-camera robustness experiment.
        noisy = torch.rand(
            len(uv), 1, 1, device=DEV
        ) < .2

        uv = uv + (
            torch.randn_like(uv)
            * (.005 / 128.0)
            * noisy
        )

        _, p = model(uv)

        loss = (
            ((p - t) / model.scale)
            .square()
            .mean()
        )

        opt.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )

        opt.step()

        if i % 100 == 0 or i == a.pose_updates:
            m, c = [
                pose_metrics(model, *v)
                for v in val
            ]

            score = (
                m["y_rmse"]
                + c["y_rmse"]
                + .015 * (
                    m["yaw_rmse_deg"]
                    + c["yaw_rmse_deg"]
                )
            )

            if score < best:
                best = score

                save({
                    "model": model.state_dict(),
                    "mixed": m,
                    "compensated": c,
                    "step": i,
                }, "pose_best.pt")

            if i % 500 == 0 or i == a.pose_updates:
                report({
                    "stage": "pose",
                    "step": i,
                    "mixed": m,
                    "compensated": c,
                })

    ck = torch.load(
        OUT / "pose_best.pt",
        map_location=DEV,
    )

    model.load_state_dict(ck["model"])
    model.eval().requires_grad_(False)

    for m in (ck["mixed"], ck["compensated"]):
        if not pose_gate(m):
            raise RuntimeError(
                "Pose validation gate failed: " + str(m)
            )

    log("[PASS] Synthetic pose gate")
    log(ck["mixed"], ck["compensated"])

    validate_archived_pose(model)

    log("[PASS] Archived simulator pose gate")

    return model

def expanded_reset(self, ids, root):
    if not RESET["enabled"]:
        return

    n = len(ids)

    l = float(self.cfg.forced_curriculum_level)
    l = max(0., min(1., l)) * RESET["scale"]

    severity = .15 + .85 * l

    k = torch.rand(n, device=DEV)

    near = k < .25
    decouple = (k >= .25) & (k < .50)
    moving = (k >= .50) & (k < .70)

    # Sample independent position and orientation.
    # Reject poses with insufficient marker visibility.
    for retry in range(12):
        off = uniform(
            n, [.65, .65, .40]
        ) * severity

        rpy = uniform(
            n, [.23, .23, .40]
        ) * severity

        # 25%: precision resets.
        off[near] = uniform(
            int(near.sum()), [.12, .20, .12]
        )

        rpy[near] *= .10

        # 25%: targeted lateral-yaw decoupling.
        off[decouple, 0] *= .25
        off[decouple, 2] *= .25
        rpy[decouple, :2] *= .15

        compensated = (
            decouple
            & (torch.rand(n, device=DEV) < .5)
        )

        rpy[compensated, 2] = torch.atan2(
            -off[compensated, 1],
            1.47 - off[compensated, 0],
        )

        _, ok = project(off, rpy)

        if retry == 0:
            final_off = off.clone()
            final_rpy = rpy.clone()
            valid = ok.clone()
        else:
            take = (~valid) & ok

            final_off[take] = off[take]
            final_rpy[take] = rpy[take]
            valid |= ok

        if bool(valid.all()):
            break

    final_off[~valid] = 0
    final_rpy[~valid] = 0

    r, p, y = (final_rpy / 2).unbind(-1)

    cr, cp, cy = r.cos(), p.cos(), y.cos()
    sr, sp, sy = r.sin(), p.sin(), y.sin()

    quat = torch.stack((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ), -1)

    vel = uniform(
        n, [.7, .7, .55]
    ) * severity

    ang = uniform(
        n, [.8, .8, 1.2]
    ) * severity

    # 20%: high-velocity disturbances.
    vel[moving] = uniform(
        int(moving.sum()), [1.1, 1.4, .8]
    ) * severity

    vel[near | decouple] *= .05
    ang[near | decouple] *= .05

    root[:, :3] = self._hover_point_w[ids] + final_off
    root[:, 3:7] = quat
    root[:, 7:10] = vel
    root[:, 10:13] = ang

def patch_reset(owner):
    """
    Install optional reset modification in memory.
    Never overwrite the original environment source file.
    """
    original = owner._reset_idx

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(original))
    )

    class Patch(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            self.generic_visit(node)

            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "super"
                and not node.args
                and not node.keywords
            ):
                node.args = [
                    ast.Name(
                        id="_overnight_owner",
                        ctx=ast.Load(),
                    ),
                    ast.Name(
                        id="self",
                        ctx=ast.Load(),
                    ),
                ]

            return node

        def visit_Expr(self, node):
            self.generic_visit(node)

            f = node.value

            if (
                isinstance(f, ast.Call)
                and isinstance(f.func, ast.Attribute)
                and f.func.attr == "write_root_pose_to_sim"
            ):
                if (
                    not f.args
                    or not isinstance(f.args[0], ast.Subscript)
                    or not isinstance(
                        f.args[0].value, ast.Name
                    )
                ):
                    raise RuntimeError(
                        "Unexpected root-state write"
                    )

                name = f.args[0].value.id
                self.count += 1

                injected = ast.parse(
                    f"_overnight_reset(self, env_ids, {name})"
                ).body[0]

                return [
                    ast.copy_location(injected, node),
                    node,
                ]

            return node

    patch = Patch()
    tree = patch.visit(tree)
    ast.fix_missing_locations(tree)

    if patch.count != 1:
        raise RuntimeError(
            f"Expected one root pose write, found {patch.count}"
        )

    g = original.__globals__

    g["_overnight_owner"] = owner
    g["_overnight_reset"] = expanded_reset

    local = {}

    exec(
        compile(tree, "<overnight reset>", "exec"),
        g,
        local,
    )

    owner._reset_idx = local[original.__name__]

    log("[PASS] Expanded reset installed in memory")

def teacher_load():
    path = TEACHER.parent / 'agent.yaml'
    if not path.is_file():
        raise RuntimeError('Teacher agent.yaml missing: ' + str(path))
    matches = re.findall('^\\s*activation:\\s*[\\\'"]?([a-zA-Z_]+)', path.read_text(), re.M)
    if not matches or len(set(matches)) != 1 or matches[0].lower() not in ('tanh', 'elu', 'relu'):
        raise RuntimeError('Cannot establish teacher activation: ' + str(matches))
    act = {'tanh': nn.Tanh, 'elu': nn.ELU, 'relu': nn.ReLU}[matches[0].lower()]
    model = nn.Sequential(nn.Linear(20, 64), act(), nn.Linear(64, 64), act(), nn.Linear(64, 4)).to(DEV)
    sd = torch.load(TEACHER, map_location=DEV, weights_only=False)['model_state_dict']
    model.load_state_dict({k[6:]: v for (k, v) in sd.items() if k.startswith('actor.')}, strict=True)
    log('[PASS] Teacher actor loaded', matches[0])
    return model.eval().requires_grad_(False)

def reset(env, level, expanded, seed):
    RESET["enabled"] = expanded

    env.unwrapped.cfg.forced_curriculum_level = float(level)

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs, _ = env.reset(seed=seed)

    episode_buf = env.unwrapped.episode_length_buf

    before_min = int(episode_buf.min())
    before_max = int(episode_buf.max())
    before_mean = float(episode_buf.float().mean())

    # All environments must start their first evaluation
    # episode at step zero.
    episode_buf.zero_()

    assert int(episode_buf.max()) == 0

    log(
        "[RESET SYNC]",
        f"before_min={before_min}",
        f"before_max={before_max}",
        f"before_mean={before_mean:.2f}",
        "after_min=0",
        "after_max=0",
    )

    assert obs["policy"].shape[-1] == 256
    assert obs["teacher"].shape[-1] == 20

    h = torch.zeros(
        a.num_envs, 8, 4, device=DEV
    )

    return obs, h

def shift(h, u, done):
    h = torch.roll(h, -1, 1)
    h[:, -1] = u
    h[done] = 0

    return h

@torch.no_grad()
def evaluate(
    env, policy, teacher, kind,
    level, expanded, seed,
):
    obs, h = reset(env, level, expanded, seed)

    n = a.num_envs

    live = torch.ones(
        n, dtype=torch.bool, device=DEV
    )

    survival = torch.zeros_like(live)

    lengths = torch.zeros(
        n,
        dtype=torch.long,
        device=DEV,
    )

    hold = torch.zeros(n, device=DEV)
    sums = torch.zeros(n, device=DEV)
    counts = torch.zeros(n, device=DEV)
    lateral = torch.zeros(n, device=DEV)
    speed = torch.zeros(n, device=DEV)

    for step in range(500):
        lengths += live.long()

        p = obs["teacher"]

        d = p[:, :3].norm(dim=-1)
        v = p[:, 3:6].norm(dim=-1)

        if step >= 399:
            sums += d * live
            lateral += p[:, 1].abs() * live
            speed += v * live
            counts += live

            hold += (
                (d < .10)
                & (v < .20)
                & live
            )

        if kind == "teacher":
            u = teacher(p)
        else:
            u = policy(obs["policy"], h)

        u = u.clamp(-1, 1)

        obs, _, term, trunc, _ = env.step(u)

        done = term | trunc

        survival |= live & trunc & ~term
        live &= ~done

        h = shift(h, u, done)

        if not bool(live.any()):
            break

    # A surviving episode must actually have been
    # observed throughout the final precision window.
    incomplete = survival & (counts < 99)

    if bool(incomplete.any()):
        log(
            "[INVALID EVALUATION]",
            "surviving_episodes_without_full_window=",
            int(incomplete.sum()),
        )

        log(
            "[EPISODE LENGTHS]",
            lengths[incomplete][:20].cpu().tolist(),
        )

        log(
            "[WINDOW COUNTS]",
            counts[incomplete][:20].cpu().tolist(),
        )

        raise RuntimeError(
            "Surviving episodes lack complete precision windows"
        )

    ok = survival & (hold >= 99)

    if bool(survival.any()):
        log(
            "[EVALUATION WINDOW]",
            "surviving_length_min=",
            int(lengths[survival].min()),
            "surviving_length_max=",
            int(lengths[survival].max()),
            "window_samples_min=",
            int(counts[survival].min()),
            "window_samples_max=",
            int(counts[survival].max()),
        )

    settled = sums / counts.clamp_min(1)
    settled_y = lateral / counts.clamp_min(1)
    settled_v = speed / counts.clamp_min(1)

    return {
        "survival": float(survival.float().mean()),
        "hold": float(ok.float().mean()),
        "mean_episode_length": float(lengths.float().mean()),
        "min_surviving_length": (
            int(lengths[survival].min())
            if survival.any() else 0
        ),
        "max_surviving_length": (
            int(lengths[survival].max())
            if survival.any() else 0
        ),
        "mean_surviving_window_samples": (
            float(counts[survival].mean())
            if survival.any() else 0.0
        ),
        "settled_error": (
            float(settled[survival].mean())
            if survival.any() else 10.
        ),
        "settled_lateral_error": (
            float(settled_y[survival].mean())
            if survival.any() else 10.
        ),
        "settled_speed": (
            float(settled_v[survival].mean())
            if survival.any() else 10.
        ),
        "level": level,
        "expanded": expanded,
        "seed": seed,
        "kind": kind,
    }

def export(model, name):
    """Export an inference-only copy without changing training weights."""

    inference = copy.deepcopy(model)
    inference.eval().requires_grad_(False)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(4300)

    def inputs(batch):
        c = torch.rand(
            batch, 256, generator=generator
        ).to(DEV)

        h = (
            torch.rand(
                batch, 8, 4, generator=generator
            ) * 2 - 1
        ).to(DEV)

        return c, h

    target = OUT / name
    temporary = target.with_suffix(target.suffix + ".tmp")

    assert not temporary.exists(), temporary

    with torch.no_grad():
        example = inputs(2)

        traced = torch.jit.trace(
            inference,
            example,
            check_trace=False,
        )

        cases = [
            inputs(1),
            inputs(2),
            inputs(32),
            inputs(128),
            (
                torch.zeros(2, 256, device=DEV),
                torch.zeros(2, 8, 4, device=DEV),
            ),
        ]

        maximum = 0.0

        for c, h in cases:
            expected = inference(c, h)
            actual = traced(c, h)

            torch.testing.assert_close(
                expected, actual,
                atol=1e-5, rtol=1e-4,
            )

            maximum = max(
                maximum,
                float((expected - actual).abs().max()),
            )

        traced.save(str(temporary))

        reloaded = torch.jit.load(
            str(temporary), map_location=DEV
        ).eval()

        for c, h in cases:
            expected = inference(c, h)
            actual = reloaded(c, h)

            torch.testing.assert_close(
                expected, actual,
                atol=1e-5, rtol=1e-4,
            )

            maximum = max(
                maximum,
                float((expected - actual).abs().max()),
            )

    temporary.replace(target)

    log("[EXPORT PASS]", str(target),
        "maximum_error=", maximum)


def validate_flight(env, policy, teacher, kind,
                    level, expanded, seed):

    obs, h = reset(env, level, expanded, seed)

    n = a.num_envs
    horizon = int(env.unwrapped.max_episode_length) - 1
    dt = float(env.unwrapped.step_dt)

    window = round(2.0 / dt)

    live = torch.ones(n, dtype=torch.bool, device=DEV)
    survived = torch.zeros_like(live)

    lengths = torch.zeros(n, dtype=torch.long, device=DEV)
    counts = torch.zeros_like(lengths)
    hold = torch.zeros_like(lengths)

    sums = torch.zeros(n, 3, device=DEV)

    sustained_counts = torch.zeros_like(lengths)
    sustained_good = torch.zeros_like(lengths)
    sustained_peak = torch.zeros(n, device=DEV)

    for step in range(horizon + 1):

        lengths += live.long()

        p = obs["teacher"]

        distance = p[:, :3].norm(dim=-1)
        lateral = p[:, 1].abs()
        speed = p[:, 3:6].norm(dim=-1)

        precise = (distance < .10) & (speed < .20)

        final_window = (
            live
            & (lengths > horizon - window)
            & (lengths <= horizon)
        )

        counts += final_window.long()
        hold += (final_window & precise).long()

        values = torch.stack(
            [distance, lateral, speed], dim=-1
        )

        sums += torch.where(
            final_window[:, None], values, 0.
        )

        sustained = live & (step * dt >= 10.)

        sustained_counts += sustained.long()
        sustained_good += (sustained & precise).long()

        sustained_peak = torch.maximum(
            sustained_peak,
            torch.where(sustained, distance, 0.)
        )

        if kind == "teacher":
            u = teacher(p)
        else:
            u = policy(obs["policy"], h)

        if not torch.isfinite(u).all():
            raise RuntimeError("Nonfinite policy output")

        u = u.clamp(-1, 1)

        obs, _, term, trunc, _ = env.step(u)

        term = term.reshape(-1).bool()
        trunc = trunc.reshape(-1).bool()
        done = term | trunc

        survived |= live & trunc & ~term
        live &= ~done

        h = shift(h, u, done)

        if not live.any():
            break

    if live.any():
        raise RuntimeError("Unfinished evaluation episodes")

    bad = survived & (
        (lengths != horizon) | (counts != window)
    )

    if bad.any():
        raise RuntimeError(
            "Incorrect evaluation window: "
            + str(lengths[bad][:10].cpu().tolist())
        )

    settled = sums / counts.clamp_min(1)[:, None]
    good = survived & (hold == window)

    result = {
        "stage": "independent_validation",
        "policy": kind,
        "seed": seed,
        "level": level,
        "expanded": expanded,
        "duration_s": horizon * dt,
        "num_envs": n,
        "survival_count": int(survived.sum()),
        "survival": float(survived.float().mean()),
        "hold_count": int(good.sum()),
        "hold": float(good.float().mean()),
        "mean_episode_length": float(lengths.float().mean()),
    }

    if survived.any():

        m = settled[survived].mean(0)

        result.update({
            "settled_error_m": float(m[0]),
            "settled_lateral_error_m": float(m[1]),
            "settled_speed_mps": float(m[2]),
        })

        if sustained_counts[survived].min() > 0:

            fraction = (
                sustained_good.float()
                / sustained_counts.clamp_min(1)
            )

            continuous = (
                survived
                & (sustained_counts > 0)
                & (sustained_good == sustained_counts)
            )

            result.update({
                "post_10s_precision_fraction":
                    float(fraction[survived].mean()),

                "continuous_hold_after_10s":
                    float(continuous.float().mean()),

                "post_10s_mean_peak_error_m":
                    float(sustained_peak[survived].mean()),

                "post_10s_worst_error_m":
                    float(sustained_peak[survived].max()),
            })

    report(result)
    return result

def train_main():
    import envs
    from isaaclab_tasks.utils import parse_env_cfg
    from envs.aruco_hover.quadrotor_aruco_hover_env import ArucoHoverTeacherEnv
    for path in (BASE, REFERENCE, TEACHER):
        if not path.is_file():
            raise FileNotFoundError(path)
    log('[PASS] Required checkpoints present')
    report({
        "stage": "training_inputs",
        "base_checkpoint": str(BASE),
        "base_policy": str(REFERENCE),
        "teacher": str(TEACHER),
        "seed": a.seed,
    })

    base = load_base()
    teacher = teacher_load()
    reference = torch.jit.load(str(REFERENCE), map_location=DEV).eval()
    pose = pretrain()
    model = Augmented(base, pose).to(DEV).eval()
    patch_reset(ArucoHoverTeacherEnv)
    cfg = parse_env_cfg(TASK, device=a.device, num_envs=a.num_envs, use_fabric=True)
    cfg.forced_curriculum_level = 1.0
    cfg.enable_onboard_camera = False
    cfg.visual_dr_enabled = False
    cfg.debug_vis = False
    if hasattr(cfg, 'eval_mode'):
        cfg.eval_mode = True
    env = gym.make(TASK, cfg=cfg)
    try:
        (obs, h) = reset(env, 1.0, False, a.seed)
        with torch.no_grad():
            orig = reference(obs['policy'], h)
            aug = model(obs['policy'], h)
            torch.testing.assert_close(orig, aug, atol=0.002, rtol=0.001)
        log('[PASS] Initial policy behavior preserved')
        log('=' * 70)
        log('STAGE 2: EXPANDED CURRICULUM TEACHER GATE')
        log('=' * 70)
        for scale in (1.0, 0.75, 0.5):
            RESET['scale'] = scale
            m = evaluate(env, None, teacher, 'teacher', 1.0, True, a.seed + 700)
            report({'stage': 'teacher_gate', 'scale': scale, **m})
            if m['survival'] >= 0.9 and m['hold'] >= 0.7:
                break
        else:
            raise RuntimeError('Teacher cannot reliably solve expanded resets')
        log('[PASS] Accepted reset severity:', RESET['scale'])
        configs = [(1.0, False, a.seed + 800), (1.0, True, a.seed + 900)]
        baseline = [evaluate(env, reference, teacher, 'base', *c) for c in configs]
        for m in baseline:
            report({'stage': 'baseline', **m})
        best_score = sum((x['hold'] for x in baseline))
        best_round = 0
        cap = 180000
        joint = a.train_representation

        # Frozen mode caches features; end-to-end mode
        # stores raw observations so gradients reach the encoders.
        X = torch.empty(
            cap, 288 if joint else 293, device=DEV
        )
        Y = torch.empty(cap, 4, device=DEV)
        W = torch.empty(cap, device=DEV)
        ptr = 0
        size = 0
        groups = [{
            "params": list(model.head.parameters()),
            "lr": a.head_lr,
        }]

        branches = {
            "geometry": model.base.estimator.geometry,
            "linear_velocity": model.base.estimator.linear_velocity,
            "angular_velocity": model.base.estimator.angular_velocity,
            "spatial_pose": model.pose,
        }

        if joint:
            for name, branch in branches.items():
                branch.requires_grad_(True)

                # Temporal prediction heads do not participate
                # in the deployed control computation.
                if name != "spatial_pose":
                    branch.output.requires_grad_(False)

                groups.append({
                    "params": [
                        p for p in branch.parameters()
                        if p.requires_grad
                    ],
                    "lr": a.representation_lr,
                })

        # Keep dropout disabled, matching deployment behavior.
        # eval() does not disable autograd.
        model.eval()

        trainable = [
            p for group in groups
            for p in group["params"]
        ]

        initial = {
            name: [
                p.detach().clone()
                for p in branch.parameters()
                if p.requires_grad
            ]
            for name, branch in branches.items()
        }

        opt = torch.optim.AdamW(
            groups, weight_decay=1e-6
        )

        def representation_state():
            if joint:
                return {
                    "base_state": model.base.state_dict(),
                    "train_representation": True,
                }
            return {"train_representation": False}

        report({
            "stage": "optimization_config",
            "train_representation": joint,
            "head_lr": a.head_lr,
            "representation_lr": a.representation_lr,
            "batch": a.batch,
            "replay_input_dim": X.shape[1],
            "trainable_parameters": sum(
                p.numel() for p in trainable
            ),
            "pose_checkpoint": a.pose_checkpoint,
        })
        log('=' * 70)
        log('STAGE 3: TARGETED CONTROLLER DISTILLATION')
        log('=' * 70)
        for r in range(1, a.rounds + 1):
            expanded = r % 4 != 0
            level = min(1.0, 0.25 + 0.25 * ((r - 1) // 3)) if expanded else 1.0
            (obs, h) = reset(env, level, expanded, a.seed + 10000 + r)
            beta = 1.0 if r <= 2 else max(0.05, 0.4 * 0.8 ** (r - 3))
            use_teacher = torch.rand(a.num_envs, device=DEV) < beta
            for step in range(a.rollout_steps):
                with torch.no_grad():
                    z = model.features(obs['policy'], h)
                    t = teacher(obs['teacher']).clamp(-1, 1)
                    u = model.head(z).clamp(-1, 1)
                    w = 1.0 + 2.0 * (obs['teacher'][:, :3].norm(dim=-1) < 0.25).float()
                    idx = (torch.arange(a.num_envs, device=DEV) + ptr) % cap
                    if joint:
                        X[idx] = torch.cat((
                            obs["policy"].reshape(-1, 256),
                            h.reshape(-1, 32),
                        ), dim=-1)
                    else:
                        X[idx] = z
                    Y[idx] = t
                    W[idx] = w
                    ptr = (ptr + a.num_envs) % cap
                    size = min(cap, size + a.num_envs)
                    issued = torch.where(use_teacher[:, None], t, u)
                    (obs, _, term, trunc, _) = env.step(issued)
                    done = term | trunc
                    h = shift(h, issued, done)
                    use_teacher = torch.where(done, torch.rand(a.num_envs, device=DEV) < beta, use_teacher)
            for i in range(a.updates_per_round):
                ids = torch.randint(
                    size, (a.batch,), device=DEV
                )

                if joint:
                    pred = model(
                        X[ids, :256],
                        X[ids, 256:].reshape(-1, 8, 4),
                    )
                else:
                    pred = model.head(X[ids])
                loss = ((pred - Y[ids]).square().mean(-1) * W[ids]).mean()
                opt.zero_grad()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite DAgger loss")

                loss.backward()

                if joint and i == 1:
                    norms = {}

                    for name, branch in branches.items():
                        grads = [
                            p.grad.detach().norm()
                            for p in branch.parameters()
                            if p.grad is not None
                        ]

                        if not grads:
                            raise RuntimeError(
                                f"No gradient for {name}"
                            )

                        norm = float(
                            torch.stack(grads).norm()
                        )

                        if not math.isfinite(norm) or norm <= 0:
                            raise RuntimeError(
                                f"Invalid gradient for {name}: {norm}"
                            )

                        norms[name] = norm

                    report({
                        "stage": "representation_gradients",
                        "round": r,
                        "norms": norms,
                    })

                nn.utils.clip_grad_norm_(
                    trainable, 1.0, error_if_nonfinite=True
                )
                opt.step()

            if joint:
                with torch.no_grad():
                    deltas = {}

                    for name, branch in branches.items():
                        current = [
                            p for p in branch.parameters()
                            if p.requires_grad
                        ]

                        deltas[name] = float(torch.stack([
                            (p - q).norm()
                            for p, q in zip(
                                current, initial[name]
                            )
                        ]).norm())

                    report({
                        "stage": "representation_parameter_changes",
                        "round": r,
                        "l2_from_initialization": deltas,
                    })
            save({'head': model.head.state_dict(), 'pose': pose.state_dict(), 'optimizer': opt.state_dict(), 'round': r, 'base': str(BASE),
                  **representation_state(),
                  'reset_scale': RESET['scale']}, 'latest.pt')
            report({'stage': 'dagger', 'round': r, 'level': level, 'expanded': expanded, 'teacher_fraction': beta, 'loss': float(loss.detach()), 'replay': size})
            if r % 3 == 0 or r == a.rounds:
                results = [evaluate(env, model, teacher, 'augmented', *c) for c in configs]
                for m in results:
                    report({'stage': 'validation', 'round': r, **m})
                score = sum((x['hold'] for x in results))
                safe = all((x['survival'] >= max(0.85, b['survival'] - 0.03) for (x, b) in zip(results, baseline)))
                safe &= results[0]['hold'] >= baseline[0]['hold'] - 0.03
                if safe and score > best_score:
                    best_score = score
                    best_round = r
                    save({'head': model.head.state_dict(), 'pose': pose.state_dict(), 'round': r, 'base': str(BASE),
                          **representation_state(),
                          'results': results}, 'best.pt')
                    log('[BEST]', r, score)
                    try:
                        export(model, 'best_policy.ts')
                    except Exception as ex:
                        log('[EXPORT WARNING] Weights saved:', repr(ex))
        report({'stage': 'training_complete', 'best_round': best_round, 'best_score': best_score, 'base_score': sum((x['hold'] for x in baseline))})
        if best_round == 0:
            log('[NO PROMOTION] Original policy remains best')
        else:
            ck = torch.load(OUT / 'best.pt', map_location=DEV, weights_only=False)
            if "base_state" in ck:
                model.base.load_state_dict(
                    ck["base_state"], strict=True
                )
            model.head.load_state_dict(ck['head'])
            model.pose.load_state_dict(ck['pose'])
            export(model, 'best_policy.ts')
            log('=' * 70)
            log('STAGE 4: INDEPENDENT FINAL EVALUATION')
            log('=' * 70)
            for level in (0.0, 0.25, 0.5, 0.75, 1.0):
                for expanded in (False, True):
                    seed = a.seed + 22000 + int(level * 100)
                    for (kind, policy) in (('base', reference), ('augmented', model)):
                        m = evaluate(env, policy, teacher, kind, level, expanded, seed)
                        report({'stage': 'final_test', 'round': best_round, **m})
        (OUT / 'COMPLETE').write_text(str(best_round) + '\n')
    finally:
        env.close()
