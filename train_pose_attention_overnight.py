import argparse
import ast
import copy
import inspect
import json
import math
import random
import re
import sys
import textwrap
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument("--out", required=True)
P.add_argument("--num-envs", type=int, default=512)
P.add_argument("--pose-updates", type=int, default=12000)
P.add_argument("--rounds", type=int, default=24)
P.add_argument("--rollout-steps", type=int, default=192)
P.add_argument("--updates-per-round", type=int, default=180)
P.add_argument("--seed", type=int, default=14001)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(P)
a = P.parse_args()
a.headless = True
a.enable_cameras = False
app = AppLauncher(a).app

import gymnasium as gym
import numpy as np
import torch
from torch import nn

ROOT = Path.cwd()
B = ROOT / "experiments/state_bottleneck"
OUT = Path(a.out).resolve()
OUT.mkdir(parents=True, exist_ok=False)

DEV = torch.device(a.device)

torch.manual_seed(a.seed)
np.random.seed(a.seed)
random.seed(a.seed)

BASE = B / "long_horizon_gae_20261123T032419Z/model_9.pt"
REFERENCE = B / "precision_ab_control_20261123T041432Z/policy_initial.ts"

TEACHER = (
    ROOT / "logs/rsl_rl/CTBR_Hover_Teacher"
    / "2026-06-25_12-41-52/model_749.pt"
)

TASK = "Isaac-QuadrotorHoverStudentDistill-v0"


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


# ============================================================
# SYNTHETIC GEOMETRIC SUPERVISION
# ============================================================

def rotations(rpy):
    r, p, y = rpy.unbind(-1)
    c, s = torch.cos, torch.sin

    cr, cp, cy = c(r), c(p), c(y)
    sr, sp, sy = s(r), s(p), s(y)

    return torch.stack((
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        -sp,
        cp * sr,
        cp * cr,
    ), -1).reshape(-1, 3, 3)


def project(offset, rpy):
    # Calibrated marker-to-hover distance: 1.47 m.
    # Marker side length: 0.10 m.
    pts = offset.new_tensor([
        [1.47, -.05, -.05],
        [1.47,  .05, -.05],
        [1.47,  .05,  .05],
        [1.47, -.05,  .05],
    ])

    body = (pts[None] - offset[:, None]) @ rotations(rpy)

    cam = torch.stack((
        -body[..., 1],
        -body[..., 2],
        body[..., 0],
    ), -1)

    uv = (
        .5
        + (24.0 / 20.955)
        * cam[..., :2]
        / cam[..., 2:].clamp_min(.001)
    )

    valid = (
        (cam[..., 2] > .1).all(-1)
        & ((uv > .035) & (uv < .965)).all((1, 2))
    )

    return uv, valid


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


# ============================================================
# FOURTH ATTENTION BRANCH
# ============================================================

class PoseAttention(nn.Module):
    """
    Spatial self-attention over four ordered marker corners.

    Outputs:
        64-dimensional learned pose features
        5-dimensional supervised pose prediction

    The existing temporal attention branches remain unchanged.
    """

    def __init__(self):
        super().__init__()

        self.embed = nn.Linear(9, 64)
        self.pos = nn.Parameter(torch.zeros(1, 4, 64))

        self.qkv = nn.Linear(64, 192)
        self.mix = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)

        self.ff = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        self.readout = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 5),
        )

        self.register_buffer(
            "scale",
            torch.tensor([.3, .3, .3, .25, .1]),
        )

    def forward(self, c):
        c = c.reshape(-1, 4, 2)
        ctr = c.mean(1, keepdim=True)

        edge = (
            (c - torch.roll(c, -1, 1))
            .square()
            .sum(-1)
            .clamp_min(1e-12)
            .sqrt()
        )

        size = edge.mean(1, keepdim=True).clamp_min(.002)

        shape = (c - ctr) / size[:, :, None]

        # Opposite edge-length differences expose
        # perspective distortion rather than only centering.
        persp = torch.stack((
            edge[:, 0] - edge[:, 2],
            edge[:, 1] - edge[:, 3],
        ), -1) / size * 20

        globals_ = torch.cat((
            (ctr[:, 0] - .5) * 4,
            torch.log(size / .078),
            persp,
        ), -1)

        z = torch.cat((
            shape,
            c - .5,
            globals_[:, None].expand(-1, 4, -1),
        ), -1)

        h = self.embed(z) + self.pos

        q, k, v = self.qkv(h).chunk(3, -1)

        attention = torch.softmax(
            q @ k.transpose(-1, -2) / 8.0,
            -1,
        )

        h = self.norm(h + self.mix(attention @ v))
        h = h + self.ff(h)

        features = self.readout(h.flatten(1))
        prediction = self.decoder(features) * self.scale

        return features, prediction


@torch.no_grad()
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

    archive = (
        B / "frozen_representation_diagnostic_20261123T221903Z"
        / "seed_9513_distill/frozen_features.npz"
    )

    with np.load(archive) as data:
        corners = data["corners"][::8, -8:].reshape(-1, 4, 2)
        truth = data["teacher_state"][::8, :3]

    obj = np.array([
        [-.05,  .05, 0],
        [ .05,  .05, 0],
        [ .05, -.05, 0],
        [-.05, -.05, 0],
    ], np.float64)

    K = np.array([
        [128 * 24 / 20.955, 0, 64],
        [0, 128 * 24 / 20.955, 64],
        [0, 0, 1],
    ], np.float64)

    Rwo = np.array([
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ], np.float64)

    Rcb = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0],
    ], np.float64)

    inputs = []
    labels = []
    check = []

    for c, t in zip(corners, truth):
        if not np.isfinite(c).all():
            continue

        if not ((c >= 0) & (c <= 1)).all():
            continue

        if cv2.contourArea(c.astype(np.float32)) < 1e-5:
            continue

        n, rv, tv, reproj = cv2.solvePnPGeneric(
            obj,
            c.astype(np.float64) * 128,
            K,
            None,
            flags=cv2.SOLVEPNP_IPPE,
        )

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
        raise RuntimeError("Insufficient archived pose samples")

    convention_rmse = np.sqrt(
        np.mean(np.asarray(check)**2)
    )

    if convention_rmse > .003:
        raise RuntimeError("Archived pose convention check failed")

    metrics = pose_metrics(
        model,
        torch.tensor(
            np.asarray(inputs),
            device=DEV,
            dtype=torch.float32,
        ),
        torch.tensor(
            np.asarray(labels),
            device=DEV,
            dtype=torch.float32,
        ),
    )

    report({
        "stage": "archived_pose_validation",
        "samples": len(inputs),
        "convention_rmse": float(convention_rmse),
        **metrics,
    })

    if not pose_gate(metrics):
        raise RuntimeError(
            "Pose branch did not generalize to simulator observations"
        )


def pretrain():
    log("=" * 70)
    log("STAGE 1: SUPERVISED POSE ATTENTION")
    log("=" * 70)

    model = PoseAttention().to(DEV)

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


# ============================================================
# EXPANDED RESET CURRICULUM
# ============================================================

RESET = {
    "enabled": False,
    "scale": 1.0,
}


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


# ============================================================
# LOAD ESTABLISHED TEACHER AND STUDENT
# ============================================================

def teacher_load():
    path = TEACHER.parent / "params/agent.yaml"

    if not path.is_file():
        raise RuntimeError(
            "Teacher agent.yaml missing: " + str(path)
        )

    matches = re.findall(
        r'^\s*activation:\s*[\'"]?([a-zA-Z_]+)',
        path.read_text(),
        re.M,
    )

    if (
        not matches
        or len(set(matches)) != 1
        or matches[0].lower() not in ("tanh", "elu", "relu")
    ):
        raise RuntimeError(
            "Cannot establish teacher activation: "
            + str(matches)
        )

    act = {
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "relu": nn.ReLU,
    }[matches[0].lower()]

    model = nn.Sequential(
        nn.Linear(20, 64),
        act(),
        nn.Linear(64, 64),
        act(),
        nn.Linear(64, 4),
    ).to(DEV)

    sd = torch.load(
        TEACHER,
        map_location=DEV,
        weights_only=False,
    )["model_state_dict"]

    model.load_state_dict({
        k[6:]: v
        for k, v in sd.items()
        if k.startswith("actor.")
    }, strict=True)

    log("[PASS] Teacher actor loaded", matches[0])

    return model.eval().requires_grad_(False)


def load_base():
    import importlib.util

    p = B / "launch_action_history_study.py"

    sys.path.insert(0, str(B))

    spec = importlib.util.spec_from_file_location(
        "overnight_original_policy", p
    )

    mod = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    base = mod.Policy(8).to(DEV)

    sd = torch.load(
        BASE,
        map_location=DEV,
        weights_only=False,
    )["model_state_dict"]

    base.load_state_dict({
        k[6:]: v
        for k, v in sd.items()
        if k.startswith("actor.")
    }, strict=True)

    log("[PASS] Original K=8 policy loaded")

    return base.eval().requires_grad_(False)


# ============================================================
# EXPANDED CONTROLLER
# ============================================================

class Augmented(nn.Module):
    def __init__(self, base, pose):
        super().__init__()

        self.base = base.eval().requires_grad_(False)
        self.pose = pose.eval().requires_grad_(False)

        self.head = copy.deepcopy(base.head)

        first = self.head[0]

        assert isinstance(first, nn.Linear)
        assert first.in_features == 224

        # 224 original features + 64 pose features
        # + 5 supervised pose predictions.
        wide = nn.Linear(
            293, first.out_features
        ).to(DEV)

        # Preserve original controller at initialization.
        with torch.no_grad():
            wide.weight.zero_()
            wide.weight[:, :224].copy_(first.weight)
            wide.bias.copy_(first.bias)

        self.head[0] = wide
        self.head.requires_grad_(True)

    def features(self, c, h):
        c = c.reshape(-1, 32, 8)
        b = self.base

        z = torch.cat([
            b._branch_features(b.estimator.geometry, c),
            b._branch_features(b.estimator.linear_velocity, c),
            b._branch_features(b.estimator.angular_velocity, c),
            h.reshape(-1, 32),
        ], -1)

        f, p = self.pose(c[:, -1])

        valid = (
            c[:, -1].abs().sum(-1, keepdim=True) > 1e-6
        ).to(f.dtype)

        return torch.cat((
            z,
            f * valid,
            p / self.pose.scale * valid,
        ), -1)

    def forward(self, c, h):
        return self.head(self.features(c, h))


# ============================================================
# SIMULATOR HELPERS
# ============================================================

def reset(env, level, expanded, seed):
    RESET["enabled"] = expanded

    env.unwrapped.cfg.forced_curriculum_level = float(level)

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs, _ = env.reset(seed=seed)

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

    hold = torch.zeros(n, device=DEV)
    sums = torch.zeros(n, device=DEV)
    counts = torch.zeros(n, device=DEV)
    lateral = torch.zeros(n, device=DEV)
    speed = torch.zeros(n, device=DEV)

    for step in range(500):
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

    ok = survival & (hold >= 99)

    settled = sums / counts.clamp_min(1)
    settled_y = lateral / counts.clamp_min(1)
    settled_v = speed / counts.clamp_min(1)

    return {
        "survival": float(survival.float().mean()),
        "hold": float(ok.float().mean()),
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
    model.eval()

    c = torch.zeros(2, 256, device=DEV)
    h = torch.zeros(2, 8, 4, device=DEV)

    traced = torch.jit.trace(
        model,
        (c, h),
        check_trace=False,
    )

    test_c = torch.rand_like(c)
    test_h = torch.rand_like(h)

    with torch.no_grad():
        torch.testing.assert_close(
            traced(test_c, test_h),
            model(test_c, test_h),
            atol=1e-5,
            rtol=1e-4,
        )

    traced.save(str(OUT / name))


# ============================================================
# COMPLETE TRAINING PIPELINE
# ============================================================

def main():
    import envs

    from isaaclab_tasks.utils import parse_env_cfg

    from envs.aruco_hover.quadrotor_aruco_hover_env import (
        ArucoHoverTeacherEnv,
    )

    for path in (BASE, REFERENCE, TEACHER):
        if not path.is_file():
            raise FileNotFoundError(path)

    log("[PASS] Required checkpoints present")

    base = load_base()
    teacher = teacher_load()

    reference = torch.jit.load(
        str(REFERENCE),
        map_location=DEV,
    ).eval()

    pose = pretrain()

    model = Augmented(
        base, pose
    ).to(DEV).eval()

    patch_reset(ArucoHoverTeacherEnv)

    cfg = parse_env_cfg(
        TASK,
        device=a.device,
        num_envs=a.num_envs,
        use_fabric=True,
    )

    cfg.forced_curriculum_level = 1.
    cfg.enable_onboard_camera = False
    cfg.visual_dr_enabled = False
    cfg.debug_vis = False

    if hasattr(cfg, "eval_mode"):
        cfg.eval_mode = True

    env = gym.make(TASK, cfg=cfg)

    try:
        obs, h = reset(
            env, 1., False, a.seed
        )

        with torch.no_grad():
            orig = reference(obs["policy"], h)
            aug = model(obs["policy"], h)

            torch.testing.assert_close(
                orig, aug,
                atol=.002,
                rtol=.001,
            )

        log("[PASS] Initial policy behavior preserved")

        # ----------------------------------------------------
        # TEACHER RECOVERABILITY GATE
        # ----------------------------------------------------

        log("=" * 70)
        log("STAGE 2: EXPANDED CURRICULUM TEACHER GATE")
        log("=" * 70)

        for scale in (1., .75, .5):
            RESET["scale"] = scale

            m = evaluate(
                env, None, teacher, "teacher",
                1., True, a.seed + 700,
            )

            report({
                "stage": "teacher_gate",
                "scale": scale,
                **m,
            })

            if (
                m["survival"] >= .90
                and m["hold"] >= .70
            ):
                break

        else:
            raise RuntimeError(
                "Teacher cannot reliably solve expanded resets"
            )

        log("[PASS] Accepted reset severity:", RESET["scale"])

        # ----------------------------------------------------
        # PAIRED ORIGINAL AND EXPANDED BASELINES
        # ----------------------------------------------------

        configs = [
            (1., False, a.seed + 800),
            (1., True, a.seed + 900),
        ]

        baseline = [
            evaluate(
                env, reference, teacher, "base", *c
            )
            for c in configs
        ]

        for m in baseline:
            report({
                "stage": "baseline",
                **m,
            })

        best_score = sum(
            x["hold"] for x in baseline
        )

        best_round = 0

        # ----------------------------------------------------
        # FROZEN FEATURE REPLAY
        # ----------------------------------------------------

        cap = 180000

        X = torch.empty(
            cap, 293, device=DEV
        )

        Y = torch.empty(
            cap, 4, device=DEV
        )

        W = torch.empty(
            cap, device=DEV
        )

        ptr = 0
        size = 0

        opt = torch.optim.AdamW(
            model.head.parameters(),
            lr=1e-4,
            weight_decay=1e-6,
        )

        log("=" * 70)
        log("STAGE 3: TARGETED CONTROLLER DISTILLATION")
        log("=" * 70)

        for r in range(1, a.rounds + 1):
            # Retain original-curriculum experience.
            expanded = r % 4 != 0

            level = (
                min(1., .25 + .25 * ((r - 1) // 3))
                if expanded else 1.
            )

            obs, h = reset(
                env, level, expanded,
                a.seed + 10000 + r,
            )

            # Start with teacher trajectories, then
            # progressively collect student trajectories.
            beta = (
                1. if r <= 2
                else max(.05, .4 * (.80 ** (r - 3)))
            )

            use_teacher = (
                torch.rand(a.num_envs, device=DEV)
                < beta
            )

            for step in range(a.rollout_steps):
                with torch.no_grad():
                    z = model.features(obs["policy"], h)

                    t = teacher(
                        obs["teacher"]
                    ).clamp(-1, 1)

                    u = model.head(z).clamp(-1, 1)

                    # Emphasize corrective actions close
                    # to the target without discarding
                    # disturbance-recovery examples.
                    w = 1. + 2. * (
                        obs["teacher"][:, :3].norm(dim=-1)
                        < .25
                    ).float()

                    idx = (
                        torch.arange(
                            a.num_envs, device=DEV
                        ) + ptr
                    ) % cap

                    X[idx] = z
                    Y[idx] = t
                    W[idx] = w

                    ptr = (
                        ptr + a.num_envs
                    ) % cap

                    size = min(
                        cap, size + a.num_envs
                    )

                    issued = torch.where(
                        use_teacher[:, None],
                        t, u,
                    )

                    obs, _, term, trunc, _ = env.step(
                        issued
                    )

                    done = term | trunc

                    h = shift(h, issued, done)

                    use_teacher = torch.where(
                        done,
                        torch.rand(
                            a.num_envs, device=DEV
                        ) < beta,
                        use_teacher,
                    )

            # Update only the expanded policy head.
            for i in range(a.updates_per_round):
                ids = torch.randint(
                    size, (2048,), device=DEV
                )

                pred = model.head(X[ids])

                loss = (
                    (pred - Y[ids])
                    .square()
                    .mean(-1)
                    * W[ids]
                ).mean()

                opt.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.head.parameters(),
                    1.,
                )

                opt.step()

            save({
                "head": model.head.state_dict(),
                "pose": pose.state_dict(),
                "optimizer": opt.state_dict(),
                "round": r,
                "base": str(BASE),
                "reset_scale": RESET["scale"],
            }, "latest.pt")

            report({
                "stage": "dagger",
                "round": r,
                "level": level,
                "expanded": expanded,
                "teacher_fraction": beta,
                "loss": float(loss.detach()),
                "replay": size,
            })

            # ------------------------------------------------
            # CHECKPOINT SELECTION
            # ------------------------------------------------

            if r % 3 == 0 or r == a.rounds:
                results = [
                    evaluate(
                        env, model, teacher,
                        "augmented", *c
                    )
                    for c in configs
                ]

                for m in results:
                    report({
                        "stage": "validation",
                        "round": r,
                        **m,
                    })

                score = sum(
                    x["hold"] for x in results
                )

                safe = all(
                    x["survival"] >= max(
                        .85,
                        b["survival"] - .03,
                    )
                    for x, b in zip(results, baseline)
                )

                safe &= (
                    results[0]["hold"]
                    >= baseline[0]["hold"] - .03
                )

                if safe and score > best_score:
                    best_score = score
                    best_round = r

                    save({
                        "head": model.head.state_dict(),
                        "pose": pose.state_dict(),
                        "round": r,
                        "base": str(BASE),
                        "results": results,
                    }, "best.pt")

                    log("[BEST]", r, score)

                    try:
                        export(
                            model, "best_policy.ts"
                        )
                    except Exception as ex:
                        log(
                            "[EXPORT WARNING] Weights saved:",
                            repr(ex),
                        )

        # ----------------------------------------------------
        # FINAL INDEPENDENT-SEED EVALUATION
        # ----------------------------------------------------

        report({
            "stage": "training_complete",
            "best_round": best_round,
            "best_score": best_score,
            "base_score": sum(
                x["hold"] for x in baseline
            ),
        })

        if best_round == 0:
            log(
                "[NO PROMOTION] Original policy remains best"
            )

        else:
            ck = torch.load(
                OUT / "best.pt",
                map_location=DEV,
            )

            model.head.load_state_dict(ck["head"])
            model.pose.load_state_dict(ck["pose"])

            export(model, "best_policy.ts")

            log("=" * 70)
            log("STAGE 4: INDEPENDENT FINAL EVALUATION")
            log("=" * 70)

            for level in (0., .25, .5, .75, 1.):
                for expanded in (False, True):
                    seed = (
                        a.seed + 22000 + int(level * 100)
                    )

                    for kind, policy in (
                        ("base", reference),
                        ("augmented", model),
                    ):
                        m = evaluate(
                            env, policy, teacher, kind,
                            level, expanded, seed,
                        )

                        report({
                            "stage": "final_test",
                            "round": best_round,
                            **m,
                        })

        (OUT / "COMPLETE").write_text(
            str(best_round) + "\n"
        )

    finally:
        env.close()


try:
    main()

except BaseException:
    import traceback

    traceback.print_exc()

    (OUT / "FAILED").write_text(
        traceback.format_exc()
    )

    raise

finally:
    app.close()
