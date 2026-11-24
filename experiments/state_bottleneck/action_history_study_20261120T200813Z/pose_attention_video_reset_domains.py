
import ast
import inspect
import textwrap
import math
import torch

RESET = {
    "enabled": False,
    "scale": 1.0,
}

DEV = None

def log(*args):
    print(*args, flush=True)

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

def configure_video_domain(cfg, domain, level):
    global DEV

    DEV = torch.device(cfg.sim.device)

    cfg.forced_curriculum_level = float(level)

    RESET["enabled"] = domain == "expanded"
    RESET["scale"] = 1.0

    if domain == "expanded":
        from envs.aruco_hover.quadrotor_aruco_hover_env import (
            ArucoHoverTeacherEnv,
        )

        patch_reset(ArucoHoverTeacherEnv)

    elif domain != "original":
        raise ValueError(f"Unknown reset domain: {domain}")

    log("[VIDEO DOMAIN]", domain)
    log("[VIDEO CURRICULUM]", level)
    log("[VIDEO RESET SCALE]", RESET["scale"])
    log("[VIDEO DEVICE]", DEV)

