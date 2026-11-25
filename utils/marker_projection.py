"""
Calibrated analytic marker projection.

The implementation is extracted from the successful
relative-pose supervision experiment.

Marker side length: 0.10 m.
Nominal marker-to-hover distance: 1.47 m.
"""

import torch

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
