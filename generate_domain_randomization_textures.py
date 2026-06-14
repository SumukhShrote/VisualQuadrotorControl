"""
generate_domain_randomization_textures.py

Creates procedural textures for domain randomization.

Run:
    python generate_domain_randomization_textures.py
"""

import os
import random
import numpy as np

from PIL import Image
from PIL import ImageFilter


OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "models",
        "textures",
        "generated",
    )
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXTURE_SIZE = 1024
NUM_TEXTURES = 100


# ============================================================
# Utilities
# ============================================================

def random_color():
    return np.array(
        [
            random.randint(30, 225),
            random.randint(30, 225),
            random.randint(30, 225),
        ],
        dtype=np.uint8,
    )


def save_texture(arr, filename):
    Image.fromarray(arr.astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR, filename)
    )


# ============================================================
# Texture Types
# ============================================================

def texture_rgb_noise(size):

    img = Image.fromarray(
        np.random.randint(
            0,
            256,
            (size, size, 3),
            dtype=np.uint8,
        )
    )

    img = img.filter(
        ImageFilter.GaussianBlur(
            random.uniform(1.0, 4.0)
        )
    )

    return np.array(img)


def texture_checkerboard(size):

    block = random.randint(16, 128)

    c1 = random_color()
    c2 = random_color()

    img = np.zeros(
        (size, size, 3),
        dtype=np.uint8,
    )

    for y in range(0, size, block):
        for x in range(0, size, block):

            if ((x // block) + (y // block)) % 2 == 0:
                img[y:y + block, x:x + block] = c1
            else:
                img[y:y + block, x:x + block] = c2

    return img


def texture_stripes(size):

    stripe_width = random.randint(8, 64)

    c1 = random_color()
    c2 = random_color()

    img = np.zeros(
        (size, size, 3),
        dtype=np.uint8,
    )

    vertical = random.choice(
        [True, False]
    )

    if vertical:

        for x in range(
            0,
            size,
            stripe_width,
        ):

            color = (
                c1
                if (x // stripe_width) % 2 == 0
                else c2
            )

            img[:, x:x + stripe_width] = color

    else:

        for y in range(
            0,
            size,
            stripe_width,
        ):

            color = (
                c1
                if (y // stripe_width) % 2 == 0
                else c2
            )

            img[y:y + stripe_width] = color

    return img


def texture_gradient(size):

    c1 = random_color().astype(np.float32)
    c2 = random_color().astype(np.float32)

    horizontal = random.choice(
        [True, False]
    )

    t = np.linspace(0, 1, size)

    if horizontal:

        grad = (
            c1[None] * (1 - t[:, None])
            + c2[None] * t[:, None]
        )

        img = np.tile(
            grad[None],
            (size, 1, 1),
        )

    else:

        grad = (
            c1[None] * (1 - t[:, None])
            + c2[None] * t[:, None]
        )

        img = np.tile(
            grad[:, None],
            (1, size, 1),
        )

    return img.astype(np.uint8)


def texture_blobs(size):

    img = Image.fromarray(
        np.random.randint(
            0,
            256,
            (size, size, 3),
            dtype=np.uint8,
        )
    )

    img = img.filter(
        ImageFilter.GaussianBlur(
            random.uniform(5, 20)
        )
    )

    return np.array(img)


def texture_smooth_noise(size):

    img = Image.fromarray(
        np.random.randint(
            0,
            256,
            (
                size // 4,
                size // 4,
                3,
            ),
            dtype=np.uint8,
        )
    )

    img = img.resize(
        (size, size),
        Image.Resampling.BICUBIC,
    )

    img = img.filter(
        ImageFilter.GaussianBlur(
            random.uniform(3, 8)
        )
    )

    return np.array(img)


def texture_rectangles(size):

    img = np.ones(
        (size, size, 3),
        dtype=np.uint8,
    ) * random.randint(
        50,
        200,
    )

    num_rects = random.randint(
        20,
        100,
    )

    for _ in range(num_rects):

        x1 = random.randint(
            0,
            size - 1,
        )

        y1 = random.randint(
            0,
            size - 1,
        )

        x2 = random.randint(
            x1,
            min(size, x1 + size // 3),
        )

        y2 = random.randint(
            y1,
            min(size, y1 + size // 3),
        )

        img[y1:y2, x1:x2] = random_color()

    return img


def texture_circles(size):

    img = np.ones(
        (size, size, 3),
        dtype=np.uint8,
    ) * random.randint(
        50,
        200,
    )

    yy, xx = np.mgrid[
        0:size,
        0:size,
    ]

    for _ in range(
        random.randint(20, 80)
    ):

        cx = random.randint(
            0,
            size,
        )

        cy = random.randint(
            0,
            size,
        )

        r = random.randint(
            size // 50,
            size // 10,
        )

        mask = (
            (xx - cx) ** 2
            +
            (yy - cy) ** 2
        ) < r ** 2

        img[mask] = random_color()

    return img


def texture_radial(size):

    yy, xx = np.mgrid[
        0:size,
        0:size,
    ]

    cx = random.randint(
        size // 4,
        3 * size // 4,
    )

    cy = random.randint(
        size // 4,
        3 * size // 4,
    )

    dist = np.sqrt(
        (xx - cx) ** 2 +
        (yy - cy) ** 2
    )

    dist /= dist.max()

    c1 = random_color().astype(
        np.float32
    )

    c2 = random_color().astype(
        np.float32
    )

    img = (
        c1 * (1 - dist[..., None])
        +
        c2 * dist[..., None]
    )

    return img.astype(np.uint8)


def texture_marble(size):

    x = np.linspace(
        0,
        8 * np.pi,
        size,
    )

    y = np.linspace(
        0,
        8 * np.pi,
        size,
    )

    xx, yy = np.meshgrid(
        x,
        y,
    )

    phase = random.uniform(
        0,
        2 * np.pi,
    )

    pattern = np.sin(
        xx
        + yy
        + phase
        + np.random.normal(
            0,
            0.5,
            (size, size),
        )
    )

    pattern = (
        (pattern - pattern.min())
        /
        (pattern.max() - pattern.min())
    )

    c1 = random_color().astype(
        np.float32
    )

    c2 = random_color().astype(
        np.float32
    )

    img = (
        c1 * (1 - pattern[..., None])
        +
        c2 * pattern[..., None]
    )

    img = Image.fromarray(
        img.astype(np.uint8)
    )

    img = img.filter(
        ImageFilter.GaussianBlur(1.5)
    )

    return np.array(img)


def texture_voronoi(size):

    n_points = random.randint(
        10,
        40,
    )

    points = np.random.randint(
        0,
        size,
        (n_points, 2),
    )

    colors = np.array(
        [
            random_color()
            for _ in range(n_points)
        ]
    )

    yy, xx = np.mgrid[
        0:size,
        0:size,
    ]

    pixels = np.stack(
        [
            xx.ravel(),
            yy.ravel(),
        ],
        axis=1,
    )

    dist = np.sum(
        (
            pixels[:, None, :]
            -
            points[None]
        ) ** 2,
        axis=2,
    )

    nearest = np.argmin(
        dist,
        axis=1,
    )

    img = colors[
        nearest
    ].reshape(
        size,
        size,
        3,
    )

    return img.astype(np.uint8)


# ============================================================
# Main
# ============================================================

GENERATORS = [

    texture_checkerboard,
    texture_stripes,
    texture_gradient,

    texture_blobs,
    texture_circles,

    texture_rectangles,

    texture_radial,
    texture_marble,

    texture_smooth_noise,
    texture_voronoi,

    texture_rgb_noise,
]

print(
    f"\nGenerating {NUM_TEXTURES} textures..."
)

print(
    f"Output: {OUTPUT_DIR}\n"
)

for i in range(NUM_TEXTURES):

    generator = random.choice(
        GENERATORS
    )

    img = generator(
        TEXTURE_SIZE
    )

    save_texture(
        img,
        f"texture_{i:03d}.png",
    )

    print(
        f"[{i+1:03d}/{NUM_TEXTURES}] "
        f"{generator.__name__}"
    )

print("\nDone.")