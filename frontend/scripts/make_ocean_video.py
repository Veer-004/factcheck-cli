"""Render the site's background video: an original, seamlessly looping underwater scene.

    .venv\\Scripts\\python.exe frontend\\scripts\\make_ocean_video.py

Procedural (numpy) and fully offline: drifting light caustics near the surface, slow light shafts,
rising particles and a depth gradient in the Deep Ocean palette. Every motion term is periodic in
the loop length, so the last frame flows into the first. Writes public/media/ocean-loop.mp4 (H.264,
faststart) and public/media/ocean-poster.jpg. Requires the project's video extra (PyAV, numpy, Pillow).
"""

from __future__ import annotations

import math
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageFilter

W, H = 960, 540
FPS = 24
SECONDS = 12
FRAMES = FPS * SECONDS
OUT = Path(__file__).resolve().parent.parent / "public" / "media"

rng = np.random.default_rng(7)
ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
X, Y = xs / W, ys / H  # 0..1

# depth gradient: deep teal near the surface -> abyss below
TOP = np.array([12, 46, 56], np.float32)
BOTTOM = np.array([3, 12, 17], np.float32)
depth = Y[..., None] ** 0.8
base = TOP * (1 - depth) + BOTTOM * depth

# light shafts from a point above the frame
ANG = np.arctan2(X - 0.42, Y + 0.35)
SHAFT_FADE = np.clip(1.15 - Y * 1.25, 0, 1) ** 2

# vignette
VIG = np.clip(1.25 - np.sqrt((X - 0.5) ** 2 * 1.1 + (Y - 0.45) ** 2 * 1.6) * 1.5, 0, 1)

# particles: integer cycles per loop so they wrap seamlessly
N = 140
P_X = rng.random(N)
P_Y = rng.random(N)
P_CYC = rng.integers(1, 3, N)  # vertical passes per loop
P_SIZE = rng.uniform(0.6, 2.0, N)
P_BRIGHT = rng.uniform(0.25, 0.8, N)
P_SWAY = rng.uniform(0.002, 0.01, N)

TEAL = np.array([0, 213, 192], np.float32)
ICE = np.array([229, 250, 255], np.float32)


def caustics(phi: float, scale: float, speed: int) -> np.ndarray:
    """Sharp, web-like light pattern from interfering waves (periodic in phi)."""
    x = X * scale * 6.0
    y = Y * scale * 3.4
    a = np.sin(x + np.sin(y * 1.3 + speed * phi) * 1.6 + phi)
    b = np.sin(y * 1.7 + np.sin(x * 0.8 - speed * phi) * 1.8 - 2 * phi)
    c = np.sin((x + y) * 0.9 + np.cos(x * 0.5 + phi) * 1.4 + phi)
    v = (a + b + c) / 3.0
    return np.exp(-(v ** 2) * 18.0)


def frame(i: int) -> np.ndarray:
    phi = 2 * math.pi * i / FRAMES
    surface = np.clip(1.0 - Y * 1.35, 0, 1) ** 1.6
    caus = (0.65 * caustics(phi, 1.0, 1) + 0.35 * caustics(phi + 1.3, 1.9, 2)) * surface

    shafts = np.zeros_like(X)
    for m, n, w in ((9, 1, 0.5), (14, -2, 0.3), (5, 1, 0.35)):
        shafts += w * (0.5 + 0.5 * np.cos(ANG * m + n * phi + m))
    shafts = (shafts / 1.15) ** 3 * SHAFT_FADE

    img = base.copy()
    img += TEAL * (caus * 0.46)[..., None]
    img += ICE * (shafts * 0.09)[..., None]

    # particles (soft dots)
    t = i / FRAMES
    px = (P_X + P_SWAY * np.sin(2 * math.pi * (t * 2 + P_X * 3))) % 1.0
    py = (P_Y - t * P_CYC) % 1.0
    for k in range(N):
        cx, cy, s = px[k] * W, py[k] * H, P_SIZE[k]
        x0, x1 = int(max(0, cx - 4 * s)), int(min(W, cx + 4 * s + 1))
        y0, y1 = int(max(0, cy - 4 * s)), int(min(H, cy + 4 * s + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        gx = xs[y0:y1, x0:x1] - cx
        gy = ys[y0:y1, x0:x1] - cy
        g = np.exp(-(gx * gx + gy * gy) / (2 * s * s)) * P_BRIGHT[k] * (0.4 + 0.6 * (1 - py[k]))
        img[y0:y1, x0:x1] += ICE * (g * 0.55)[..., None]

    img *= VIG[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ocean-loop.mp4"
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as out:
        stream = out.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.options = {"crf": "27", "preset": "slow", "tune": "grain"}
        for i in range(FRAMES):
            rgb = frame(i)
            # a slight blur keeps the light soft and the file small
            rgb = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(0.8)))
            if i == 0:
                Image.fromarray(rgb).save(OUT / "ocean-poster.jpg", quality=82)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB, {FRAMES} frames, {SECONDS} s loop)")


if __name__ == "__main__":
    main()
