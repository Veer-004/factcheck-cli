"""Render synthetic OCR evaluation images with known text and reading order.

    python eval/make_ocr_samples.py            # writes eval/ocr_samples/*.png + expected.jsonl

Each sample stores its expected *lines in reading order*. The images use Windows
system fonts (Arial, Times New Roman, Impact, Comic Sans, Segoe Script); a font
that is missing is replaced by Pillow's default font and noted. These are synthetic:
they test layout handling and stylised fonts, not real photos or compression
artefacts, so also try your own screenshots and memes.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).parent / "ocr_samples"
FONTS = Path("C:/Windows/Fonts")


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(FONTS / name), size)
    except OSError:
        print(f"note: font {name} not found, using Pillow's default")
        return ImageFont.load_default(size)


def lines_at(draw, lines, x, y, f, fill="black", step=None, **kw):
    step = step or int(f.size * 1.35)
    for i, text in enumerate(lines):
        draw.text((x, y + i * step), text, font=f, fill=fill, **kw)
    return y + len(lines) * step


def two_column():
    im = Image.new("RGB", (1200, 900), "white")
    d = ImageDraw.Draw(im)
    title = ["Library Extends Weekend Opening Hours"]
    left = ["The central library will open on", "Sundays from next month, the city", "council confirmed on Tuesday after", "a public consultation."]
    left2 = ["More than four thousand residents", "responded to the survey."]
    right = ["Staff numbers will rise by twelve", "to cover the extra shifts, and the", "reading rooms will stay open until", "eight in the evening."]
    right2 = ["The change costs about two hundred", "thousand euros per year."]
    lines_at(d, title, 60, 50, font("timesbd.ttf", 48))
    f = font("times.ttf", 30)
    y = lines_at(d, left, 60, 170, f)
    lines_at(d, left2, 60, y + 40, f)
    y = lines_at(d, right, 640, 170, f)
    lines_at(d, right2, 640, y + 40, f)
    return im, title + left + left2 + right + right2


def three_column():
    im = Image.new("RGB", (1500, 700), (250, 248, 240))
    d = ImageDraw.Draw(im)
    f = font("georgia.ttf", 26)
    cols = [
        ["Rainfall in March was", "twice the average for", "the region this year."],
        ["Farmers expect a late", "harvest because fields", "stayed wet for weeks."],
        ["Reservoirs are now at", "ninety percent of their", "capacity, officials say."],
    ]
    lines_at(d, ["WEATHER REPORT"], 60, 40, font("arialbd.ttf", 44))
    for i, col in enumerate(cols):
        lines_at(d, col, 60 + i * 480, 150, f)
    return im, ["WEATHER REPORT"] + [l for c in cols for l in c]


def meme(seed=3):
    rng = random.Random(seed)
    im = Image.new("RGB", (900, 700))
    px = im.load()
    for y in range(700):  # busy photo-like background: gradient plus noise blobs
        for x in range(0, 900):
            px[x, y] = (int(90 + 80 * x / 900) + rng.randint(-25, 25), int(120 + 60 * y / 700) + rng.randint(-25, 25), 140 + rng.randint(-30, 30))
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    d = ImageDraw.Draw(im)
    for _ in range(25):
        x, y, r = rng.randint(0, 900), rng.randint(0, 700), rng.randint(20, 90)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    f = font("impact.ttf", 72)
    top, bottom = "WHEN THE TESTS PASS", "ON THE FIRST TRY"
    for text, y in ((top, 30), (bottom, 580)):
        w = d.textlength(text, font=f)
        d.text(((900 - w) / 2, y), text, font=f, fill="white", stroke_width=4, stroke_fill="black")
    return im, [top, bottom]


def poster():
    im = Image.new("RGB", (1000, 1300), (20, 40, 90))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 900, 1000, 1300), fill=(240, 180, 40))
    head = ["SUMMER", "JAZZ NIGHT"]
    f = font("arialbd.ttf", 120)
    for i, t in enumerate(head):
        w = d.textlength(t, font=f)
        d.text(((1000 - w) / 2, 80 + i * 150), t, font=f, fill=(255, 220, 120))
    sub = "Live music under the stars"
    fs = font("segoepr.ttf", 48)
    d.text(((1000 - d.textlength(sub, font=fs)) / 2, 450), sub, font=fs, fill="white")
    fd = font("arial.ttf", 44)
    d.text((80, 960), "Saturday 12 July", font=fd, fill=(20, 40, 90))
    d.text((620, 960), "Riverside Park", font=fd, fill=(20, 40, 90))
    d.text((80, 1040), "Doors open 7 pm", font=fd, fill=(20, 40, 90))
    d.text((620, 1040), "Free entry", font=fd, fill=(20, 40, 90))
    return im, head + [sub, "Saturday 12 July", "Riverside Park", "Doors open 7 pm", "Free entry"]


def chat():
    im = Image.new("RGB", (750, 1000), (236, 229, 221))
    d = ImageDraw.Draw(im)
    f = font("segoeui.ttf", 30)
    msgs = [
        ("l", "Are you coming to the match?"),
        ("r", "Yes, leaving in ten minutes"),
        ("l", "Great, bring the tickets"),
        ("r", "Already in my bag"),
        ("l", "See you at gate four"),
    ]
    d.rectangle((0, 0, 750, 90), fill=(7, 94, 84))
    d.text((30, 25), "Sam", font=font("segoeuib.ttf", 34), fill="white")
    y = 130
    for side, text in msgs:
        w = d.textlength(text, font=f) + 40
        x0 = 25 if side == "l" else 750 - 25 - w
        d.rounded_rectangle((x0, y, x0 + w, y + 64), 16, fill="white" if side == "l" else (220, 248, 198))
        d.text((x0 + 20, y + 12), text, font=f, fill="black")
        y += 110
    return im, ["Sam"] + [t for _, t in msgs]


def low_contrast():
    im = Image.new("RGB", (1000, 450), (200, 190, 170))
    d = ImageDraw.Draw(im)
    rng = random.Random(5)
    for _ in range(4000):  # paper texture
        x, y = rng.randint(0, 999), rng.randint(0, 449)
        d.point((x, y), fill=(170, 160, 140))
    lines = ["Handwritten style notes are", "harder for OCR to read than", "clean printed documents."]
    lines_at(d, lines, 50, 60, font("comic.ttf", 44), fill=(110, 95, 80), step=110)
    return im.filter(ImageFilter.GaussianBlur(0.8)), lines


SAMPLES = {
    "two_column": two_column, "three_column": three_column, "meme": meme,
    "poster": poster, "chat_screenshot": chat, "low_contrast": low_contrast,
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = []
    for name, make in SAMPLES.items():
        im, expected = make()
        path = OUT / f"{name}.png"
        im.save(path)
        rows.append({"image": path.name, "lines": expected})
        print(f"wrote {path}")
    (OUT / "expected.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
