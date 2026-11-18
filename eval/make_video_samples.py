"""Synthetic test videos for the video module (all content generated here; no third-party media).

    python eval/make_video_samples.py            # writes eval/video_samples/*.mp4 + manifest.json

Speech is synthesised offline with the Windows SAPI voices (System.Speech), so
generating the spoken cases needs Windows; ``write_video`` itself is portable
and is also used by the offline tests (silence, tones, no audio, captions).

Each case documents what the video contains and what the extraction should
produce; eval/run_video_eval.py measures extraction, de-duplication and
provenance against these expectations.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np

OUT = Path(__file__).parent / "video_samples"
RATE = 16000


@dataclass
class Caption:
    start: float
    end: float
    text: str
    position: str = "bottom"  # bottom | center | top


@dataclass
class Speech:
    start: float
    text: str


@dataclass
class Case:
    name: str
    duration: float
    description: str
    captions: list[Caption] = field(default_factory=list)
    speech: list[Speech] = field(default_factory=list)
    audio: str = "speech"  # speech | silence | none | noise
    # expectations
    expect_merged_units: int = 0  # speech segments identical to on-screen text
    expect_distinct_screen: Optional[int] = None  # distinct on-screen observations
    expect_claims: list[dict] = field(default_factory=list)  # for the optional live fact-check
    corrupt: bool = False
    size: tuple[int, int] = (640, 360)
    font_px: Optional[int] = None
    noise: float = 0.05  # noise amplitude for audio="noise"


def _font(size: int):
    from PIL import ImageFont

    for name in ("arialbd.ttf", "arial.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_frame(size: tuple[int, int], captions: list[Caption], bg=(30, 40, 60), font_px: Optional[int] = None):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    w, h = size
    font = _font(font_px or max(18, h // 14))
    for c in captions:
        lines = c.text.split("\n")
        line_h = font.size + 8
        y0 = {"bottom": h - 30 - line_h * len(lines), "center": (h - line_h * len(lines)) // 2, "top": 30}[c.position]
        for i, line in enumerate(lines):
            tw = d.textlength(line, font=font)
            d.text(((w - tw) / 2, y0 + i * line_h), line, font=font, fill="white", stroke_width=2, stroke_fill="black")
    return im


def write_video(path: Path, *, duration: float, captions: list[Caption] = (), audio: Optional[np.ndarray] = None,
                fps: int = 10, size: tuple[int, int] = (640, 360), codec: str = "libx264",
                audio_codec: str = "aac", backgrounds: list[tuple[float, float, tuple]] = (),
                font_px: Optional[int] = None) -> Path:
    """Encode a video with PyAV. ``audio`` is 16 kHz mono float32 in [-1, 1], or None for no audio stream.

    ``backgrounds`` = [(start, end, (r, g, b))] overrides the background colour (the offline tests'
    fake OCR engine identifies frames by colour).
    """
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as c:
        vs = c.add_stream(codec, rate=fps)
        vs.width, vs.height = size
        vs.pix_fmt = "yuv420p"
        if audio is not None:
            a_s = c.add_stream(audio_codec, rate=RATE)
            a_s.layout = "mono"
        cache: dict[tuple, object] = {}
        for i in range(int(round(duration * fps))):
            t = i / fps
            active = tuple(cc for cc in captions if cc.start <= t < cc.end)
            bg = next((rgb for a, b, rgb in backgrounds if a <= t < b), (30, 40, 60))
            key = (tuple((cc.text, cc.position) for cc in active), bg)
            if key not in cache:
                cache[key] = render_frame(size, list(active), bg, font_px)
            frame = av.VideoFrame.from_image(cache[key])
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for pkt in vs.encode(frame):
                c.mux(pkt)
        for pkt in vs.encode():
            c.mux(pkt)
        if audio is not None:
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            step = 1024
            for i in range(0, len(pcm), step):
                chunk = pcm[i:i + step]
                af = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
                af.sample_rate = RATE
                af.pts = i
                af.time_base = Fraction(1, RATE)
                for pkt in a_s.encode(af):
                    c.mux(pkt)
            for pkt in a_s.encode():
                c.mux(pkt)
    return path


def tts(text: str, voice: str = "Microsoft David Desktop", rate: int = 0) -> np.ndarray:
    """Offline speech synthesis with Windows SAPI -> 16 kHz mono float32."""
    if sys.platform != "win32":
        raise RuntimeError("speech synthesis for the samples uses Windows SAPI")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "tts.wav"
        safe = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            f"$s.SelectVoice('{voice}'); $s.Rate = {rate};"
            "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
            "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono);"
            f"$s.SetOutputToWaveFile('{wav}', $fmt); $s.Speak('{safe}'); $s.Dispose()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True)
        with wave.open(str(wav), "rb") as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def build_audio(case: Case, seed: int = 0) -> Optional[np.ndarray]:
    n = int(case.duration * RATE)
    if case.audio == "none":
        return None
    track = np.zeros(n, dtype=np.float32)
    if case.audio == "noise":
        track += np.random.default_rng(seed).normal(0, case.noise, n).astype(np.float32)
    for s in case.speech:
        clip = tts(s.text) * 0.8
        a = int(s.start * RATE)
        track[a:a + len(clip)] += clip[: max(0, n - a)]
    return track


CASES = [
    Case("speech_only", 9.0, "Speech with no visible text.",
         speech=[Speech(0.5, "The Eiffel Tower was completed in 1889."),
                 Speech(4.5, "Mount Everest is the highest mountain on Earth.")],
         expect_distinct_screen=0,
         expect_claims=[{"needle": "eiffel", "verdict": "SUPPORTED"}, {"needle": "everest", "verdict": "SUPPORTED"}]),
    Case("text_only_no_audio", 8.0, "On-screen text, and no audio stream at all.", audio="none",
         captions=[Caption(0, 8, "THE GREAT WALL OF CHINA IS\nVISIBLE FROM THE MOON", "center")],
         expect_distinct_screen=1, expect_claims=[{"needle": "great wall", "verdict": "REFUTED"}]),
    Case("same_claim_both", 7.0, "The same sentence is spoken and shown as a subtitle (should be checked once).",
         speech=[Speech(0.5, "Water boils at 100 degrees Celsius at sea level.")],
         captions=[Caption(0.3, 5.5, "Water boils at 100 degrees Celsius at sea level.")],
         expect_merged_units=1, expect_distinct_screen=1,
         expect_claims=[{"needle": "boils", "verdict": "SUPPORTED", "count": 1}]),
    Case("contradiction", 8.0, "Speech says one thing, the on-screen text the opposite (both must be kept).",
         speech=[Speech(0.5, "The Great Wall of China is not visible from the Moon with the naked eye.")],
         captions=[Caption(0.3, 7.5, "The Great Wall of China is visible\nfrom the Moon with the naked eye", "top")],
         expect_distinct_screen=1,
         expect_claims=[{"needle": "not visible", "verdict": "SUPPORTED"}, {"needle": "is visible", "verdict": "REFUTED"}]),
    Case("different_numbers", 8.0, "Similar wording, different years in speech and on screen (both must be kept).",
         speech=[Speech(0.5, "The Eiffel Tower was completed in 1889.")],
         captions=[Caption(0.3, 7.5, "The Eiffel Tower was completed in 1887.")],
         expect_distinct_screen=1,
         expect_claims=[{"needle": "1889", "verdict": "SUPPORTED"}, {"needle": "1887", "verdict": "REFUTED"}]),
    Case("timeline", 14.0, "Different claims at different times, in both modalities.",
         speech=[Speech(0.5, "The Pacific is the largest ocean on Earth."),
                 Speech(8.0, "Light travels faster than sound.")],
         captions=[Caption(4.0, 7.5, "Canberra is the capital of Australia", "center"),
                   Caption(11.0, 14.0, "Bats are blind", "center")],
         expect_distinct_screen=2),
    Case("repeated_subtitles", 12.0, "One subtitle persists across many frames, then reappears later.",
         speech=[], audio="silence",
         captions=[Caption(0, 5, "The Amazon is the longest river in South America"),
                   Caption(5, 7, "Fact check demo", "center"),
                   Caption(7, 12, "The Amazon is the longest river in South America")],
         expect_distinct_screen=2),
    Case("silence_with_track", 6.0, "An audio track containing only silence, and no text.", audio="silence",
         expect_distinct_screen=0),
    Case("noise_only", 6.0, "Background noise only; no speech, no text.", audio="noise", expect_distinct_screen=0),
    Case("noisy_speech", 8.0, "Speech under heavy background noise (low-quality transcription).", audio="noise",
         noise=0.25, speech=[Speech(0.5, "The Moon orbits the Earth about once every 27 days.")],
         expect_distinct_screen=0),
    Case("tiny_text", 6.0, "Small caption in a low-resolution (256x144) video (low-quality OCR).", audio="none",
         size=(256, 144), font_px=9, captions=[Caption(0, 6, "The Sahara is the largest hot desert")]),
]


def make_corrupt(src: Path, dst: Path) -> Path:
    data = bytearray(src.read_bytes())
    for i in range(64, len(data), 97):  # scramble the stream data but keep the size
        data[i] ^= 0x5A
    dst.write_bytes(bytes(data[: len(data) // 3]))
    return dst


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case in CASES:
        path = OUT / f"{case.name}.mp4"
        write_video(path, duration=case.duration, captions=case.captions, audio=build_audio(case), size=case.size,
                    font_px=case.font_px)
        manifest.append({**asdict(case), "file": path.name})
        print(f"wrote {path}")
    bad = make_corrupt(OUT / "speech_only.mp4", OUT / "corrupt.mp4")
    manifest.append({"name": "corrupt", "file": bad.name, "description": "Truncated, scrambled MP4.", "corrupt": True})
    print(f"wrote {bad}")
    (OUT / "notavideo.mp4").write_text("this is a text file with a video extension", encoding="utf-8")
    manifest.append({"name": "notavideo", "file": "notavideo.mp4", "description": "Text file named .mp4.",
                     "corrupt": True})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
