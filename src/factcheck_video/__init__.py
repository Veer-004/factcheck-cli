"""Video fact-checking: speech (faster-whisper) and on-screen text (factcheck_ocr) -> the shared pipeline.

Heavy dependencies (PyAV, faster-whisper, RapidOCR) are imported lazily, so
importing this package costs nothing for the text and image commands.
"""

__all__ = ["main"]


def main() -> int:
    from .cli import main as _main

    return _main()
