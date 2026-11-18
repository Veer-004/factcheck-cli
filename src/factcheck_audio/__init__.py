"""Audio fact-checking: English speech (faster-whisper) -> the shared fact-checking pipeline.

Heavy dependencies (PyAV, faster-whisper) are imported lazily, so importing
this package costs nothing for the other commands.
"""

__all__ = ["main"]


def main() -> int:
    from .cli import main as _main

    return _main()
