"""Local image-to-text (OCR) with layout-aware reading order.

Separate from the ``factcheck`` package: it shares no code with the fact-checking
pipeline and importing it loads no LangGraph, Gemini or Tavily code.
"""

from .engines import EngineFailure, EngineUnavailable, create_engine
from .image_io import ImageInputError
from .models import Block, Line, PageResult, TextBox
from .pipeline import ocr_image

__all__ = [
    "Block", "EngineFailure", "EngineUnavailable", "ImageInputError", "Line", "PageResult", "TextBox",
    "create_engine", "ocr_image",
]
