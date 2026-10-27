"""Evidence-grounded fact-checking with LangGraph, Gemini and Tavily."""

from .config import Config, ConfigError
from .graph import RunFailure, verify_text
from .models import Report, Verdict
from .validation import InputError

__all__ = ["Config", "ConfigError", "InputError", "Report", "RunFailure", "Verdict", "verify_text"]
__version__ = "0.1.0"
