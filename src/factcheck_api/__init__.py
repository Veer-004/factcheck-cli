"""REST API (FastAPI) for the four fact-checking modules.

The API is a thin layer: every request is handed to the existing module function
(``verify_text``, ``verify_image``, ``verify_video``, ``verify_audio``), which stays
the source of truth for claims, evidence and verdicts. Run it with::

    uvicorn factcheck_api.app:app --reload
"""
