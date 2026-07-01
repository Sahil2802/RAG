"""LangSmith / tracing setup.

Importing this module loads the backend `.env` so LANGSMITH_* variables are
present in the environment. LangChain components (e.g. ChatOllama) then emit
traces automatically when LANGSMITH_TRACING=true. Import it before any
LangChain code at each entry point.
"""
from pathlib import Path

from dotenv import load_dotenv

# Load backend/.env regardless of the current working directory.
load_dotenv(Path(__file__).resolve().parent / ".env")
