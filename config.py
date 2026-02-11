import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
GEMINI_SUB_AGENT_MODEL = os.getenv("GEMINI_SUB_AGENT_MODEL", "gemini-3-flash-preview")
GEMINI_PLANNER_MODEL = os.getenv("GEMINI_PLANNER_MODEL", GEMINI_MODEL)
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
DATA_BACKEND = os.getenv("DATA_BACKEND", "cdf")  # "cdf" or "hapi"
CATALOG_SEARCH_METHOD = os.getenv("CATALOG_SEARCH_METHOD", "semantic")  # "semantic" or "substring"
DEFAULT_TIME_RANGE = "2024-01-01 to 2024-01-07"
