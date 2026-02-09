import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
GEMINI_SUB_AGENT_MODEL = os.getenv("GEMINI_SUB_AGENT_MODEL", "gemini-3-flash-preview")
GEMINI_PLANNER_MODEL = os.getenv("GEMINI_PLANNER_MODEL", GEMINI_MODEL)
DEFAULT_TIME_RANGE = "2024-01-01 to 2024-01-07"
