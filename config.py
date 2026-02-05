import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AUTOPLOT_JAR = os.getenv("AUTOPLOT_JAR", "~/autoplot/autoplot.jar")
JAVA_HOME = os.getenv("JAVA_HOME")
DEFAULT_TIME_RANGE = "2024-01-01 to 2024-01-07"
