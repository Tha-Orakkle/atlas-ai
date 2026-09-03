import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

# Reliability Configurations
RETRY_MAX_DELAY = 30.0
RETRY_BASE_DELAY = 1.0
RETRY_MAX_ATTEMPTS = 3
