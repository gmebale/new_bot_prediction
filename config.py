import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("FOOTBALL_DATA_API_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")