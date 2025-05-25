import os
from dotenv import load_dotenv

load_dotenv()

def main():
    token = os.getenv("FOOTBALL_DATA_API_TOKEN")
    if token:
        print("FOOTBALL_DATA_API_TOKEN is set.")
    else:
        print("FOOTBALL_DATA_API_TOKEN is NOT set.")

if __name__ == "__main__":
    main()
