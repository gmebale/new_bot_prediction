import requests
import pandas as pd
import datetime
import os

OPENWEATHER_API_KEY = "1748866d68a1ed3307fe7fd70c1b3040"  # Replace with your actual API key
OPENWEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

def fetch_weather_data(city_name, date):
    """
    Fetch weather data for a given city and date using OpenWeatherMap API.
    Note: OpenWeatherMap free API provides current weather and forecast, historical data requires paid plan.
    For demonstration, this function fetches current weather only.
    """
    # Mapping of team names to city names for weather API
    team_to_city = {
        "Manchester City FC": "Manchester",
        "Everton FC": "Liverpool",
        "Newcastle United FC": "Newcastle",
        "Southampton FC": "Southampton",
        "Arsenal FC": "London",
        "Aston Villa FC": "Birmingham",
        "Fulham FC": "London",
        "AFC Bournemouth": "Bournemouth",
        "Brentford FC": "London",
        "Crystal Palace FC": "London",
        "Manchester United FC": "Manchester",
        # Add other teams as needed
    }

    city = team_to_city.get(city_name, city_name)  # Default to city_name if no mapping found

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(OPENWEATHER_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        weather = {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "weather_main": data["weather"][0]["main"],
            "weather_description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"]
        }
        return weather
    except Exception as e:
        print(f"Error fetching weather data for {city_name} on {date}: {e}")
        return None
import os
from config import API_TOKEN
from datetime import datetime, timedelta

BASE_URL = "https://api.football-data.org/v4"

# Mapping of competition names to their codes in the API
COMPETITION_CODES = {
    "Premier League": "PL",
    "Ligue 1": "FL1",
    "Bundesliga": "BL1",
    "Serie A": "SA",
    "Eredivisie": "DED",
    "Primeira Liga": "PPL",
    "La Liga": "PD"
}

def fetch_matches(competitions=None, seasons=None, match_status="SCHEDULED", days_range=1):
    """
    Récupère les matchs à venir pour une ou plusieurs compétitions, filtrés sur une plage de dates autour de la date du jour.
    Tente plusieurs saisons jusqu'à trouver des matchs.

    :param competitions: Liste des codes de compétitions à récupérer. Si None, utilise toutes les compétitions définies dans COMPETITION_CODES par défaut.
    :param seasons: Liste des saisons à tester (année de début). Si None, utilise [2024, 2023, 2025].
    :param match_status: Statut des matchs à récupérer ("SCHEDULED" par défaut pour matchs à venir).
    :param days_range: Nombre de jours avant et après la date du jour à inclure dans le filtrage.
    :return: DataFrame pandas avec les matchs récupérés.
    """
    if competitions is None:
        competitions = list(COMPETITION_CODES.values())
    if seasons is None:
        seasons = [2024, 2023, 2025]

    all_matches = []
    headers = {"X-Auth-Token": API_TOKEN}
    today = datetime.utcnow().date()
    start_date = today - timedelta(days=days_range)
    end_date = today + timedelta(days=days_range)

    for competition in competitions:
        matches_found = False
        for season in seasons:
            url = f"{BASE_URL}/competitions/{competition}/matches?season={season}"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            print(f"Competition {competition}, saison {season}: {len(data.get('matches', []))} matchs retournés par l'API")
            if len(data.get('matches', [])) > 0:
                print("Exemples de dates de matchs récupérés :")
                for m in data.get('matches', [])[:5]:
                    print(f" - {m['utcDate']} (status: {m['status']})")

            count_matches = 0
            for match in data.get("matches", []):
                match_date = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00")).date()
                if match["status"] == match_status and start_date <= match_date <= end_date:
                    all_matches.append({
                        "date": match["utcDate"],
                        "competition": competition,
                        "home_team": match["homeTeam"]["name"],
                        "away_team": match["awayTeam"]["name"],
                        "home_score": match["score"]["fullTime"]["home"],
                        "away_score": match["score"]["fullTime"]["away"],
                    })
                    count_matches += 1
            print(f"Competition {competition}, saison {season}: {count_matches} matchs récupérés entre {start_date} et {end_date}")

            if count_matches > 0:
                matches_found = True
                break  # Stop searching other seasons for this competition

        if not matches_found:
            print(f"Aucun match trouvé pour la compétition {competition} dans les saisons testées.")

    df = pd.DataFrame(all_matches)
    df.to_csv("historical_matches.csv", index=False)
    return df

def fetch_matches_by_matchday(competition, matchday):
    """
    Récupère les matchs d'une compétition pour un matchday donné.

    :param competition: Code de la compétition (ex: "PD" pour La Liga).
    :param matchday: Numéro du matchday (journée).
    :return: DataFrame pandas avec les matchs récupérés.
    """
    headers = {"X-Auth-Token": API_TOKEN}
    url = f"{BASE_URL}/competitions/{competition}/matches?matchday={matchday}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    matches = []
    for match in data.get("matches", []):
        matches.append({
            "date": match["utcDate"],
            "competition": competition,
            "home_team": match["homeTeam"]["name"],
            "away_team": match["awayTeam"]["name"],
            "home_score": match["score"]["fullTime"]["home"],
            "away_score": match["score"]["fullTime"]["away"],
            "status": match["status"]
        })

    df = pd.DataFrame(matches)
    return df
