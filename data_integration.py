import requests
import pandas as pd
import os
import json
from config import API_TOKEN

# Constants for data sources
FOOTBALL_DATA_API_URL = "https://api.football-data.org/v4"
FOOTBALL_DATA_API_TOKEN = API_TOKEN  # imported from config

FOOTBALL_DATASETS_CSV_PATH = "data/football_datasets"  # folder containing CSV files
OPENFOOTBALL_JSON_BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"

def fetch_football_data_api_competition_standings(competition="PL", season=2024):
    """
    Fetch standings (classement) from football-data.org API for a given competition.
    According to the official docs, the season parameter is not supported for standings endpoint.
    We fetch the current standings only.
    """
    url = f"{FOOTBALL_DATA_API_URL}/competitions/{competition}/standings"
    headers = {"X-Auth-Token": FOOTBALL_DATA_API_TOKEN}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    # Extract standings table
    standings = []
    for standing in data.get("standings", []):
        if standing.get("type") == "TOTAL":
            for team in standing.get("table", []):
                standings.append({
                    "position": team.get("position"),
                    "team_id": team.get("team", {}).get("id"),
                    "team_name": team.get("team", {}).get("name"),
                    "played_games": team.get("playedGames"),
                    "won": team.get("won"),
                    "draw": team.get("draw"),
                    "lost": team.get("lost"),
                    "points": team.get("points"),
                    "goals_for": team.get("goalsFor"),
                    "goals_against": team.get("goalsAgainst"),
                    "goal_difference": team.get("goalDifference"),
                })
    df = pd.DataFrame(standings)
    return df

def load_football_datasets_csv(competition="E0"):
    """
    Load historical match data CSV from football-datasets for a given competition code.
    The CSV files should be downloaded and placed in FOOTBALL_DATASETS_CSV_PATH.
    """
    filename = f"{competition}.csv"
    filepath = os.path.join(FOOTBALL_DATASETS_CSV_PATH, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    df = pd.read_csv(filepath)
    return df

def fetch_openfootball_json_season(country="england", season="2023-24"):
    """
    Fetch season match data JSON from OpenFootball GitHub repository.
    country: e.g. "england", "spain", "germany"
    season: e.g. "2023-24"
    """
    url = f"{OPENFOOTBALL_JSON_BASE_URL}/{country}/{season}.json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    # Flatten matches data
    matches = []
    for round in data.get("rounds", []):
        round_name = round.get("name")
        for match in round.get("matches", []):
            matches.append({
                "round": round_name,
                "date": match.get("date"),
                "team1": match.get("team1", {}).get("name"),
                "team2": match.get("team2", {}).get("name"),
                "score1": match.get("score1"),
                "score2": match.get("score2"),
            })
    df = pd.DataFrame(matches)
    return df

def merge_datasets(*dfs):
    """
    Merge multiple dataframes on common keys (e.g. team names, dates).
    This function should be customized based on the actual data structure.
    """
    from functools import reduce
    df_merged = reduce(lambda left, right: pd.merge(left, right, how='outer', on=['team_name']), dfs)
    return df_merged

# Additional functions to process and enrich data can be added here.

def analyze_recent_form_from_csv(competition="E0", n_last_matches=5):
    """
    Analyze recent form (W/D/L) of teams from football-data.co.uk CSV files.
    Assumes CSV files are downloaded in FOOTBALL_DATASETS_CSV_PATH.
    Returns a DataFrame with team names and their recent form stats.
    """
    import os
    import pandas as pd

    filepath = os.path.join(FOOTBALL_DATASETS_CSV_PATH, f"{competition}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Normalize column names for home and away teams and results
    home_col = "HomeTeam" if "HomeTeam" in df.columns else None
    away_col = "AwayTeam" if "AwayTeam" in df.columns else None
    fthg_col = "FTHG" if "FTHG" in df.columns else None
    ftag_col = "FTAG" if "FTAG" in df.columns else None
    ftr_col = "FTR" if "FTR" in df.columns else None

    if None in [home_col, away_col, fthg_col, ftag_col, ftr_col]:
        raise ValueError("CSV file missing required columns")

    # Create a list to store recent form per team
    teams = pd.concat([df[home_col], df[away_col]]).unique()
    recent_form = []

    for team in teams:
        # Filter matches where team played as home or away
        team_matches = df[(df[home_col] == team) | (df[away_col] == team)].tail(n_last_matches)

        # Calculate W/D/L counts
        wins = 0
        draws = 0
        losses = 0
        for _, match in team_matches.iterrows():
            if match[ftr_col] == "H" and match[home_col] == team:
                wins += 1
            elif match[ftr_col] == "A" and match[away_col] == team:
                wins += 1
            elif match[ftr_col] == "D":
                draws += 1
            else:
                losses += 1

        recent_form.append({
            "team": team,
            "wins_last_5": wins,
            "draws_last_5": draws,
            "losses_last_5": losses
        })

    recent_form_df = pd.DataFrame(recent_form)
    return recent_form_df
