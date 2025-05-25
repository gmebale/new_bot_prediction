
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from fetch_data import fetch_weather_data

def prepare_data(filepath="historical_matches.csv"):
    df = pd.read_csv(filepath)
    df["result"] = df.apply(
        lambda x: 1 if x["home_score"] > x["away_score"] else (-1 if x["home_score"] < x["away_score"] else 0),
        axis=1
    )
    le = LabelEncoder()
    all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    le.fit(all_teams)
    df["home_team_enc"] = le.transform(df["home_team"])
    df["away_team_enc"] = le.transform(df["away_team"])
    X = df[["home_team_enc", "away_team_enc"]]
    y = df["result"]
    return X, y, le

def prepare_enriched_data(competition="PL", season=2024):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    import os

    # Load historical match data from local historical_matches.csv
    try:
        historical_df = pd.read_csv("historical_matches.csv")
    except FileNotFoundError:
        historical_df = pd.DataFrame()

    if historical_df.empty:
        # Return empty dataframes and a fitted LabelEncoder on empty list
        le = LabelEncoder()
        le.fit([])
        X = pd.DataFrame()
        y = pd.Series(dtype='int')
        return X, y, le

    # Compute result column
    historical_df["result"] = historical_df.apply(
        lambda x: 1 if x["home_score"] > x["away_score"] else (-1 if x["home_score"] < x["away_score"] else 0),
        axis=1
    )

    # Add weather features (demo: fetch current weather for home team city)
    # Removed weather features as per user request

    # Add match importance feature (simple placeholder based on competition and season)
    def classify_match_importance(row):
        """
        Classify match importance based on league standings and qualification/relegation stakes.
        Importance scale: 1 (low) to 5 (very high)
        """
        competition = row.get("competition", "")
        home_rank = row.get("home_rank", 0)
        away_rank = row.get("away_rank", 0)
        max_rank = max(home_rank, away_rank)
        min_rank = min(home_rank, away_rank)

        # Define league-specific qualification and relegation thresholds
        league_rules = {
            "PL": {"cl_spots": 5, "el_spots": 1, "cl_conf_spots": 1, "relegation_spots": 3},
            "FL1": {"cl_spots": 4, "el_spots": 1, "cl_conf_spots": 1, "relegation_spots": 3},
            "BL1": {"cl_spots": 4, "el_spots": 1, "cl_conf_spots": 1, "relegation_spots": 2},
            "SA": {"cl_spots": 4, "el_spots": 1, "cl_conf_spots": 1, "relegation_spots": 3},
            "DED": {"cl_spots": 2, "el_spots": 1, "cl_conf_spots": 1, "relegation_spots": 3},
            "PPL": {"cl_spots": 2, "el_spots": 1, "cl_conf_spots": 1, "relegation_spots": 2},
            "PD": {"cl_spots": 5, "el_spots": 1, "cl_conf_spots": 1, "relegation_spots": 3},
        }

        rules = league_rules.get(competition, None)
        if not rules:
            return 1  # Default low importance if unknown league

        cl_spots = rules["cl_spots"]
        el_spots = rules["el_spots"]
        cl_conf_spots = rules["cl_conf_spots"]
        relegation_spots = rules["relegation_spots"]

        # Title race importance: if either team is in top 2
        if home_rank <= 2 or away_rank <= 2:
            return 5

        # Champions League qualification battle
        if (home_rank <= cl_spots + el_spots + cl_conf_spots and home_rank > 2) or \
           (away_rank <= cl_spots + el_spots + cl_conf_spots and away_rank > 2):
            return 4

        # Europa League qualification battle
        if (home_rank <= cl_spots + el_spots + cl_conf_spots + 2 and home_rank > cl_spots + el_spots + cl_conf_spots) or \
           (away_rank <= cl_spots + el_spots + cl_conf_spots + 2 and away_rank > cl_spots + el_spots + cl_conf_spots):
            return 3

        # Relegation battle importance
        total_teams = 20  # Assuming 20 teams per league; adjust if needed
        relegation_zone_start = total_teams - relegation_spots + 1
        if home_rank >= relegation_zone_start or away_rank >= relegation_zone_start:
            return 5

        # Mid-table matches with no stakes
        return 2

    # Add columns for season and competition if not present
    if "season" not in historical_df.columns:
        historical_df["season"] = 2024  # default season
    if "competition" not in historical_df.columns:
        historical_df["competition"] = competition

    historical_df["match_importance"] = historical_df.apply(classify_match_importance, axis=1)

    # Encode team names
    le = LabelEncoder()
    all_teams = pd.concat([historical_df["home_team"], historical_df["away_team"]]).dropna().unique()
    le.fit(all_teams)
    historical_df["home_team_enc"] = le.transform(historical_df["home_team"].fillna("Unknown"))
    historical_df["away_team_enc"] = le.transform(historical_df["away_team"].fillna("Unknown"))

    # Load standings data for home and away teams
    standings_home = pd.DataFrame()
    standings_away = pd.DataFrame()
    standings_path_home = f"standings_{competition}.csv"
    standings_path_away = f"standings_{competition}.csv"

    if os.path.exists(standings_path_home):
        standings_home = pd.read_csv(standings_path_home)
        standings_home = standings_home.rename(columns={
            "team_name": "home_team",
            "position": "home_rank",
            "points": "home_points",
            "goals_for": "home_goals_for",
            "goals_against": "home_goals_against",
            "goal_difference": "home_goal_difference"
        })
        standings_home = standings_home[["home_team", "home_rank", "home_points", "home_goals_for", "home_goals_against", "home_goal_difference"]]

    if os.path.exists(standings_path_away):
        standings_away = pd.read_csv(standings_path_away)
        standings_away = standings_away.rename(columns={
            "team_name": "away_team",
            "position": "away_rank",
            "points": "away_points",
            "goals_for": "away_goals_for",
            "goals_against": "away_goals_against",
            "goal_difference": "away_goal_difference"
        })
        standings_away = standings_away[["away_team", "away_rank", "away_points", "away_goals_for", "away_goals_against", "away_goal_difference"]]

    # Merge standings data with historical matches
    if not standings_home.empty:
        historical_df = historical_df.merge(standings_home, on="home_team", how="left")
    else:
        historical_df["home_rank"] = 0
        historical_df["home_points"] = 0
        historical_df["home_goals_for"] = 0
        historical_df["home_goals_against"] = 0
        historical_df["home_goal_difference"] = 0

    if not standings_away.empty:
        historical_df = historical_df.merge(standings_away, on="away_team", how="left")
    else:
        historical_df["away_rank"] = 0
        historical_df["away_points"] = 0
        historical_df["away_goals_for"] = 0
        historical_df["away_goals_against"] = 0
        historical_df["away_goal_difference"] = 0

    # Integrate recent form features
    try:
        from data_integration import analyze_recent_form_from_csv
        recent_form_df = analyze_recent_form_from_csv(competition)
    except Exception:
        recent_form_df = pd.DataFrame()

    if not recent_form_df.empty:
        recent_form_home = recent_form_df.rename(columns={
            "team": "home_team",
            "wins_last_5": "home_wins_last_5",
            "draws_last_5": "home_draws_last_5",
            "losses_last_5": "home_losses_last_5"
        })
        recent_form_away = recent_form_df.rename(columns={
            "team": "away_team",
            "wins_last_5": "away_wins_last_5",
            "draws_last_5": "away_draws_last_5",
            "losses_last_5": "away_losses_last_5"
        })
        historical_df = historical_df.merge(recent_form_home, on="home_team", how="left")
        historical_df = historical_df.merge(recent_form_away, on="away_team", how="left")
    else:
        historical_df["home_wins_last_5"] = 0
        historical_df["home_draws_last_5"] = 0
        historical_df["home_losses_last_5"] = 0
        historical_df["away_wins_last_5"] = 0
        historical_df["away_draws_last_5"] = 0
        historical_df["away_losses_last_5"] = 0

    # Calculate head-to-head stats and home/away performance from historical matches
    # Head-to-head: number of wins, draws, losses for home team against away team
    h2h_stats = []
    for idx, row in historical_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        # Filter past matches between these two teams
        past_matches = historical_df[
            ((historical_df["home_team"] == home) & (historical_df["away_team"] == away)) |
            ((historical_df["home_team"] == away) & (historical_df["away_team"] == home))
        ]
        wins = 0
        draws = 0
        losses = 0
        for _, match in past_matches.iterrows():
            if match["home_team"] == home:
                if match["result"] == 1:
                    wins += 1
                elif match["result"] == 0:
                    draws += 1
                else:
                    losses += 1
            else:
                if match["result"] == -1:
                    wins += 1
                elif match["result"] == 0:
                    draws += 1
                else:
                    losses += 1
        h2h_stats.append({
            "home_wins_h2h": wins,
            "home_draws_h2h": draws,
            "home_losses_h2h": losses
        })
    h2h_df = pd.DataFrame(h2h_stats)
    historical_df = pd.concat([historical_df.reset_index(drop=True), h2h_df], axis=1)

    # Home performance: average goals scored and conceded at home
    home_perf = historical_df.groupby("home_team").agg({
        "home_score": "mean",
        "away_score": "mean"
    }).rename(columns={
        "home_score": "home_avg_goals_scored",
        "away_score": "home_avg_goals_conceded"
    })

    # Away performance: average goals scored and conceded away
    away_perf = historical_df.groupby("away_team").agg({
        "away_score": "mean",
        "home_score": "mean"
    }).rename(columns={
        "away_score": "away_avg_goals_scored",
        "home_score": "away_avg_goals_conceded"
    })

    historical_df = historical_df.merge(home_perf, on="home_team", how="left")
    historical_df = historical_df.merge(away_perf, on="away_team", how="left")

    # Fill missing values for new features
    new_features = [
        "home_wins_last_5", "home_draws_last_5", "home_losses_last_5",
        "away_wins_last_5", "away_draws_last_5", "away_losses_last_5",
        "home_wins_h2h", "home_draws_h2h", "home_losses_h2h",
        "home_avg_goals_scored", "home_avg_goals_conceded",
        "away_avg_goals_scored", "away_avg_goals_conceded",
        "match_importance"
    ]
    historical_df[new_features] = historical_df[new_features].fillna(0)

    # Update feature columns
    feature_cols = [
        "home_team_enc", "away_team_enc",
        "home_rank", "home_points", "home_goals_for", "home_goals_against", "home_goal_difference",
        "away_rank", "away_points", "away_goals_for", "away_goals_against", "away_goal_difference",
        "home_wins_last_5", "home_draws_last_5", "home_losses_last_5",
        "away_wins_last_5", "away_draws_last_5", "away_losses_last_5",
        "home_wins_h2h", "home_draws_h2h", "home_losses_h2h",
        "home_avg_goals_scored", "home_avg_goals_conceded",
        "away_avg_goals_scored", "away_avg_goals_conceded",
        "match_importance"
    ]

    # Fill missing feature values with zeros
    historical_df[feature_cols] = historical_df[feature_cols].fillna(0)

    X = historical_df[feature_cols]
    y = historical_df["result"]

    return X, y, le
