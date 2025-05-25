import pickle
import pandas as pd
from prepare_data import prepare_enriched_data
import numpy as np
import os

def safe_transform(le, labels):
    classes = set(le.classes_)
    transformed = []
    for label in labels:
        if label in classes:
            transformed.append(le.transform([label])[0])
        else:
            # Assign a default encoding for unseen labels, e.g., -1
            transformed.append(-1)
    return np.array(transformed)

def predict(matches):
    # Load model and label encoder
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # Encode teams using label encoder with safe transform
    matches["home_team_enc"] = safe_transform(le, matches["home_team"])
    matches["away_team_enc"] = safe_transform(le, matches["away_team"])

    # Load standings data for home and away teams
    competition_codes = ["BL1", "FL1"]  # Competitions considered
    standings_dfs_home = []
    standings_dfs_away = []

    for comp in competition_codes:
        standings_path = f"standings_{comp}.csv"
        if os.path.exists(standings_path):
            standings_df = pd.read_csv(standings_path)
            standings_home = standings_df.rename(columns={
                "team_name": "home_team",
                "position": "home_rank",
                "points": "home_points",
                "goals_for": "home_goals_for",
                "goals_against": "home_goals_against",
                "goal_difference": "home_goal_difference"
            })
            standings_home = standings_home[["home_team", "home_rank", "home_points", "home_goals_for", "home_goals_against", "home_goal_difference"]]
            standings_dfs_home.append(standings_home)

            standings_away = standings_df.rename(columns={
                "team_name": "away_team",
                "position": "away_rank",
                "points": "away_points",
                "goals_for": "away_goals_for",
                "goals_against": "away_goals_against",
                "goal_difference": "away_goal_difference"
            })
            standings_away = standings_away[["away_team", "away_rank", "away_points", "away_goals_for", "away_goals_against", "away_goal_difference"]]
            standings_dfs_away.append(standings_away)

    if standings_dfs_home:
        standings_home_df = pd.concat(standings_dfs_home, ignore_index=True)
        matches = matches.merge(standings_home_df, on="home_team", how="left")
    else:
        matches["home_rank"] = 0
        matches["home_points"] = 0
        matches["home_goals_for"] = 0
        matches["home_goals_against"] = 0
        matches["home_goal_difference"] = 0

    if standings_dfs_away:
        standings_away_df = pd.concat(standings_dfs_away, ignore_index=True)
        matches = matches.merge(standings_away_df, on="away_team", how="left")
    else:
        matches["away_rank"] = 0
        matches["away_points"] = 0
        matches["away_goals_for"] = 0
        matches["away_goals_against"] = 0
        matches["away_goal_difference"] = 0

    # Load historical matches for head-to-head and home/away performance
    try:
        historical_df = pd.read_csv("historical_matches.csv")
        # Compute result column
        historical_df["result"] = historical_df.apply(
            lambda x: 1 if x["home_score"] > x["away_score"] else (-1 if x["home_score"] < x["away_score"] else 0),
            axis=1
        )
    except FileNotFoundError:
        historical_df = pd.DataFrame()

    # Calculate head-to-head stats and home/away performance from historical matches
    h2h_stats = []
    for idx, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
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
    matches = pd.concat([matches.reset_index(drop=True), h2h_df], axis=1)

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

    matches = matches.merge(home_perf, on="home_team", how="left")
    matches = matches.merge(away_perf, on="away_team", how="left")

    # Integrate recent form features
    try:
        from data_integration import analyze_recent_form_from_csv
        recent_form_df = analyze_recent_form_from_csv()
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
        matches = matches.merge(recent_form_home, on="home_team", how="left")
        matches = matches.merge(recent_form_away, on="away_team", how="left")
    else:
        matches["home_wins_last_5"] = 0
        matches["home_draws_last_5"] = 0
        matches["home_losses_last_5"] = 0
        matches["away_wins_last_5"] = 0
        matches["away_draws_last_5"] = 0
        matches["away_losses_last_5"] = 0

    # Define function to classify match importance
    def classify_match_importance(row):
        competition = row.get("competition", "")
        home_rank = row.get("home_rank", 0)
        away_rank = row.get("away_rank", 0)

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

        if home_rank <= 2 or away_rank <= 2:
            return 5

        if (home_rank <= cl_spots + el_spots + cl_conf_spots and home_rank > 2) or \
           (away_rank <= cl_spots + el_spots + cl_conf_spots and away_rank > 2):
            return 4

        if (home_rank <= cl_spots + el_spots + cl_conf_spots + 2 and home_rank > cl_spots + el_spots + cl_conf_spots) or \
           (away_rank <= cl_spots + el_spots + cl_conf_spots + 2 and away_rank > cl_spots + el_spots + cl_conf_spots):
            return 3

        total_teams = 20
        relegation_zone_start = total_teams - relegation_spots + 1
        if home_rank >= relegation_zone_start or away_rank >= relegation_zone_start:
            return 5

        return 2

    # Add default competition column if missing
    if "competition" not in matches.columns:
        matches["competition"] = "PL"

    # Compute match importance
    matches["match_importance"] = matches.apply(classify_match_importance, axis=1)

    # Select features for prediction
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
    matches[feature_cols] = matches[feature_cols].fillna(0)

    X_pred = matches[feature_cols]

    # Predict probabilities
    probs = model.predict_proba(X_pred)

    # Map predictions to labels
    labels = model.classes_

    predictions = []
    for i, row in matches.iterrows():
        prob_dict = dict(zip(labels, probs[i]))
        pred_label = labels[probs[i].argmax()]
        predictions.append({
            "match": f"{row['home_team']} vs {row['away_team']}",
            "prediction": pred_label,
            "probability": prob_dict[pred_label]
        })

    return predictions

def predict_match(home_team, away_team):
    # Prepare a DataFrame for a single match
    match_df = pd.DataFrame([{"home_team": home_team, "away_team": away_team}])
    predictions = predict(match_df)
    pred = predictions[0]

    # Map prediction label to result and emoji
    label_map = {
        1: ("Victoire domicile", "🏠"),
        0: ("Match nul", "⚖️"),
        -1: ("Victoire extérieur", "🏃")
    }
    result_text, emoji = label_map.get(pred["prediction"], ("Inconnu", "❓"))

    confidence = pred["probability"]

    # Define 4 confidence levels with emojis
    if confidence < 0.4:
        confidence_level = "🔴"  # Boule Rouge: Pas très sûr
    elif confidence < 0.6:
        confidence_level = "🟠"  # Boule Orange: Peu sûr
    elif confidence < 0.8:
        confidence_level = "🟡"  # Boule Jaune: Sûr
    else:
        confidence_level = "🟢"  # Boule Verte: Très sûr

    confidence_percent = round(confidence * 100, 1)

    # Improved heuristic for predicted score and over/under
    # Use predicted probabilities and historical averages to estimate scores
    home_prob = pred["probability"] if pred["prediction"] == 1 else 0.5
    away_prob = pred["probability"] if pred["prediction"] == -1 else 0.5
    draw_prob = pred["probability"] if pred["prediction"] == 0 else 0.5

    # Use historical averages if available
    home_avg_goals = match_df.loc[0, "home_avg_goals_scored"] if "home_avg_goals_scored" in match_df.columns else 1.2
    away_avg_goals = match_df.loc[0, "away_avg_goals_scored"] if "away_avg_goals_scored" in match_df.columns else 1.0

    # Calculate expected goals with some weighting
    expected_home_goals = max(0, round(home_avg_goals * home_prob * 2))
    expected_away_goals = max(0, round(away_avg_goals * away_prob * 2))

    # Adjust for draw probability
    if pred["prediction"] == 0:
        expected_home_goals = expected_away_goals = max(0, round((home_avg_goals + away_avg_goals) / 2))

    predicted_score = f"{expected_home_goals}-{expected_away_goals}"

    # Determine over/under based on total goals
    total_goals = expected_home_goals + expected_away_goals
    over_under = "Over 2.5" if total_goals > 2.5 else "Under 2.5"

    # Bets based on expected goals
    home_total_bets = "Over 1.5" if expected_home_goals > 1.5 else "Under 1.5"
    away_total_bets = "Over 1.5" if expected_away_goals > 1.5 else "Under 1.5"

    return {
        "result": result_text,
        "emoji": emoji,
        "confidence": confidence_percent,
        "confidence_level": confidence_level,
        "predicted_score": predicted_score,
        "over_under": over_under,
        "home_total_bets": f"🏠 Pari total buts domicile: {home_total_bets}",
        "away_total_bets": f"🏃 Pari total buts extérieur: {away_total_bets}"
    }
