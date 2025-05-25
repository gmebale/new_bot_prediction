import csv
from datetime import datetime, timezone
from fetch_data import fetch_matches
from predict import predict_match
from telegram_utils import send_message
import train_model

def main():
    print("Fetching historical data...")
    df_past = fetch_matches(match_status="FINISHED", days_range=365)  # Fetch past matches for training
    df_past.to_csv("historical_matches.csv", index=False)

    print("Training model...")
    train_model.train()

    print("Fetching current standings...")
    from data_integration import fetch_football_data_api_competition_standings

    competition_codes = ["PD", "SA"]  # Bundesliga and Ligue 1
    for comp in competition_codes:
        standings_df = fetch_football_data_api_competition_standings(comp)
        standings_df.to_csv(f"standings_{comp}.csv", index=False)

    print("Fetching upcoming matches...")
    from fetch_data import fetch_matches_by_matchday

    # Fetch upcoming matches for Bundesliga and Ligue 1 for a given matchday
    import pandas as pd
    from fetch_data import fetch_matches_by_matchday

    competition_codes = ["PD", "SA"]  # Bundesliga and Ligue 1
    upcoming_matchday = 38

    dfs = []
    for comp in competition_codes:
        df = fetch_matches_by_matchday(comp, upcoming_matchday)
        dfs.append(df)

    df_upcoming = pd.concat(dfs, ignore_index=True)

    if df_upcoming.empty:
        print("No upcoming matches found.")
        return

    print("Predicting upcoming matches...")
    for i in range(len(df_upcoming)):
        home = df_upcoming.iloc[i]["home_team"]
        away = df_upcoming.iloc[i]["away_team"]
        prediction = predict_match(home, away)

        confidence = prediction["confidence"]
        level = prediction.get("confidence_level", "")

        msg = f"""
📊 *PRÉDICTION DE MATCH*

🏟️ {home} 🆚 {away}
🔮 Résultat attendu : {prediction['emoji']} {prediction['result']}
📈 Probabilité : {confidence}% ({level})
📊 Score estimé : {prediction['predicted_score']}
⚽️ Pari total buts : {prediction['over_under']}
{prediction.get('home_total_bets', '')}
{prediction.get('away_total_bets', '')}

📅 {datetime.now(timezone.utc).strftime("%Y-%m-%d")}
🤖 Bot: FootPredict
"""

        send_message(msg.strip())

        with open("predictions_log.csv", "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now(timezone.utc),
                home,
                away,
                prediction['result'],
                prediction['confidence'],
                prediction['predicted_score'],
                prediction['over_under'],
                prediction.get('home_total_bets', ''),
                prediction.get('away_total_bets', '')
            ])

if __name__ == "__main__":
    main()
