import requests
import json
import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Secure API Key
API_KEY = os.environ.get("API_SPORTS_KEY")
HEADERS = {
    "x-apisports-key": API_KEY,
    "x-apisports-host": "v3.football.api-sports.io"
}

# The 17 formations the Streamlit app uses
FORMATIONS_MAP = {
    "3-4-3": 0, "3-5-2": 1, "3-4-1-2": 2, "3-2-4-1": 3, "3-4-2-1": 4, "3-3-1-3": 5,
    "4-2-3-1": 6, "4-3-3": 7, "4-4-2": 8, "4-4-2 Diamond": 9, "4-1-4-1": 10,
    "4-3-2-1": 11, "4-2-2-2": 12, "5-3-2": 13, "5-4-1": 14, "5-2-2-1": 15, "5-2-3": 16
}

# ============================================================
# API BUDGET for this script:
#   1 request  — fetch fixtures list
#   up to 20   — lineup fetches (one per fixture, capped at 20)
#   TOTAL: ≤ 21 requests
#
# auto_updater.py uses 20 requests.
# Combined daily spend: ≤ 41 / 100 — well within the free limit.
# ============================================================
MAX_LINEUP_FETCHES = 20


def harvest_and_train():
    print("🧠 Booting up the True ML Harvester...")

    # --- 1. Fetch completed Premier League fixtures ---
    url = "https://v3.football.api-sports.io/fixtures?league=39&season=2024&status=FT"
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"🚨 Network error fetching fixtures: {e}")
        return

    if response.status_code != 200:
        print(f"🚨 API Connection Failed! Status: {response.status_code}")
        return

    fixtures = response.json().get('response', [])
    print(f"📡 Found {len(fixtures)} completed matches. Fetching lineups for up to {MAX_LINEUP_FETCHES}...")

    historical_data = []
    lineup_fetches = 0

    # --- 2. Fetch lineups for a capped batch of fixtures ---
    for fixture in fixtures[:MAX_LINEUP_FETCHES]:
        fixture_id  = fixture['fixture']['id']
        home_team   = fixture['teams']['home']['name']
        away_team   = fixture['teams']['away']['name']
        home_goals  = fixture['goals']['home']
        away_goals  = fixture['goals']['away']

        lineup_url = f"https://v3.football.api-sports.io/fixtures/lineups?fixture={fixture_id}"
        try:
            lineup_res = requests.get(lineup_url, headers=HEADERS, timeout=15)
            lineup_fetches += 1
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error for fixture {fixture_id}: {e}. Skipping.")
            continue

        if lineup_res.status_code == 429:
            print(f"🚨 RATE LIMITED after {lineup_fetches} lineup fetches. Stopping early.")
            break

        lineup_data = lineup_res.json().get('response', [])

        if len(lineup_data) == 2:
            home_form = lineup_data[0].get('formation', '')
            away_form = lineup_data[1].get('formation', '')

            if home_form in FORMATIONS_MAP and away_form in FORMATIONS_MAP:
                historical_data.append({
                    'Formation': FORMATIONS_MAP[home_form],
                    'Team_Attack': 80, 'Team_Defense': 80,
                    'Opp_Attack': 80, 'Opp_Defense': 80,
                    'Win': 1 if home_goals > away_goals else 0
                })
                historical_data.append({
                    'Formation': FORMATIONS_MAP[away_form],
                    'Team_Attack': 80, 'Team_Defense': 80,
                    'Opp_Attack': 80, 'Opp_Defense': 80,
                    'Win': 1 if away_goals > home_goals else 0
                })
                print(f"  ✅ Logged: {home_team} ({home_form}) vs {away_team} ({away_form})")

        time.sleep(0.3)  # polite delay between requests

    print(f"📊 Total lineup API requests used: {lineup_fetches}")

    # --- 3. Train the model ---
    if historical_data:
        df = pd.DataFrame(historical_data)
        print(f"🤖 Training Random Forest on {len(df)} match records...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(
            df[['Formation', 'Team_Attack', 'Team_Defense', 'Opp_Attack', 'Opp_Defense']],
            df['Win']
        )
        joblib.dump(model, 'tactical_model.pkl')
        print("🏆 tactical_model.pkl saved! The AI brain is ready.")
    else:
        print("⚠️  No valid formation data found. Model not updated.")


if __name__ == "__main__":
    harvest_and_train()
