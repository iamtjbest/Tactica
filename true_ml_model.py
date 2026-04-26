import requests
import json
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Secure API Key
API_KEY = os.environ.get("API_SPORTS_KEY")
HEADERS = {
    "x-apisports-key": API_KEY,
    "x-apisports-host": "v3.football.api-sports.io"
}

# The 17 formations your Streamlit app uses
FORMATIONS_MAP = {
    "3-4-3": 0, "3-5-2": 1, "3-4-1-2": 2, "3-2-4-1": 3, "3-4-2-1": 4, "3-3-1-3": 5,
    "4-2-3-1": 6, "4-3-3": 7, "4-4-2": 8, "4-4-2 Diamond": 9, "4-1-4-1": 10, "4-3-2-1": 11, "4-2-2-2": 12,
    "5-3-2": 13, "5-4-1": 14, "5-2-2-1": 15, "5-2-3": 16
}

def harvest_and_train():
    print("🧠 Booting up the True ML Harvester...")
    
    # We will look at the English Premier League (League ID 39, 2023 Season)
    url = "https://v3.football.api-sports.io/fixtures?league=39&season=2023&status=FT"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print("🚨 API Connection Failed!")
        return

    fixtures = response.json().get('response', [])
    print(f"📡 Found {len(fixtures)} completed matches. Fetching formations for a batch of 20 to stay under daily limits...")
    
    historical_data = []
    
    # Loop through just 20 matches so we don't blow the 100 request/day limit
    for fixture in fixtures[:20]:
        fixture_id = fixture['fixture']['id']
        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']
        home_goals = fixture['goals']['home']
        away_goals = fixture['goals']['away']
        
        # Ping the lineup endpoint to see what formations they actually used
        lineup_url = f"https://v3.football.api-sports.io/fixtures/lineups?fixture={fixture_id}"
        lineup_res = requests.get(lineup_url, headers=HEADERS).json().get('response', [])
        
        if len(lineup_res) == 2:
            home_form = lineup_res[0]['formation']
            away_form = lineup_res[1]['formation']
            
            # Map strings back to our engine's internal codes
            if home_form in FORMATIONS_MAP and away_form in FORMATIONS_MAP:
                # We save two rows per match (from the perspective of both teams)
                
                # Home perspective
                historical_data.append({
                    'Formation': FORMATIONS_MAP[home_form],
                    'Team_Attack': 80, 'Team_Defense': 80, # Hardcoded baseline for this prototype
                    'Opp_Attack': 80, 'Opp_Defense': 80,
                    'Win': 1 if home_goals > away_goals else 0
                })
                
                # Away perspective
                historical_data.append({
                    'Formation': FORMATIONS_MAP[away_form],
                    'Team_Attack': 80, 'Team_Defense': 80,
                    'Opp_Attack': 80, 'Opp_Defense': 80,
                    'Win': 1 if away_goals > home_goals else 0
                })
                print(f"✅ Logged: {home_team} ({home_form}) vs {away_team} ({away_form})")

    # Train the True ML Model
    if historical_data:
        df = pd.DataFrame(historical_data)
        print("🤖 Training the Random Forest on real historical data...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(df[['Formation', 'Team_Attack', 'Team_Defense', 'Opp_Attack', 'Opp_Defense']], df['Win'])
        
        # Save the brain!
        joblib.dump(model, 'tactical_model.pkl')
        print("🏆 tactical_model.pkl saved! The true AI brain is ready.")
    else:
        print("⚠️ No valid formation data found to train on.")

if __name__ == "__main__":
    harvest_and_train()
