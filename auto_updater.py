import json
import requests
import os

# Securely grab the API key from GitHub Secrets
API_KEY = os.environ.get("API_SPORTS_KEY")

HEADERS = {
    "x-apisports-key": API_KEY,
    "x-apisports-host": "v3.football.api-sports.io"
}

# 🌍 Expanded European Elite Database
# We are tracking 12 teams. This will only use 12 of your 100 daily API requests!
# 🌍 The Complete Top 5 Leagues Database (96 Teams)
# 🚨 CRITICAL WARNING: This uses exactly 96 of your 100 daily free API requests. 
# Do not trigger this manually if your GitHub Action is running on the same day.
TEAM_IDS = {
    # Premier League (England - 20 Teams)
    "Manchester City": 47, "Arsenal": 42, "Liverpool": 40, "Aston Villa": 66, "Tottenham": 43, "Manchester Utd": 33, "Chelsea": 49, "Newcastle": 34, "Brighton": 51, "West Ham": 48, "Crystal Palace": 52, "Everton": 45, "Fulham": 36, "Brentford": 55, "Bournemouth": 35, "Nott'm Forest": 65, "Wolves": 39, "Leicester": 46, "Southampton": 41, "Ipswich": 62,
    
    # La Liga (Spain - 20 Teams)
    "Real Madrid": 541, "Barcelona": 529, "Atletico Madrid": 530, "Girona": 547, "Athletic Club": 531, "Real Sociedad": 548, "Real Betis": 543, "Villarreal": 533, "Valencia": 532, "Sevilla": 536, "Getafe": 546, "Osasuna": 724, "Alaves": 542, "Mallorca": 534, "Celta Vigo": 538, "Las Palmas": 539, "Rayo Vallecano": 728, "Leganes": 537, "Valladolid": 719, "Espanyol": 540,
    
    # Serie A (Italy - 20 Teams)
    "Inter Milan": 505, "AC Milan": 489, "Juventus": 496, "Atalanta": 499, "Napoli": 492, "AS Roma": 497, "Lazio": 487, "Fiorentina": 502, "Bologna": 500, "Torino": 503, "Genoa": 495, "Monza": 511, "Empoli": 504, "Lecce": 867, "Udinese": 494, "Cagliari": 490, "Verona": 506, "Parma": 523, "Como": 859, "Venezia": 517,
    
    # Bundesliga (Germany - 18 Teams)
    "Bayer Leverkusen": 168, "Bayern Munich": 157, "RB Leipzig": 173, "Borussia Dortmund": 165, "VfB Stuttgart": 172, "Eintracht Frankfurt": 169, "SC Freiburg": 160, "Hoffenheim": 167, "Werder Bremen": 162, "Wolfsburg": 161, "Borussia M.Gladbach": 163, "Augsburg": 170, "Heidenheim": 184, "Mainz 05": 164, "VfL Bochum": 176, "Union Berlin": 182, "FC St. Pauli": 186, "Holstein Kiel": 192,
    
    # Ligue 1 (France - 18 Teams)
    "PSG": 85, "AS Monaco": 79, "Lille": 77, "Marseille": 81, "Brest": 106, "Lens": 84, "Nice": 82, "Lyon": 80, "Rennes": 94, "Reims": 93, "Toulouse": 96, "Strasbourg": 95, "Nantes": 83, "Montpellier": 87, "Le Havre": 92, "Auxerre": 108, "Angers": 76, "Saint-Etienne": 1063
}

# 🌍 Complete Team Ratings (Attack, Defense out of 100)
TEAM_RATINGS = {
    # England
    "Manchester City": {"Attack": 92, "Defense": 88}, "Arsenal": {"Attack": 88, "Defense": 90}, "Liverpool": {"Attack": 89, "Defense": 87}, "Aston Villa": {"Attack": 84, "Defense": 82}, "Tottenham": {"Attack": 86, "Defense": 81}, "Manchester Utd": {"Attack": 82, "Defense": 80}, "Chelsea": {"Attack": 84, "Defense": 81}, "Newcastle": {"Attack": 83, "Defense": 82}, "Brighton": {"Attack": 82, "Defense": 80}, "West Ham": {"Attack": 81, "Defense": 81}, "Crystal Palace": {"Attack": 79, "Defense": 79}, "Everton": {"Attack": 77, "Defense": 82}, "Fulham": {"Attack": 79, "Defense": 78}, "Brentford": {"Attack": 79, "Defense": 78}, "Bournemouth": {"Attack": 78, "Defense": 77}, "Nott'm Forest": {"Attack": 77, "Defense": 77}, "Wolves": {"Attack": 78, "Defense": 78}, "Leicester": {"Attack": 76, "Defense": 75}, "Southampton": {"Attack": 75, "Defense": 75}, "Ipswich": {"Attack": 74, "Defense": 74},
    
    # Spain
    "Real Madrid": {"Attack": 91, "Defense": 86}, "Barcelona": {"Attack": 87, "Defense": 84}, "Atletico Madrid": {"Attack": 84, "Defense": 88}, "Girona": {"Attack": 85, "Defense": 80}, "Athletic Club": {"Attack": 83, "Defense": 84}, "Real Sociedad": {"Attack": 82, "Defense": 83}, "Real Betis": {"Attack": 81, "Defense": 81}, "Villarreal": {"Attack": 83, "Defense": 79}, "Valencia": {"Attack": 80, "Defense": 81}, "Sevilla": {"Attack": 81, "Defense": 80}, "Getafe": {"Attack": 76, "Defense": 82}, "Osasuna": {"Attack": 78, "Defense": 80}, "Alaves": {"Attack": 76, "Defense": 78}, "Mallorca": {"Attack": 75, "Defense": 79}, "Celta Vigo": {"Attack": 78, "Defense": 77}, "Las Palmas": {"Attack": 75, "Defense": 76}, "Rayo Vallecano": {"Attack": 76, "Defense": 76}, "Leganes": {"Attack": 74, "Defense": 75}, "Valladolid": {"Attack": 74, "Defense": 74}, "Espanyol": {"Attack": 75, "Defense": 74},
    
    # Italy
    "Inter Milan": {"Attack": 87, "Defense": 90}, "AC Milan": {"Attack": 85, "Defense": 84}, "Juventus": {"Attack": 84, "Defense": 88}, "Atalanta": {"Attack": 86, "Defense": 82}, "Napoli": {"Attack": 85, "Defense": 83}, "AS Roma": {"Attack": 84, "Defense": 83}, "Lazio": {"Attack": 82, "Defense": 82}, "Fiorentina": {"Attack": 83, "Defense": 81}, "Bologna": {"Attack": 81, "Defense": 83}, "Torino": {"Attack": 78, "Defense": 84}, "Genoa": {"Attack": 79, "Defense": 80}, "Monza": {"Attack": 78, "Defense": 79}, "Empoli": {"Attack": 75, "Defense": 77}, "Lecce": {"Attack": 75, "Defense": 76}, "Udinese": {"Attack": 77, "Defense": 78}, "Cagliari": {"Attack": 76, "Defense": 75}, "Verona": {"Attack": 75, "Defense": 75}, "Parma": {"Attack": 75, "Defense": 74}, "Como": {"Attack": 74, "Defense": 74}, "Venezia": {"Attack": 73, "Defense": 73},
    
    # Germany
    "Bayer Leverkusen": {"Attack": 88, "Defense": 85}, "Bayern Munich": {"Attack": 90, "Defense": 84}, "RB Leipzig": {"Attack": 85, "Defense": 84}, "Borussia Dortmund": {"Attack": 85, "Defense": 83}, "VfB Stuttgart": {"Attack": 84, "Defense": 81}, "Eintracht Frankfurt": {"Attack": 83, "Defense": 81}, "SC Freiburg": {"Attack": 81, "Defense": 82}, "Hoffenheim": {"Attack": 82, "Defense": 78}, "Werder Bremen": {"Attack": 80, "Defense": 79}, "Wolfsburg": {"Attack": 79, "Defense": 80}, "Borussia M.Gladbach": {"Attack": 81, "Defense": 78}, "Augsburg": {"Attack": 79, "Defense": 78}, "Heidenheim": {"Attack": 77, "Defense": 78}, "Mainz 05": {"Attack": 78, "Defense": 78}, "VfL Bochum": {"Attack": 76, "Defense": 75}, "Union Berlin": {"Attack": 77, "Defense": 79}, "FC St. Pauli": {"Attack": 74, "Defense": 75}, "Holstein Kiel": {"Attack": 73, "Defense": 73},
    
    # France
    "PSG": {"Attack": 88, "Defense": 83}, "AS Monaco": {"Attack": 84, "Defense": 81}, "Lille": {"Attack": 82, "Defense": 83}, "Marseille": {"Attack": 83, "Defense": 81}, "Brest": {"Attack": 80, "Defense": 82}, "Lens": {"Attack": 81, "Defense": 82}, "Nice": {"Attack": 79, "Defense": 84}, "Lyon": {"Attack": 82, "Defense": 80}, "Rennes": {"Attack": 81, "Defense": 80}, "Reims": {"Attack": 79, "Defense": 80}, "Toulouse": {"Attack": 78, "Defense": 79}, "Strasbourg": {"Attack": 78, "Defense": 78}, "Nantes": {"Attack": 76, "Defense": 77}, "Montpellier": {"Attack": 77, "Defense": 76}, "Le Havre": {"Attack": 75, "Defense": 76}, "Auxerre": {"Attack": 75, "Defense": 75}, "Angers": {"Attack": 74, "Defense": 74}, "Saint-Etienne": {"Attack": 75, "Defense": 74}
}

def fetch_latest_stats():
    print("🤖 Booting up API-SPORTS data fetcher...")
    
    # 1. Load your existing database
    try:
        with open('players.json', 'r', encoding='utf-8') as f:
            players_db = json.load(f)
    except Exception as e:
        print(f"🚨 Error loading players.json. Creating a new database.")
        players_db = {} # Start fresh if file doesn't exist

    # 2. Loop through your teams and ping the API
    for team_name, team_id in TEAM_IDS.items():
        print(f"📡 Fetching live data for {team_name}...")
        
        url = f"https://v3.football.api-sports.io/players?team={team_id}&season=2025" # Note: Using 2024 since 2025 season hasn't officially started stats-wise in Europe
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            players_list = data.get('response', [])
            print(f"✅ Successfully downloaded stats for {len(players_list)} players from {team_name}!")
            
            # --- THE REAL DATA PARSER ---
            team_roster = []
            for p in players_list:
                player_info = p.get('player', {})
                # Get the stats for the first listed competition
                stats = p.get('statistics', [{}])[0] if p.get('statistics') else {}
                
                games = stats.get('games', {})
                goals_stats = stats.get('goals', {})
                
                # Calculate metrics
                minutes = games.get('minutes') or 0
                goals = goals_stats.get('total') or 0
                assists = goals_stats.get('assists') or 0
                
                # Map the API positions (Attacker, Defender, etc.) to our Engine's format (FW, DF)
                raw_pos = games.get('position', 'Unknown')
                pos_map = {"Attacker": "FW", "Midfielder": "MF", "Defender": "DF", "Goalkeeper": "GK"}
                mapped_pos = pos_map.get(raw_pos, raw_pos)

                team_roster.append({
                    "Name": player_info.get('name', 'Unknown'),
                    "Pos": mapped_pos,
                    "Min": minutes,
                    "G_A": goals + assists
                })
            
            # Update the specific team's roster in our database
            players_db[team_name] = team_roster
            
        else:
            print(f"🚨 API Connection Failed for {team_name}. Status Code: {response.status_code}")

    # 3. Save the fresh data back to the JSON file
    with open('players.json', 'w', encoding='utf-8') as f:
        json.dump(players_db, f, indent=2, ensure_ascii=False)
    print("🏆 players.json auto-update sequence complete!")

    # 4. Save the Team Ratings to teams.json
    with open('teams.json', 'w', encoding='utf-8') as f:
        json.dump(TEAM_RATINGS, f, indent=2, ensure_ascii=False)
    print("🛡️ teams.json auto-update sequence complete!")

if __name__ == "__main__":
    fetch_latest_stats()
