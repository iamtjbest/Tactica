import json
import requests
import os
import time

# Securely grab the API key from GitHub Secrets
API_KEY = os.environ.get("API_SPORTS_KEY")

HEADERS = {
    "x-apisports-key": API_KEY,
    "x-apisports-host": "v3.football.api-sports.io"
}

# ============================================================
# API BUDGET: 100 requests/day (free plan)
# auto_updater.py uses 1 request per team.
# We track 20 Premier League teams = 20 requests.
# true_ml_model.py uses 1 (fixtures) + up to 20 (lineups) = 21 requests.
# GRAND TOTAL: 20 + 21 = 41 requests — safely under the 100/day limit.
#
# ⚠️  The GitHub Actions workflow runs this ONCE per day.
#     Do NOT run this script manually on the same day the workflow fires.
# ============================================================

# Reduced to Premier League only (20 teams = 20 API requests)
TEAM_IDS = {
    "Manchester City": 47, "Arsenal": 42, "Liverpool": 40, "Aston Villa": 66,
    "Tottenham": 43, "Manchester Utd": 33, "Chelsea": 49, "Newcastle": 34,
    "Brighton": 51, "West Ham": 48, "Crystal Palace": 52, "Everton": 45,
    "Fulham": 36, "Brentford": 55, "Bournemouth": 35, "Nott'm Forest": 65,
    "Wolves": 39, "Leicester": 46, "Southampton": 41, "Ipswich": 62,
}

# Static team ratings — no API needed for these.
TEAM_RATINGS = {
    "Manchester City": {"Attack": 92, "Defense": 88},
    "Arsenal":         {"Attack": 88, "Defense": 90},
    "Liverpool":       {"Attack": 89, "Defense": 87},
    "Aston Villa":     {"Attack": 84, "Defense": 82},
    "Tottenham":       {"Attack": 86, "Defense": 81},
    "Manchester Utd":  {"Attack": 82, "Defense": 80},
    "Chelsea":         {"Attack": 84, "Defense": 81},
    "Newcastle":       {"Attack": 83, "Defense": 82},
    "Brighton":        {"Attack": 82, "Defense": 80},
    "West Ham":        {"Attack": 81, "Defense": 81},
    "Crystal Palace":  {"Attack": 79, "Defense": 79},
    "Everton":         {"Attack": 77, "Defense": 82},
    "Fulham":          {"Attack": 79, "Defense": 78},
    "Brentford":       {"Attack": 79, "Defense": 78},
    "Bournemouth":     {"Attack": 78, "Defense": 77},
    "Nott'm Forest":   {"Attack": 77, "Defense": 77},
    "Wolves":          {"Attack": 78, "Defense": 78},
    "Leicester":       {"Attack": 76, "Defense": 75},
    "Southampton":     {"Attack": 75, "Defense": 75},
    "Ipswich":         {"Attack": 74, "Defense": 74},
}


def check_api_quota():
    """Check remaining API requests before running the full update."""
    try:
        res = requests.get(
            "https://v3.football.api-sports.io/status",
            headers=HEADERS,
            timeout=10
        )
        if res.status_code == 200:
            data = res.json().get("response", {})
            requests_used = data.get("requests", {}).get("current", 0)
            requests_limit = data.get("requests", {}).get("limit_day", 100)
            remaining = requests_limit - requests_used
            print(f"📊 API Quota: {requests_used}/{requests_limit} used — {remaining} remaining today.")
            return remaining
        else:
            print(f"⚠️  Could not check quota. Status: {res.status_code}")
            return None
    except Exception as e:
        print(f"⚠️  Quota check failed: {e}")
        return None


def fetch_latest_stats():
    print("🤖 Booting up API-SPORTS data fetcher...")

    # --- Safety gate: abort if quota is too low ---
    remaining = check_api_quota()
    teams_to_fetch = len(TEAM_IDS)  # 20

    if remaining is not None and remaining < teams_to_fetch + 5:
        print(f"🚨 ABORT: Only {remaining} requests remaining — not enough for {teams_to_fetch} teams. Skipping.")
        return

    # 1. Load existing database
    try:
        with open('players.json', 'r', encoding='utf-8') as f:
            players_db = json.load(f)
    except Exception:
        print("📂 players.json not found — creating fresh database.")
        players_db = {}

    requests_made = 0

    # 2. Loop through teams and fetch player stats
    for team_name, team_id in TEAM_IDS.items():
        print(f"📡 Fetching data for {team_name}...")

        url = f"https://v3.football.api-sports.io/players?team={team_id}&season=2024"
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            requests_made += 1
        except requests.exceptions.RequestException as e:
            print(f"🚨 Network error for {team_name}: {e}. Skipping.")
            continue

        if response.status_code == 200:
            data = response.json()
            errors = data.get("errors", {})
            if errors:
                print(f"🚨 API error for {team_name}: {errors}. Skipping.")
                continue

            players_list = data.get('response', [])
            print(f"  ✅ {len(players_list)} players downloaded for {team_name}.")

            team_roster = []
            for p in players_list:
                player_info = p.get('player', {})
                stats = p.get('statistics', [{}])[0] if p.get('statistics') else {}
                games = stats.get('games', {})
                goals_stats = stats.get('goals', {})

                minutes = games.get('minutes') or 0
                goals   = goals_stats.get('total') or 0
                assists = goals_stats.get('assists') or 0

                raw_pos = games.get('position', 'Unknown')
                pos_map = {"Attacker": "FW", "Midfielder": "MF", "Defender": "DF", "Goalkeeper": "GK"}
                mapped_pos = pos_map.get(raw_pos, raw_pos)

                team_roster.append({
                    "Name": player_info.get('name', 'Unknown'),
                    "Pos":  mapped_pos,
                    "Min":  minutes,
                    "G_A":  goals + assists
                })

            players_db[team_name] = team_roster

        elif response.status_code == 429:
            print(f"🚨 RATE LIMITED after {requests_made} requests. Stopping.")
            break
        else:
            print(f"🚨 API error for {team_name}. Status: {response.status_code}. Skipping.")

        time.sleep(0.5)  # polite delay

    # 3. Save player data
    with open('players.json', 'w', encoding='utf-8') as f:
        json.dump(players_db, f, indent=2, ensure_ascii=False)
    print(f"🏆 players.json saved! ({requests_made} API requests used this run.)")

    # 4. Save team ratings (no API call needed)
    with open('teams.json', 'w', encoding='utf-8') as f:
        json.dump(TEAM_RATINGS, f, indent=2, ensure_ascii=False)
    print("🛡️  teams.json saved!")


if __name__ == "__main__":
    fetch_latest_stats()
