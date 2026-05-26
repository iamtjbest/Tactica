import json, requests, os, time

# ================================================================
# BSD API v2 — auto_updater.py
# Docs: https://sports.bzzoiro.com/docs/v2/
#
# Auth header : Authorization: Token YOUR_KEY
# Base URL    : https://sports.bzzoiro.com/api/v2/
#
# Calls per team:
#   1  x GET /api/v2/teams/{id}/squad/           → names + positions
#   1  x GET /api/v2/events/?team_id&status=finished&limit=5 → last 5 match IDs
#   5  x GET /api/v2/events/{id}/player-stats/   → mins + goals + assists
#   ─────────────────────────────────────────────────────────────
#   7 calls per team × 20 teams = 140 calls total
#   BSD free plan has NO rate limit — this is completely safe.
# ================================================================

BSD_KEY  = os.environ.get("BSD_API_KEY")
HEADERS  = {"Authorization": f"Token {BSD_KEY}"}
BASE     = "https://sports.bzzoiro.com/api/v2"

# ----------------------------------------------------------------
# BSD position codes (confirmed from docs squad response):
#   "G" → GK, "D" → DF, "M" → MF, "F" → FW
# ----------------------------------------------------------------
POS_MAP = {"G": "GK", "D": "DF", "M": "MF", "F": "FW"}

# ----------------------------------------------------------------
# BSD Team IDs for Premier League clubs.
# To verify / find IDs run once:
#   curl -H "Authorization: Token YOUR_KEY" \
#        "https://sports.bzzoiro.com/api/v2/teams/?league_id=17&limit=20"
# Then match "name" → "id" and update this dict.
# Premier League league_id = 17 (confirmed from BSD docs example)
# ----------------------------------------------------------------
BSD_TEAM_IDS = {
    "Manchester City": 267,
    "Arsenal":         2,
    "Liverpool":       10,
    "Aston Villa":     24,
    "Tottenham":       6,
    "Manchester Utd":  8,
    "Chelsea":         4,
    "Newcastle":       19,
    "Brighton":        36,
    "West Ham":        20,
    "Crystal Palace":  31,
    "Everton":         14,
    "Fulham":          43,
    "Brentford":       189,
    "Bournemouth":     91,
    "Nott'm Forest":   17,
    "Wolves":          39,
    "Leicester":       26,
    "Southampton":     57,
    "Ipswich":         40,
}

# Static fallback ratings (no API needed — update manually each season)
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


def get(url, params=None):
    """Simple GET helper with error handling."""
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        print(f"  ⚠️  {url} → HTTP {r.status_code}")
        return None
    except Exception as e:
        print(f"  🚨 {url} → {e}")
        return None


def fetch_latest_stats():
    print("🤖 BSD Auto-Updater starting...")

    if not BSD_KEY:
        print("🚨 BSD_API_KEY not set in environment. Aborting.")
        return

    players_db = {}

    for team_name, team_id in BSD_TEAM_IDS.items():
        print(f"\n📡 Processing {team_name} (BSD id={team_id})...")

        # ── Step 1: Fetch squad (names + positions) ───────────────────────
        # Docs: GET /api/v2/teams/{id}/squad/
        # Response: {"team_id": N, "count": N, "players": [{id, name,
        #            short_name, position, jersey_number, nationality, dob}]}
        squad_data = get(f"{BASE}/teams/{team_id}/squad/")
        if not squad_data:
            print(f"  ⚠️  No squad data — skipping {team_name}")
            players_db[team_name] = []
            continue

        players_by_id = {}
        for p in squad_data.get("players", []):
            pid = p["id"]
            raw_pos = p.get("position", "M")  # G / D / M / F
            players_by_id[pid] = {
                "Name": p.get("name") or p.get("short_name", "Unknown"),
                "Pos":  POS_MAP.get(raw_pos, "MF"),
                "Min":  0,
                "G_A":  0,
            }

        print(f"  ✅ {len(players_by_id)} players in squad")
        time.sleep(0.2)

        # ── Step 2: Fetch last 5 finished matches for this team ───────────
        # Docs: GET /api/v2/events/
        # Params: team_id, status=finished, limit=5
        # Response: {"count": N, "results": [event_objects]}
        # event object: id, home_team_id, away_team_id, home_score,
        #               away_score, league_id, status, ...
        events_data = get(
            f"{BASE}/events/",
            params={"team_id": team_id, "status": "finished", "limit": 5}
        )
        recent_match_ids = []
        if events_data:
            for ev in events_data.get("results", []):
                recent_match_ids.append(ev["id"])
        print(f"  ✅ {len(recent_match_ids)} recent finished matches found")
        time.sleep(0.2)

        # ── Step 3: Fetch player stats from each recent match ─────────────
        # Docs: GET /api/v2/events/{id}/player-stats/
        # Response: {"event_id": N, "count": N, "player_stats": [
        #   {player_id, team_id, minutes_played, goals, goal_assist, ...}]}
        for match_id in recent_match_ids:
            ps_data = get(f"{BASE}/events/{match_id}/player-stats/")
            if not ps_data:
                continue
            for ps in ps_data.get("player_stats", []):
                pid = ps.get("player_id")
                # Only count stats for players on THIS team
                if ps.get("team_id") != team_id:
                    continue
                if pid in players_by_id:
                    players_by_id[pid]["Min"] += ps.get("minutes_played") or 0
                    players_by_id[pid]["G_A"] += (
                        (ps.get("goals") or 0) + (ps.get("goal_assist") or 0)
                    )
            time.sleep(0.2)

        roster = list(players_by_id.values())
        players_db[team_name] = roster
        print(f"  ✅ {team_name} — {len(roster)} players with stats")

    # ── Save outputs ──────────────────────────────────────────────────────
    with open("players.json", "w", encoding="utf-8") as f:
        json.dump(players_db, f, indent=2, ensure_ascii=False)
    print("\n🏆 players.json saved!")

    with open("teams.json", "w", encoding="utf-8") as f:
        json.dump(TEAM_RATINGS, f, indent=2, ensure_ascii=False)
    print("🛡️  teams.json saved!")


if __name__ == "__main__":
    fetch_latest_stats()
