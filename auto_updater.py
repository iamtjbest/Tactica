import json, requests, os, time

# ================================================================
# BSD API v2 — auto_updater.py
# Docs: https://sports.bzzoiro.com/docs/v2/
#
# Endpoints used:
#   GET /api/v2/players/?team_id={id}&limit=50
#       → player name + position (G/D/M/F)
#   GET /api/v2/teams/{id}/fixtures/?status=finished&limit=5
#       → last 5 match IDs for the team
#   GET /api/v2/events/{id}/player-stats/
#       → per-player minutes_played, goals, goal_assist
#
# No rate limits on BSD free plan.
# Cost: ~7 calls per team × 20 teams = ~140 calls total (no limit)
#
# NOTE: BSD Team IDs may differ from placeholders below.
#   To verify: GET /api/v2/teams/?league_id=17  (17 = Premier League on BSD)
#   Or run the helper: python auto_updater.py --lookup
# ================================================================

BSD_KEY = os.environ.get("BSD_API_KEY")
HEADERS = {"Authorization": f"Token {BSD_KEY}"}
BASE    = "https://sports.bzzoiro.com/api/v2"

# BSD position codes from docs
POS_MAP = {"G": "GK", "D": "DF", "M": "MF", "F": "FW"}

# ----------------------------------------------------------------
# BSD Team IDs — verify with GET /api/v2/teams/?league_id=17
# Docs example confirms Man City = 267
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

# Static fallback ratings (no API call needed)
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
    """Safe GET wrapper with error handling."""
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        return r
    except Exception as e:
        print(f"    🚨 Network error: {e}")
        return None


def lookup_team_ids():
    """Helper: print BSD IDs for all PL teams. Run once to verify IDs."""
    print("🔍 Looking up Premier League team IDs on BSD...")
    r = get(f"{BASE}/teams/", {"league_id": 17, "limit": 30})
    if r and r.status_code == 200:
        teams = r.json().get("results", [])
        for t in teams:
            print(f"  {t['name']:30s} → id = {t['id']}")
    else:
        print("  ❌ Failed to fetch teams list")


def fetch_latest_stats():
    print("🤖 BSD Auto-Updater starting...")

    if not BSD_KEY:
        print("🚨 BSD_API_KEY not set in environment. Aborting.")
        return

    players_db = {}

    for team_name, team_id in BSD_TEAM_IDS.items():
        print(f"\n📡 Processing {team_name} (BSD id={team_id})...")

        # ── Step 1: Get squad (names + positions) ────────────────────────
        # Docs: GET /api/v2/players/?team_id={id}&limit=50
        # Response shape: {"count":N, "results":[{"id":..,"name":..,"position":"M",...}]}
        r = get(f"{BASE}/players/", {"team_id": team_id, "limit": 50})
        if not r or r.status_code != 200:
            print(f"  ⚠️  Could not fetch players list (status {r.status_code if r else 'N/A'})")
            players_db[team_name] = []
            continue

        player_list = r.json().get("results", [])
        # Map player_id → {Name, Pos, Min:0, G_A:0}
        player_map = {}
        for p in player_list:
            raw_pos = str(p.get("position") or "M")
            player_map[p["id"]] = {
                "Name": p.get("name") or p.get("short_name") or "Unknown",
                "Pos":  POS_MAP.get(raw_pos, "MF"),
                "Min":  0,
                "G_A":  0,
            }
        print(f"  👥 {len(player_map)} players found")

        # ── Step 2: Get last 5 finished fixtures for this team ────────────
        # Docs: GET /api/v2/teams/{id}/fixtures/?status=finished&limit=5
        # Response: same shape as events list (results array of event objects)
        r2 = get(f"{BASE}/teams/{team_id}/fixtures/",
                 {"status": "finished", "limit": 5})
        fixture_ids = []
        if r2 and r2.status_code == 200:
            fixtures = r2.json().get("results", [])
            fixture_ids = [f["id"] for f in fixtures]
            print(f"  📅 {len(fixture_ids)} recent fixtures to scan for stats")
        else:
            print(f"  ⚠️  Could not fetch fixtures — using positions only (no stats)")

        # ── Step 3: Aggregate player stats across those fixtures ──────────
        # Docs: GET /api/v2/events/{id}/player-stats/
        # Response: {"event_id":..,"count":..,"player_stats":[
        #   {"player_id":..,"team_id":..,"minutes_played":..,"goals":..,"goal_assist":..}
        # ]}
        for fid in fixture_ids:
            r3 = get(f"{BASE}/events/{fid}/player-stats/")
            time.sleep(0.2)
            if not r3 or r3.status_code != 200:
                continue
            for ps in r3.json().get("player_stats", []):
                pid = ps.get("player_id")
                # Only count stats for players on this team
                if ps.get("team_id") != team_id:
                    continue
                if pid in player_map:
                    player_map[pid]["Min"] += ps.get("minutes_played") or 0
                    player_map[pid]["G_A"] += (ps.get("goals") or 0) + (ps.get("goal_assist") or 0)

        roster = list(player_map.values())
        players_db[team_name] = roster
        print(f"  ✅ {len(roster)} players saved for {team_name}")
        time.sleep(0.3)

    # ── Save outputs ──────────────────────────────────────────────────────
    with open("players.json", "w", encoding="utf-8") as f:
        json.dump(players_db, f, indent=2, ensure_ascii=False)
    print("\n🏆 players.json saved!")

    with open("teams.json", "w", encoding="utf-8") as f:
        json.dump(TEAM_RATINGS, f, indent=2, ensure_ascii=False)
    print("🛡️  teams.json saved!")


if __name__ == "__main__":
    import sys
    if "--lookup" in sys.argv:
        lookup_team_ids()
    else:
        fetch_latest_stats()
