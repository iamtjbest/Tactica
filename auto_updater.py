import json, requests, os, time

# ================================================================
# BSD API v2 — auto_updater.py
# Docs: https://sports.bzzoiro.com/docs/v2/
#
# Uses GET /api/v2/players/?team_id={id}&limit=100
# instead of the squad endpoint — because the PLAYERS endpoint
# returns specific_position (RW, LM, CAM, ST, etc.) in addition
# to the generic position (G/D/M/F). This is what lets us correctly
# classify Saka as FW and not MF.
#
# API calls per team:
#   1 x GET /api/v2/players/?team_id={id}&limit=100  → full squad + specific positions
#   1 x GET /api/v2/events/?team_id={id}&status=finished&limit=5 → last 5 match IDs
#   5 x GET /api/v2/events/{id}/player-stats/        → mins + goals + assists
#   ──────────────────────────────────────────────────────────────
#   7 calls per team × 20 teams = 140 total
#   BSD free plan: NO rate limits — completely safe.
# ================================================================

BSD_KEY  = os.environ.get("BSD_API_KEY")
HEADERS  = {"Authorization": f"Token {BSD_KEY}"}
BASE     = "https://sports.bzzoiro.com/api/v2"

# ----------------------------------------------------------------
# POSITION MAPPING — specific_position → internal code
#
# BSD specific_position field values (from /api/v2/players/):
#   Goalkeepers : GK
#   Defenders   : CB, RB, LB, RWB, LWB
#   True mids   : CM, CDM, DM, CAM, AM
#   WINGERS     : RM, LM, RW, LW, RWF, LWF  ← the key fix
#   Forwards    : ST, CF, SS
#
# RM and LM are wide midfielders who in modern football play as
# wide forwards (like Saka, Salah, Mane, Mbappe at times).
# We map them → FW so they fill wide forward slots in 4-3-3 etc.
# ----------------------------------------------------------------
SPECIFIC_POS_MAP = {
    # Goalkeepers
    "GK": "GK",
    # Defenders
    "CB": "DF", "RB": "DF", "LB": "DF", "RWB": "DF", "LWB": "DF", "SW": "DF",
    # True central midfielders
    "CM": "MF", "CDM": "MF", "DM": "MF",
    # Attacking mids — kept as MF (they fill the 10 role, not wide forward)
    "CAM": "MF", "AM": "MF",
    # Wide players — the key fix: RM/LM are wingers in modern football
    "RM": "FW", "LM": "FW",
    "RW": "FW", "LW": "FW",
    "RWF": "FW", "LWF": "FW",
    # Forwards
    "ST": "FW", "CF": "FW", "SS": "FW",
}

# Fallback: generic BSD position code (G/D/M/F) → internal code
GENERIC_POS_MAP = {"G": "GK", "D": "DF", "M": "MF", "F": "FW"}

# ----------------------------------------------------------------
# BSD Team IDs for Premier League clubs.
# Run once to verify: GET /api/v2/teams/?league_id=17&limit=20
# Manchester City confirmed ID = 267 from BSD docs example.
# Others estimated — verify and update if any return empty squads.
# ----------------------------------------------------------------
BSD_TEAM_IDS = {
    # Premier League
    "Manchester City": 267, "Arsenal": 2,   "Liverpool": 10,
    "Aston Villa": 24,      "Tottenham": 6, "Manchester Utd": 8,
    "Chelsea": 4,           "Newcastle": 19,"Brighton": 36,
    "West Ham": 20,         "Crystal Palace":31,"Everton": 14,
    "Fulham": 43,           "Brentford": 189,"Bournemouth": 91,
    "Nott'm Forest": 17,   "Wolves": 39,   "Leicester": 26,
    "Southampton": 57,      "Ipswich": 40,
}

TEAM_RATINGS = {
    "Manchester City": {"Attack":92,"Defense":88}, "Arsenal":       {"Attack":88,"Defense":90},
    "Liverpool":       {"Attack":89,"Defense":87}, "Aston Villa":   {"Attack":84,"Defense":82},
    "Tottenham":       {"Attack":86,"Defense":81}, "Manchester Utd":{"Attack":82,"Defense":80},
    "Chelsea":         {"Attack":84,"Defense":81}, "Newcastle":     {"Attack":83,"Defense":82},
    "Brighton":        {"Attack":82,"Defense":80}, "West Ham":      {"Attack":81,"Defense":81},
    "Crystal Palace":  {"Attack":79,"Defense":79}, "Everton":       {"Attack":77,"Defense":82},
    "Fulham":          {"Attack":79,"Defense":78}, "Brentford":     {"Attack":79,"Defense":78},
    "Bournemouth":     {"Attack":78,"Defense":77}, "Nott'm Forest": {"Attack":77,"Defense":77},
    "Wolves":          {"Attack":78,"Defense":78}, "Leicester":     {"Attack":76,"Defense":75},
    "Southampton":     {"Attack":75,"Defense":75}, "Ipswich":       {"Attack":74,"Defense":74},
}


def resolve_position(generic_pos, specific_pos):
    """
    Resolve the best internal position from BSD's two position fields.
    specific_position takes priority when available and recognised.
    generic_position (G/D/M/F) is the fallback.
    """
    if specific_pos:
        sp = specific_pos.strip().upper()
        if sp in SPECIFIC_POS_MAP:
            return SPECIFIC_POS_MAP[sp]
    # Fallback to generic
    gp = (generic_pos or "M").strip().upper()
    return GENERIC_POS_MAP.get(gp, "MF")


def get_json(url, params=None):
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        print(f"  ⚠️  HTTP {r.status_code}: {url}")
    except Exception as e:
        print(f"  🚨 {e}: {url}")
    return None


def fetch_latest_stats():
    print("🤖 BSD Auto-Updater starting...")
    if not BSD_KEY:
        print("🚨 BSD_API_KEY not set. Aborting.")
        return

    players_db = {}

    for team_name, team_id in BSD_TEAM_IDS.items():
        print(f"\n📡 {team_name} (id={team_id})")

        # ── Step 1: Players list endpoint (has specific_position) ─────────
        # GET /api/v2/players/?team_id={id}&limit=100
        # Returns: {"count":N, "results":[{id, name, short_name, position,
        #           specific_position, jersey_number, ...}]}
        player_data = get_json(f"{BASE}/players/", params={"team_id": team_id, "limit": 100})
        if not player_data:
            print(f"  ⚠️  No player data — skipping")
            players_db[team_name] = []
            continue

        players_by_id = {}
        for p in player_data.get("results", []):
            name = p.get("name") or p.get("short_name", "")
            if not name or name.strip() == "":
                continue  # skip null/empty names — fixes "None" cards

            pid          = p["id"]
            generic_pos  = p.get("position", "M")          # G / D / M / F
            specific_pos = p.get("specific_position", "")   # RW / LM / CAM / ST / etc.
            internal_pos = resolve_position(generic_pos, specific_pos)

            players_by_id[pid] = {
                "Name":    name.strip(),
                "Pos":     internal_pos,
                "SpecPos": specific_pos or generic_pos,  # store for display/debug
                "Min":     0,
                "G_A":     0,
            }

        print(f"  ✅ {len(players_by_id)} players (with specific positions)")
        time.sleep(0.2)

        # ── Step 2: Last 5 finished matches ───────────────────────────────
        # GET /api/v2/events/?team_id={id}&status=finished&limit=5
        events = get_json(f"{BASE}/events/",
                          params={"team_id": team_id, "status": "finished", "limit": 5})
        match_ids = [ev["id"] for ev in (events or {}).get("results", [])]
        print(f"  ✅ {len(match_ids)} recent matches found")
        time.sleep(0.2)

        # ── Step 3: Player stats per match ────────────────────────────────
        # GET /api/v2/events/{id}/player-stats/
        # Returns: {"player_stats":[{player_id, team_id, minutes_played,
        #           goals, goal_assist, ...}]}
        for mid in match_ids:
            ps_data = get_json(f"{BASE}/events/{mid}/player-stats/")
            if not ps_data:
                continue
            for ps in ps_data.get("player_stats", []):
                if ps.get("team_id") != team_id:
                    continue
                pid = ps.get("player_id")
                if pid in players_by_id:
                    players_by_id[pid]["Min"] += ps.get("minutes_played") or 0
                    players_by_id[pid]["G_A"] += (
                        (ps.get("goals") or 0) + (ps.get("goal_assist") or 0)
                    )
            time.sleep(0.2)

        roster = list(players_by_id.values())
        players_db[team_name] = roster

        # Useful debug: show position breakdown
        from collections import Counter
        pos_count = Counter(p["Pos"] for p in roster)
        print(f"  📊 GK:{pos_count['GK']} DF:{pos_count['DF']} MF:{pos_count['MF']} FW:{pos_count['FW']}")

    with open("players.json", "w", encoding="utf-8") as f:
        json.dump(players_db, f, indent=2, ensure_ascii=False)
    print("\n🏆 players.json saved!")

    with open("teams.json", "w", encoding="utf-8") as f:
        json.dump(TEAM_RATINGS, f, indent=2, ensure_ascii=False)
    print("🛡️  teams.json saved!")


if __name__ == "__main__":
    fetch_latest_stats()
