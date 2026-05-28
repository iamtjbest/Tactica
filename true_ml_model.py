# ══════════════════════════════════════════════════════════════════
#  TACTICA — TECHNICAL FIXES  (paste into true_ml_model.py)
#  Covers:
#    1. API season fix  (Dec 2025 stale data → May 2026 current)
#    2. Position mapping fix  (Amad Diallo / pitch-role-first logic)
#    3. Formation suggester upgrade  (last-5-match optimal XI)
# ══════════════════════════════════════════════════════════════════

import requests, json, os, time
from datetime import datetime, timedelta
from collections import Counter

API_KEY    = os.environ.get("API_SPORTS_KEY", "")
BASE_URL   = "https://v3.football.api-sports.io"
HEADERS    = {"x-apisports-key": API_KEY}
CACHE_FILE = "form_cache.json"
CACHE_TTL  = 86400   # 24 hours


# ──────────────────────────────────────────────────────────────────
#  FIX 1 · SEASON PARAMETER
#  API-Sports free tier defaults to whatever season it last indexed.
#  Always pass season=2024 for the 2024/25 season (Aug 2024–May 2026).
#  Also pass a `from` date so you never get pre-season or old data.
# ──────────────────────────────────────────────────────────────────

def _current_season() -> int:
    """
    Returns 2024 for the 2024/25 season which runs until ~June 2026.
    Change to 2025 once the 2025/26 season starts (typically Aug 2026).
    """
    today = datetime.utcnow()
    # Season year = the year the season STARTED
    # 2024/25 started Aug 2024, ends ~June 2026
    if today >= datetime(2026, 8, 1):
        return 2025
    return 2024


def _season_start_date() -> str:
    """
    Returns the ISO start date for the current season.
    Used as `from` filter so we never get stale prior-season fixtures.
    """
    season = _current_season()
    return f"{season}-08-01"   # e.g. "2024-08-01"


def get_last_5_fixtures(team_id: int) -> list:
    """
    Fetch the last 5 COMPLETED fixtures for a team in the current season.
    Passes season + from-date to avoid stale Dec-2025 data on free tier.
    """
    cache = _load_cache()
    cache_key = f"fixtures_{team_id}"

    if cache_key in cache:
        if time.time() - cache[cache_key]["ts"] < CACHE_TTL:
            return cache[cache_key]["data"]

    params = {
        "team":   team_id,
        "season": _current_season(),   # ← FIX: was missing, caused stale data
        "from":   _season_start_date(),  # ← FIX: floor date prevents old fixtures
        "status": "FT",                # finished matches only
        "last":   5,
    }
    r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params=params)
    if r.status_code != 200:
        return []

    fixtures = r.json().get("response", [])

    cache[cache_key] = {"ts": time.time(), "data": fixtures}
    _save_cache(cache)
    return fixtures


# ──────────────────────────────────────────────────────────────────
#  FIX 2 · PITCH-ROLE-FIRST POSITION MAPPING
#
#  Problem: API-Sports returns Amad Diallo as "MF" in player profiles.
#  But in actual lineup data, his grid position is "4:3" (right wing).
#  We read the GRID field from lineup data and map that to a real role.
#
#  Grid format: "row:column"  (row 1 = GK, row 2 = DEF, etc.)
#  Columns vary by formation width.
# ──────────────────────────────────────────────────────────────────

# Fallback: map API position labels to slot categories
API_POS_MAP = {
    "G":  "GK",
    "D":  "DEF",
    "M":  "MID",
    "F":  "FWD",
    # Hybrid labels → treat by first character
}

# Grid-row to positional tier
GRID_ROW_MAP = {
    1: "GK",
    2: "DEF",
    3: "DEF",   # some 5-back formations use row 3 as CB
    4: "MID",
    5: "MID",
    6: "FWD",
    7: "FWD",
}

# For 4-3-3 / 4-2-3-1 style formations, specific column+row → wing role
WING_GRID = {
    # (row, col) : role
    (4, 1): "LM",
    (4, 3): "RM",
    (5, 1): "LW",
    (5, 3): "RW",
    (6, 1): "LW",
    (6, 3): "RW",
}


def grid_to_role(grid_str: str, formation_width: int = 3) -> str:
    """
    Convert a player's grid position (e.g. '4:3') to a positional role.
    Correctly identifies wide players as LW/RW rather than generic MID/FWD.

    Args:
        grid_str:        e.g. "4:3"
        formation_width: number of players in the widest midfield row (default 3)

    Returns:
        role string: "GK", "DEF", "MID", "LM", "RM", "LW", "RW", "FWD", "ST"
    """
    if not grid_str or ":" not in grid_str:
        return "MID"

    try:
        row, col = map(int, grid_str.split(":"))
    except ValueError:
        return "MID"

    # Check for wing position explicitly
    if (row, col) in WING_GRID:
        return WING_GRID[(row, col)]

    # Central striker (top row, central column)
    if row >= 5 and col == 2:
        return "ST"

    # Fall back to row-based tier
    return GRID_ROW_MAP.get(row, "MID")


def extract_players_from_lineup(lineup_data: dict) -> list:
    """
    Given a single fixture lineup dict (from /fixtures/lineups),
    return a list of dicts:
      { player_id, name, grid_role, api_pos, number }

    grid_role is derived from grid position first — overrides api_pos.
    This is how Amad Diallo gets tagged as RW instead of MF.
    """
    players = []
    for p in lineup_data.get("startXI", []):
        player = p.get("player", {})
        grid   = player.get("grid", "")
        api_pos = player.get("pos", "M")

        grid_role = grid_to_role(grid)   # pitch-position-first
        fallback  = API_POS_MAP.get(api_pos[0].upper(), "MID")
        role = grid_role if grid_role != "MID" else fallback

        players.append({
            "id":        player.get("id"),
            "name":      player.get("name", ""),
            "number":    player.get("number"),
            "grid":      grid,
            "grid_role": grid_role,
            "api_pos":   api_pos,
            "role":      role,       # ← FINAL role used for selection
        })
    return players


# ──────────────────────────────────────────────────────────────────
#  FIX 3 · FORMATION SUGGESTER UPGRADE
#
#  Old logic: picked a fixed formation from a list.
#  New logic:
#    1. Fetch last 5 completed matches for the team
#    2. Count which formations were used most (Counter)
#    3. From those 5 matches, collect all players who started
#    4. Fetch player stats for the season (goals, assists, rating)
#    5. For each slot in the most-common formation, pick the highest-
#       rated available player whose grid_role matches that slot
#    6. Return the optimal XI + the formation string
# ──────────────────────────────────────────────────────────────────

SLOT_PRIORITY = {
    # formation → ordered list of slots to fill
    "4-3-3": ["GK", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "LW", "ST", "RW"],
    "4-2-3-1": ["GK", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "LM", "MID", "RM", "ST"],
    "4-4-2": ["GK", "DEF", "DEF", "DEF", "DEF", "LM",  "MID", "MID", "RM",  "ST", "ST"],
    "3-5-2": ["GK", "DEF", "DEF", "DEF", "LM",  "MID", "MID", "MID", "RM",  "ST", "ST"],
    "5-3-2": ["GK", "DEF", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "ST", "ST"],
}

# Role groups for flexible matching (e.g. a MID can play LM/RM if needed)
ROLE_FLEX = {
    "LW":  ["LW", "LM", "MID", "FWD"],
    "RW":  ["RW", "RM", "MID", "FWD"],
    "LM":  ["LM", "LW", "MID"],
    "RM":  ["RM", "RW", "MID"],
    "ST":  ["ST", "FWD", "MID"],
    "MID": ["MID", "LM", "RM"],
    "DEF": ["DEF"],
    "GK":  ["GK"],
}


def suggest_optimal_xi(team_id: int) -> dict:
    """
    Main function: returns the optimal XI for a team based on last 5 matches.

    Returns:
    {
        "formation": "4-3-3",
        "players": [
            { "name": "...", "role": "GK", "number": 1, "rating": 7.2 },
            ...  (11 players)
        ],
        "form_counts": {"4-3-3": 3, "4-2-3-1": 2},
        "source_matches": 5
    }
    """
    fixtures = get_last_5_fixtures(team_id)
    if not fixtures:
        return {"error": "No fixture data available. Check API quota or season parameter."}

    formation_counts = Counter()
    # player_id → { name, role, appearances, total_rating, number }
    player_pool: dict = {}

    for fix in fixtures:
        fix_id = fix["fixture"]["id"]
        lineups = _fetch_lineups(fix_id, team_id)
        if not lineups:
            continue

        formation = lineups.get("formation", "4-3-3")
        formation_counts[formation] += 1

        players_in_match = extract_players_from_lineup(lineups)
        stats_in_match   = _fetch_player_stats_for_fixture(fix_id, team_id)

        for p in players_in_match:
            pid  = p["id"]
            stat = stats_in_match.get(pid, {})
            rating = float(stat.get("rating") or 0)

            if pid not in player_pool:
                player_pool[pid] = {
                    "name":          p["name"],
                    "number":        p["number"],
                    "role":          p["role"],   # grid-derived role
                    "appearances":   0,
                    "total_rating":  0.0,
                    "goals":         0,
                    "assists":       0,
                }
            player_pool[pid]["appearances"]  += 1
            player_pool[pid]["total_rating"] += rating
            player_pool[pid]["goals"]        += int(stat.get("goals", {}).get("total") or 0)
            player_pool[pid]["assists"]      += int(stat.get("goals", {}).get("assists") or 0)

    if not formation_counts:
        return {"error": "Could not parse lineups from last 5 fixtures."}

    best_formation = formation_counts.most_common(1)[0][0]

    # Compute average rating per player
    for pid, data in player_pool.items():
        apps = data["appearances"]
        data["avg_rating"] = round(data["total_rating"] / apps, 2) if apps > 0 else 0.0

    # Sort player pool by avg_rating descending
    ranked = sorted(player_pool.values(), key=lambda x: x["avg_rating"], reverse=True)

    # Fill slots for the best formation
    slots = SLOT_PRIORITY.get(best_formation, SLOT_PRIORITY["4-3-3"])
    selected_xi = []
    used_names  = set()

    for slot in slots:
        flex_roles = ROLE_FLEX.get(slot, [slot])
        # Find the best available player for this slot
        for candidate in ranked:
            if candidate["name"] in used_names:
                continue
            if candidate["role"] in flex_roles:
                selected_xi.append({
                    "name":       candidate["name"],
                    "number":     candidate["number"],
                    "role":       slot,
                    "grid_role":  candidate["role"],
                    "avg_rating": candidate["avg_rating"],
                    "goals":      candidate["goals"],
                    "assists":    candidate["assists"],
                })
                used_names.add(candidate["name"])
                break

    return {
        "formation":      best_formation,
        "players":        selected_xi,
        "form_counts":    dict(formation_counts),
        "source_matches": len(fixtures),
        "note": (
            "Player availability (injury/suspension) not included in v1.0. "
            "Verify squad news manually before finalising selection."
        )
    }


# ──────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────

def _fetch_lineups(fixture_id: int, team_id: int) -> dict | None:
    """Fetch lineup data for one team in one fixture."""
    r = requests.get(
        f"{BASE_URL}/fixtures/lineups",
        headers=HEADERS,
        params={"fixture": fixture_id}
    )
    if r.status_code != 200:
        return None
    for lineup in r.json().get("response", []):
        if lineup.get("team", {}).get("id") == team_id:
            return lineup
    return None


def _fetch_player_stats_for_fixture(fixture_id: int, team_id: int) -> dict:
    """
    Returns a dict of { player_id: stats_dict } for one team/fixture.
    Stats include rating, goals, assists.
    """
    r = requests.get(
        f"{BASE_URL}/fixtures/players",
        headers=HEADERS,
        params={"fixture": fixture_id}
    )
    if r.status_code != 200:
        return {}
    result = {}
    for team_block in r.json().get("response", []):
        if team_block.get("team", {}).get("id") != team_id:
            continue
        for player_block in team_block.get("players", []):
            pid   = player_block["player"]["id"]
            stats = player_block.get("statistics", [{}])[0]
            result[pid] = stats
    return result


def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(data: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f)


# ──────────────────────────────────────────────────────────────────
#  PLAYER AVAILABILITY — v1.1 ROADMAP NOTE
# ──────────────────────────────────────────────────────────────────
#
#  API-Sports has /players/injuries?team=X&season=2024
#  This returns current injury + suspension data.
#
#  WHY WE'RE DEFERRING TO v1.1:
#    - Costs 1 API call per team request
#    - On 100 req/day free tier, adding this per-team call on every
#      formation request would exhaust quota quickly
#    - Correct approach in v1.1:
#        a) Run injury sync once daily via GitHub Actions
#        b) Store results in injury_cache.json (TTL: 12 hours)
#        c) In suggest_optimal_xi(), filter player_pool by:
#           if player_id not in injured_or_suspended → include
#        d) Show UI badge: "⚠️ Availability verified" vs "⚠️ Manual check needed"
#
#  TODO v1.1:
#    def get_injury_cache(team_id):
#        r = requests.get(f"{BASE_URL}/players/injuries",
#                         headers=HEADERS,
#                         params={"team": team_id, "season": _current_season()})
#        return {p["player"]["id"] for p in r.json().get("response", [])}
#
# ──────────────────────────────────────────────────────────────────
