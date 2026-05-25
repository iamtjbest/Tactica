import json, requests, os, time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta

# ================================================================
# BSD API v2 — true_ml_model.py
# Docs: https://sports.bzzoiro.com/docs/v2/
#
# Endpoints used:
#   GET /api/v2/events/?status=finished&league_id={id}&limit=50
#       → recent finished Premier League matches
#       Response: {"count":N,"results":[event objects]}
#       Event fields: id, home_team_id, away_team_id,
#                     home_team, away_team, home_score, away_score
#
#   GET /api/v2/events/{id}/lineups/
#       → formation used by each team
#       Response: {
#           "lineup_status": "confirmed|predicted|unavailable",
#           "lineups": {
#               "home": {"formation": "4-3-3", ...},
#               "away": {"formation": "4-4-2", ...}
#           }
#       }
#       IMPORTANT: when lineup_status == "unavailable", lineups == null
#
# No rate limits on BSD free plan.
# BSD Premier League league_id = 17 (verify at /api/v2/leagues/?country=England)
# ================================================================

BSD_KEY = os.environ.get("BSD_API_KEY")
HEADERS = {"Authorization": f"Token {BSD_KEY}"}
BASE    = "https://sports.bzzoiro.com/api/v2"

BSD_PL_LEAGUE_ID = 17  # Premier League — verify at GET /api/v2/leagues/?country=England

FORMATIONS_MAP = {
    "3-4-3": 0,  "3-5-2": 1,  "3-4-1-2": 2, "3-2-4-1": 3, "3-4-2-1": 4,
    "3-3-1-3": 5,"4-2-3-1": 6,"4-3-3": 7,   "4-4-2": 8,   "4-4-2 Diamond": 9,
    "4-1-4-1": 10,"4-3-2-1": 11,"4-2-2-2": 12,"5-3-2": 13, "5-4-1": 14,
    "5-2-2-1": 15,"5-2-3": 16
}


def get(url, params=None):
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        return r
    except Exception as e:
        print(f"  🚨 Network error: {e}")
        return None


def extract_formation(event_id, is_home):
    """
    Fetch lineup for an event and return the formation for home or away side.
    Returns None if lineup is unavailable or formation not in our known list.

    Docs: GET /api/v2/events/{id}/lineups/
    Response when confirmed/predicted:
      {
        "lineup_status": "confirmed",
        "lineups": {
          "home": {"formation": "4-3-3", ...},
          "away": {"formation": "4-4-2", ...}
        }
      }
    Response when unavailable:
      {"lineup_status": "unavailable", "lineups": null}
    """
    r = get(f"{BASE}/events/{event_id}/lineups/")
    time.sleep(0.2)
    if not r or r.status_code != 200:
        return None

    data = r.json()
    status = data.get("lineup_status", "unavailable")

    # lineups is null when unavailable — must check before accessing
    if status == "unavailable" or data.get("lineups") is None:
        return None

    side     = "home" if is_home else "away"
    side_obj = data["lineups"].get(side, {})
    formation = side_obj.get("formation", "")

    return formation if formation in FORMATIONS_MAP else None


def harvest_and_train():
    print("🧠 BSD ML Trainer starting...")

    if not BSD_KEY:
        print("🚨 BSD_API_KEY not set. Aborting.")
        return

    # ── Fetch last 30 days of finished PL matches ─────────────────────────
    # Docs: GET /api/v2/events/?status=finished&league_id=17&limit=50
    # Results ordered newest-first by event_date
    date_to   = datetime.utcnow().strftime("%Y-%m-%d")
    date_from = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"📡 Fetching finished PL matches {date_from} → {date_to}...")
    r = get(f"{BASE}/events/", {
        "league_id": BSD_PL_LEAGUE_ID,
        "status":    "finished",
        "date_from": date_from,
        "date_to":   date_to,
        "limit":     50,
    })

    if not r or r.status_code != 200:
        print(f"🚨 API error: {r.status_code if r else 'no response'}")
        return

    # Response shape: {"count":N, "next":..., "previous":..., "results":[...]}
    fixtures = r.json().get("results", [])
    print(f"📊 Found {len(fixtures)} finished matches. Extracting formations...")

    historical_data = []

    for fix in fixtures:
        event_id   = fix["id"]
        home_score = fix.get("home_score") or 0
        away_score = fix.get("away_score") or 0
        home_team  = fix.get("home_team", "?")
        away_team  = fix.get("away_team", "?")

        home_form = extract_formation(event_id, is_home=True)
        away_form = extract_formation(event_id, is_home=False)

        if home_form:
            historical_data.append({
                "Formation":    FORMATIONS_MAP[home_form],
                "Team_Attack":  80, "Team_Defense": 80,
                "Opp_Attack":   80, "Opp_Defense":  80,
                "Win": 1 if home_score > away_score else 0
            })
        if away_form:
            historical_data.append({
                "Formation":    FORMATIONS_MAP[away_form],
                "Team_Attack":  80, "Team_Defense": 80,
                "Opp_Attack":   80, "Opp_Defense":  80,
                "Win": 1 if away_score > home_score else 0
            })

        if home_form or away_form:
            print(f"  ✅ {home_team} ({home_form or '?'}) vs {away_team} ({away_form or '?'})")

    # ── Train and save model ──────────────────────────────────────────────
    if historical_data:
        df = pd.DataFrame(historical_data)
        print(f"\n🤖 Training RandomForest on {len(df)} records...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(
            df[["Formation","Team_Attack","Team_Defense","Opp_Attack","Opp_Defense"]],
            df["Win"]
        )
        joblib.dump(model, "tactical_model.pkl")
        print("🏆 tactical_model.pkl saved!")
    else:
        print("⚠️  No formation data found — model not updated.")


if __name__ == "__main__":
    harvest_and_train()
