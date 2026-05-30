import streamlit as st
import pandas as pd
import numpy as np
import json, re, os, difflib, random
import joblib
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier

# ── API Keys ──────────────────────────────────────────────────────────────────
gemini_api_key = st.secrets.get("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    ai_model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(page_title="Tactica AI Engine", page_icon="⚽", layout="wide")

# ── Global CSS — consistent with mobile theme ─────────────────────────────────
st.markdown("""
<style>
/* ── Base & Background ── */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    background-color: #050d07 !important;
    color: #e2f0e6 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071209 0%, #0a1a0c 100%) !important;
    border-right: 1px solid #1a3a1e !important;
}
section[data-testid="stSidebar"] * { color: #c8e6cd !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 14px !important;
    padding: 4px 0 !important;
    color: #86efac !important;
}
section[data-testid="stSidebar"] h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    color: #22c55e !important;
    text-transform: uppercase !important;
}

/* ── Main titles ── */
h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.8rem !important;
    color: #22c55e !important;
    text-transform: uppercase !important;
    letter-spacing: 3px !important;
    text-shadow: 0 0 30px rgba(34,197,94,0.35) !important;
}
h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    color: #86efac !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* ── Selectboxes ── */
div[data-baseweb="select"] > div {
    background-color: #0d1f10 !important;
    border: 1px solid #1a4020 !important;
    border-radius: 8px !important;
    color: #e2f0e6 !important;
}
div[data-baseweb="select"] * { color: #e2f0e6 !important; }
div[data-baseweb="popover"] { background-color: #0d1f10 !important; border: 1px solid #22c55e !important; }

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, #15803d 0%, #166534 100%) !important;
    color: #f0fdf4 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 1px !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
    box-shadow: 0 0 20px rgba(34,197,94,0.4) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button:disabled {
    background: #1a2e1c !important;
    color: #4a6b4e !important;
    transform: none !important;
}

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1f10 0%, #0a1a0c 100%) !important;
    border: 1px solid #1a4020 !important;
    border-left: 3px solid #22c55e !important;
    border-radius: 10px !important;
    padding: 15px !important;
}
div[data-testid="metric-container"] label { color: #86efac !important; font-size: 13px !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #22c55e !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* ── Cards ── */
.player-card {
    background: linear-gradient(135deg, #0d1f10 0%, #091508 100%);
    border: 1px solid #1a4020;
    border-left: 3px solid #22c55e;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 6px;
    color: #e2f0e6;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: border-color 0.2s;
}
.player-card:hover { border-left-color: #4ade80; }

.pos-badge {
    background: #14532d;
    color: #86efac;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 700;
    font-family: 'Rajdhani', sans-serif;
    letter-spacing: 1px;
    margin-right: 8px;
}

.stat-text { color: #6b8f72; font-size: 13px; }

/* ── Alerts ── */
.live-alert {
    background: rgba(220,38,38,0.12);
    border-left: 4px solid #dc2626;
    border-radius: 8px;
    padding: 14px 18px;
    color: #fca5a5;
    margin: 10px 0;
}
.live-suggestion {
    background: rgba(34,197,94,0.1);
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 14px 18px;
    color: #86efac;
    margin: 10px 0;
    font-family: 'Rajdhani', sans-serif;
    font-size: 15px;
}

/* ── Section divider ── */
hr { border-color: #1a3a1e !important; }

/* ── Expander ── */
details { border: 1px solid #1a4020 !important; border-radius: 8px !important; background: #0a1a0c !important; }
summary { color: #86efac !important; }

/* ── Input / Chat ── */
div[data-testid="stChatInput"] textarea {
    background: #0d1f10 !important;
    border: 1px solid #1a4020 !important;
    color: #e2f0e6 !important;
    border-radius: 10px !important;
}
div[data-testid="stChatMessage"] { background: #0a1a0c !important; border: 1px solid #1a3a1e !important; border-radius: 10px !important; }

/* ── Caption / small text ── */
.stCaption, small, caption { color: #6b8f72 !important; }

/* ── Spinner ── */
div[data-testid="stSpinner"] p { color: #86efac !important; }
</style>
""", unsafe_allow_html=True)

# ── App Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:16px;margin-bottom:8px'>
  <span style='font-size:3rem'>⚽</span>
  <div>
    <h1 style='margin:0;line-height:1'>TACTICA</h1>
    <p style='margin:0;color:#6b8f72;font-size:13px;letter-spacing:2px'>TACTICAL AI ENGINE</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── ML Model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    FMAP = {
        0:"3-4-3",1:"3-5-2",2:"3-4-1-2",3:"3-2-4-1",4:"3-4-2-1",5:"3-3-1-3",
        6:"4-2-3-1",7:"4-3-3",8:"4-4-2",9:"4-4-2 Diamond",10:"4-1-4-1",
        11:"4-3-2-1",12:"4-2-2-2",13:"5-3-2",14:"5-4-1",15:"5-2-2-1",16:"5-2-3"
    }
    try:
        return joblib.load("tactical_model.pkl"), FMAP
    except:
        data = {"Formation":np.random.randint(0,17,1000),"Team_Attack":np.random.randint(50,99,1000),
                "Team_Defense":np.random.randint(50,99,1000),"Opp_Attack":np.random.randint(50,99,1000),
                "Opp_Defense":np.random.randint(50,99,1000),"Win":np.random.randint(0,2,1000)}
        df = pd.DataFrame(data)
        m  = RandomForestClassifier().fit(df[["Formation","Team_Attack","Team_Defense","Opp_Attack","Opp_Defense"]],df["Win"])
        return m, FMAP

model, formations_map = load_model()

# ── Load JSON data ─────────────────────────────────────────────────────────────
try:
    with open("teams.json","r",encoding="utf-8") as f: teams_db = json.load(f)
except: teams_db = {}

try:
    with open("players.json","r",encoding="utf-8") as f: players_db = json.load(f)
except: players_db = {}

# ── ALL EUROPEAN TEAMS — used for dropdowns (BSD searches by name dynamically) ─
# Top 5 leagues + UCL/Europa regulars. BSD API does name-search so no IDs needed here.
ALL_EUROPEAN_TEAMS = sorted([
    # Premier League
    "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton","Chelsea",
    "Crystal Palace","Everton","Fulham","Ipswich","Leicester","Liverpool",
    "Manchester City","Manchester Utd","Newcastle","Nott'm Forest","Southampton",
    "Tottenham","West Ham","Wolves",
    # La Liga
    "Real Madrid","Barcelona","Atletico Madrid","Athletic Club","Real Sociedad",
    "Real Betis","Villarreal","Valencia","Sevilla","Osasuna","Girona",
    "Getafe","Rayo Vallecano","Mallorca","Las Palmas","Celta Vigo",
    "Alaves","Leganes","Espanyol","Valladolid",
    # Bundesliga
    "Bayern Munich","Borussia Dortmund","Bayer Leverkusen","RB Leipzig",
    "Eintracht Frankfurt","VfB Stuttgart","SC Freiburg","Union Berlin",
    "Werder Bremen","Borussia Monchengladbach","Augsburg","Wolfsburg",
    "Hoffenheim","Mainz","FC Heidenheim","Bochum","Holstein Kiel","St Pauli",
    # Serie A
    "Inter Milan","AC Milan","Juventus","Napoli","Atalanta","AS Roma",
    "Lazio","Fiorentina","Bologna","Torino","Udinese","Genoa",
    "Cagliari","Hellas Verona","Empoli","Parma","Como","Venezia",
    "Lecce","Monza",
    # Ligue 1
    "Paris Saint-Germain","Monaco","Marseille","Lyon","Lille","Lens",
    "Nice","Rennes","Brest","Reims","Montpellier","Toulouse",
    "Strasbourg","Le Havre","Saint-Etienne","Angers","Nantes","Auxerre",
    # Eredivisie
    "Ajax","PSV Eindhoven","Feyenoord","AZ Alkmaar","FC Utrecht","FC Twente",
    # Portuguese Primeira Liga
    "Benfica","Porto","Sporting CP","Braga","Vitoria Guimaraes",
    # Scottish Premiership
    "Celtic","Rangers",
    # Belgian Pro League
    "Club Brugge","Anderlecht","Genk",
    # UCL / Europa regulars not above
    "Galatasaray","Fenerbahce","Besiktas","Trabzonspor",
    "Red Bull Salzburg","Sturm Graz",
    "Shakhtar Donetsk","Dynamo Kyiv",
    "Slavia Prague","Sparta Prague",
    "Olympiakos","Panathinaikos","PAOK",
])

# Merge with any teams already in teams_db (from players.json)
DROPDOWN_TEAMS = sorted(set(ALL_EUROPEAN_TEAMS) | set(teams_db.keys()))

# ── Position helpers ──────────────────────────────────────────────────────────
# Wide attacker specific position codes (BSD SpecPos field or compound Pos strings)
# Specific positions that mean "wide attacker" (winger / wide forward)
# RM and LM ARE wingers in modern football — Saka (RM), Salah (RM), Mbappe (LW) etc.
WIDE_ATT_SPECS = {"RM","LM","RW","LW","RWF","LWF","AM","CAM","SS","WF","W"}

def classify_player(p):
    """
    Classify a player into a tactical bucket.
    Priority: SpecPos (specific) > Pos (generic internal code)
    Returns: 'GK' | 'DF' | 'MF' | 'FW' | 'WIDE_ATT'

    WIDE_ATT players are formation-aware:
      - 3+ fwd formations (4-3-3, 3-4-3): WIDE_ATT fills FW slots (they ARE the wingers)
      - 1-2 fwd formations (4-4-2, 5-3-2): WIDE_ATT fills MF slots (wide mids / No.10)
    """
    pos  = str(p.get("Pos", "MF")).upper()
    spec = str(p.get("SpecPos", "")).strip().upper()

    # If SpecPos is explicitly a wide role, bucket as WIDE_ATT for smart placement
    if spec in WIDE_ATT_SPECS:
        return "WIDE_ATT"
    # After position re-resolution, Pos will already be FW for RM/LM players
    if "GK" in pos: return "GK"
    if "DF" in pos: return "DF"
    if "FW" in pos: return "FW"
    return "MF"

def get_display_pos(p, tactical_role):
    """Return the badge label to show on the player card (their TACTICAL role, not raw data)."""
    return tactical_role if tactical_role else classify_player(p)

def select_starting_xi(team_name, formation):
    """
    Formation-aware starting XI selection.

    KEY RULE — wide attackers (MF,FW hybrids / RW/LW/RM/LM/AM):
      - Formations with 3+ forwards (4-3-3, 3-4-3, 3-3-1-3 ...):
          Wide attackers fill FW slots first (they ARE the wingers).
      - Formations with 1-2 forwards (4-4-2, 4-2-3-1, 5-3-2 ...):
          Wide attackers fill MF slots (they play as wide mids / No.10s).

    This correctly places Saka as FW in a 4-3-3 and as wide MF in a 4-4-2.
    """
    if team_name not in players_db:
        close = difflib.get_close_matches(team_name, players_db.keys(), n=1, cutoff=0.6)
        if close: team_name = close[0]
        else: return None

    parts     = [int(x) for x in re.findall(r"\d+", formation)]
    def_count = parts[0]
    att_count = parts[-1]
    mid_count = sum(parts[1:-1]) if len(parts) > 2 else parts[1]

    # Filter null/empty names — prevents "None" cards appearing in the XI
    clean = [p for p in players_db[team_name]
             if p.get("Name") and str(p["Name"]).strip() not in ("", "None", "null")]

    # Re-resolve position using SpecPos if present (wingers stored as MF in old data)
    SPEC_OVERRIDE = {
        "GK":"GK",
        "CB":"DF","RB":"DF","LB":"DF","RWB":"DF","LWB":"DF",
        "CM":"MF","CDM":"MF","DM":"MF","CAM":"MF","AM":"MF",
        # Wide players → FW (the Saka fix)
        "RM":"FW","LM":"FW","RW":"FW","LW":"FW","RWF":"FW","LWF":"FW",
        "ST":"FW","CF":"FW","SS":"FW",
    }
    resolved = []
    for p in clean:
        p2 = dict(p)
        spec = str(p2.get("SpecPos","")).strip().upper()
        if spec and spec in SPEC_OVERRIDE:
            p2["Pos"] = SPEC_OVERRIDE[spec]
        resolved.append(p2)

    roster = sorted(resolved, key=lambda x: (x.get("Min",0), x.get("G_A",0)), reverse=True)

    # ── Bucket players by tactical type ──────────────────────────────────
    buckets = {"GK":[], "DF":[], "MF":[], "FW":[], "WIDE_ATT":[]}
    for p in roster:
        buckets[classify_player(p)].append(p)

    xi, used = [], set()

    def pick(pool, n, role):
        """Draft up to n players from pool, tagging each with their tactical role."""
        drafted = 0
        for p in pool:
            if drafted >= n: break
            if p["Name"] not in used:
                tagged = dict(p)
                tagged["_role"] = role
                xi.append(tagged)
                used.add(p["Name"])
                drafted += 1
        return drafted

    # ── Step 1: GK ────────────────────────────────────────────────────────
    if pick(buckets["GK"], 1, "GK") < 1:
        pick(roster, 1, "GK")

    # ── Step 2: Defenders ─────────────────────────────────────────────────
    df_pool = sorted(buckets["DF"], key=lambda x: (x["Min"], x["G_A"]), reverse=True)
    n = pick(df_pool, def_count, "DF")
    if n < def_count:
        # Fallback: DF hybrids (e.g. MF,DF players like Nørgaard as DF)
        df_hybrid = [p for p in roster if "DF" in p.get("Pos","") and p["Name"] not in used]
        pick(df_hybrid, def_count - n, "DF")

    # ── Step 3: Forwards — formation-aware wide attacker handling ─────────
    if att_count >= 3:
        # 3+ forwards = wide forward slots exist → wide attackers ARE the wingers
        fw_pool = sorted(buckets["FW"] + buckets["WIDE_ATT"],
                         key=lambda x: (x["Min"], x["G_A"]), reverse=True)
        mf_wide_reserve = []   # wide atts already committed to FW slots
    else:
        # 1-2 forwards = striker slots only → wide attackers fill MF as wide mids
        fw_pool          = sorted(buckets["FW"], key=lambda x: (x["Min"], x["G_A"]), reverse=True)
        mf_wide_reserve  = sorted(buckets["WIDE_ATT"], key=lambda x: (x["Min"], x["G_A"]), reverse=True)

    n_fw = pick(fw_pool, att_count, "FW")
    if n_fw < att_count:
        # Fallback: use wide attackers even in low-fw formations if no pure FW left
        remaining_wide = [p for p in buckets["WIDE_ATT"] if p["Name"] not in used]
        pick(sorted(remaining_wide, key=lambda x: (x["Min"], x["G_A"]), reverse=True),
             att_count - n_fw, "FW")

    # ── Step 4: Midfielders ───────────────────────────────────────────────
    mf_pool = sorted(buckets["MF"] + mf_wide_reserve,
                     key=lambda x: (x["Min"], x["G_A"]), reverse=True)
    n_mf = pick(mf_pool, mid_count, "MF")
    if n_mf < mid_count:
        # Fallback: use any remaining wide attackers not drafted yet
        remaining_wide = [p for p in buckets["WIDE_ATT"] if p["Name"] not in used]
        pick(sorted(remaining_wide, key=lambda x: (x["Min"], x["G_A"]), reverse=True),
             mid_count - n_mf, "MF")

    # ── Step 5: Emergency fill (data gap) ─────────────────────────────────
    for p in roster:
        if len(xi) >= 11: break
        if p["Name"] not in used:
            tagged = dict(p); tagged["_role"] = "?"; tagged["fallback"] = True
            xi.append(tagged); used.add(p["Name"])

    return xi

# ── BSD helpers ───────────────────────────────────────────────────────────────
BSD_BASE = "https://sports.bzzoiro.com/api/v2"

# Static league_id → name map (BSD events list has league_id only, not league_name)
LEAGUE_NAMES = {
    17:"Premier League", 8:"La Liga", 5:"Bundesliga", 11:"Serie A", 4:"Ligue 1",
    2:"Champions League", 3:"Europa League", 848:"Conference League",
    88:"Eredivisie", 94:"Primeira Liga", 39:"Scottish Premiership",
    144:"Belgian Pro League", 203:"Süper Lig", 197:"Austrian Bundesliga",
    235:"Ukrainian Premier League",
}

def bsd_find_team_id(team_name, api_key):
    """
    Search BSD for a team by name. Returns BSD team id or None.
    Uses GET /api/v2/teams/?name={name}&limit=3
    """
    import requests as _r
    hdrs = {"Authorization": f"Token {api_key}"}
    try:
        res = _r.get(f"{BSD_BASE}/teams/", headers=hdrs,
                     params={"name": team_name, "limit": 3}, timeout=10)
        if res.status_code == 200:
            results = res.json().get("results", [])
            if results:
                # Pick the best fuzzy match
                best = difflib.get_close_matches(
                    team_name.lower(),
                    [t["name"].lower() for t in results],
                    n=1, cutoff=0.4
                )
                if best:
                    for t in results:
                        if t["name"].lower() == best[0]:
                            return t["id"], t["name"]
                return results[0]["id"], results[0]["name"]
    except Exception:
        pass
    return None, None

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## NAVIGATION")
app_mode = st.sidebar.radio("Select Module:", [
    "🤖 Pre-Match Auto-Tactics",
    "📊 Pre-Match Opponent Analysis",
    "🧠 Coach's Sandbox",
    "⏱️ Live Match Simulator",
    "💬 Assistant Manager Chat",
])

# =============================================================================
# MODULE 1: PRE-MATCH AUTO-TACTICS
# =============================================================================
if app_mode == "🤖 Pre-Match Auto-Tactics":
    import requests as _req, time as _time

    st.markdown("## 🤖 Pre-Match Auto-Tactics")
    st.write("Select any two European teams. The engine fetches their last 5 matches via BSD API, "
             "extracts real formations, calculates live attack/defence ratings, "
             "then recommends the optimal game plan.")

    col1, col2 = st.columns(2)
    with col1:
        my_team  = st.selectbox("Your Team",  DROPDOWN_TEAMS,
                                index=DROPDOWN_TEAMS.index("Arsenal") if "Arsenal" in DROPDOWN_TEAMS else 0)
    with col2:
        opp_team = st.selectbox("Opponent",   DROPDOWN_TEAMS,
                                index=DROPDOWN_TEAMS.index("Chelsea") if "Chelsea" in DROPDOWN_TEAMS else 1)

    FORM_CACHE  = "form_cache.json"
    CACHE_TTL   = 86400  # 24 hours

    def load_fc():
        try:
            return json.load(open(FORM_CACHE, encoding="utf-8"))
        except: return {}

    def save_fc(d):
        try: json.dump(d, open(FORM_CACHE,"w",encoding="utf-8"), indent=2)
        except: pass

    def fetch_last5(team_name, api_key):
        """
        BSD v2:
          1. GET /api/v2/teams/?name={name}&limit=3  → resolve team_id dynamically
          2. GET /api/v2/teams/{id}/fixtures/?status=finished&limit=5
             Response: {"results": [event_objects]}
             event fields: id, home_team_id, home_team, away_team,
                           home_score, away_score, league_id
          3. GET /api/v2/events/{id}/lineups/
             Response: {"lineup_status": "confirmed|predicted|unavailable",
                        "lineups": {"home":{"formation":"4-3-3",...},
                                    "away":{"formation":"4-4-2",...}}}
             lineups is null when lineup_status == "unavailable"
        """
        fc    = load_fc()
        entry = fc.get(team_name, {})
        age   = _time.time() - entry.get("fetched_at", 0)
        if entry and age < CACHE_TTL:
            return entry.get("matches", []), True

        # Step 1: resolve BSD team_id from name
        team_id, matched_name = bsd_find_team_id(team_name, api_key)
        if not team_id:
            return [], False

        hdrs = {"Authorization": f"Token {api_key}"}

        # Step 2: fetch last 5 finished fixtures
        # Note: /teams/{id}/fixtures/ defaults to now-3h→now+7d so we add date_from
        # to go back 6 months and ensure we get historical results
        from datetime import datetime, timedelta
        date_from = (datetime.utcnow() - timedelta(days=180)).strftime("%Y-%m-%dT00:00:00Z")
        try:
            r = _req.get(f"{BSD_BASE}/teams/{team_id}/fixtures/", headers=hdrs,
                         params={"status":"finished","limit":5,"date_from":date_from},
                         timeout=12)
        except: return [], False
        if r.status_code != 200: return [], False

        fixtures = r.json().get("results", [])
        results  = []

        for fix in fixtures:
            fid        = fix.get("id", 0)
            home_id    = fix.get("home_team_id", 0)
            home_goals = fix.get("home_score") or 0
            away_goals = fix.get("away_score") or 0
            is_home    = (home_id == team_id)
            scored     = home_goals if is_home else away_goals
            conceded   = away_goals if is_home else home_goals
            opp_name   = fix.get("away_team","?") if is_home else fix.get("home_team","?")
            league_id  = fix.get("league_id", 0)
            competition = LEAGUE_NAMES.get(league_id, f"League {league_id}")
            result     = "W" if scored > conceded else ("D" if scored == conceded else "L")

            # Step 3: fetch lineup formation
            # lineups.home.formation / lineups.away.formation (not top-level)
            formation_used = "Unknown"
            try:
                lr = _req.get(f"{BSD_BASE}/events/{fid}/lineups/",
                              headers=hdrs, timeout=12)
                _time.sleep(0.2)
                if lr.status_code == 200:
                    ld = lr.json()
                    # lineup_status: confirmed | predicted | unavailable
                    # lineups is null when unavailable — MUST guard!
                    lu_status = ld.get("lineup_status", "unavailable")
                    lineups   = ld.get("lineups")
                    if lu_status != "unavailable" and lineups:
                        side = "home" if is_home else "away"
                        formation_used = (lineups.get(side) or {}).get("formation") or "Unknown"
            except: pass

            results.append({
                "fixture_id":  fid,
                "formation":   formation_used,
                "scored":      scored,
                "conceded":    conceded,
                "result":      result,
                "competition": competition,
                "opponent":    opp_name,
            })

        fc[team_name] = {"fetched_at": _time.time(), "matches": results,
                         "bsd_id": team_id, "bsd_name": matched_name}
        save_fc(fc)
        return results, False

    def compute_ratings(last5):
        if not last5: return None, None
        avg_s = sum(m["scored"]   for m in last5) / len(last5)
        avg_c = sum(m["conceded"] for m in last5) / len(last5)
        return min(99, int(60 + avg_s * 9.75)), max(60, min(99, int(99 - avg_c * 9.75)))

    def best_formation(last5):
        counts = {}
        for m in last5:
            f = m.get("formation","Unknown")
            if f and f != "Unknown": counts[f] = counts.get(f,0)+1
        return max(counts, key=counts.get) if counts else None

    # ── Cache status ─────────────────────────────────────────────────────────
    api_key = st.secrets.get("BSD_API_KEY", "")
    fc      = load_fc()
    my_c    = fc.get(my_team, {})
    opp_c   = fc.get(opp_team, {})

    if my_c and opp_c:
        mh  = int((_time.time() - my_c.get("fetched_at",0))  / 3600)
        oh  = int((_time.time() - opp_c.get("fetched_at",0)) / 3600)
        st.caption(f"📦 Cached form data — {my_team}: {mh}h ago | {opp_team}: {oh}h ago")

    with st.expander("ℹ️ API Usage Info"):
        st.markdown(
            "**Fresh fetch costs ~12 BSD API calls** (name lookup + 5 fixtures + 5 lineups per team). "
            "BSD has **no rate limits** on the free plan. Results cached for **24 hours** — "
            "re-clicking costs zero calls. Works for any European team."
        )

    btn = ("♻️ Generate Tactics (Cached) / Re-Fetch"
           if my_c and opp_c
           else "🔍 Fetch Last 5 Matches & Generate Optimal Tactics")

    if st.button(btn, use_container_width=True, type="primary"):
        if my_team == opp_team:
            st.error("🚨 A team cannot face itself!")
        elif not api_key:
            st.error("🚨 BSD_API_KEY missing from Streamlit Secrets.")
        else:
            with st.spinner(f"📡 Fetching last 5 matches for {my_team} and {opp_team}..."):
                my5,  my_cached  = fetch_last5(my_team,  api_key)
                opp5, opp_cached = fetch_last5(opp_team, api_key)

            if not my5 and not opp5:
                st.error("🚨 No match data returned. Possible causes:\n"
                         "- BSD_API_KEY not set in Streamlit Secrets\n"
                         "- Team name not found in BSD database\n"
                         "- Check the team name spelling matches BSD exactly")
            else:
                my_att_d,  my_def_d  = compute_ratings(my5)
                opp_att_d, opp_def_d = compute_ratings(opp5)

                # Fall back to static ratings if no live data
                fb = teams_db.get(my_team,  {"Attack":80,"Defense":80})
                fb2= teams_db.get(opp_team, {"Attack":80,"Defense":80})
                my_att  = my_att_d  or fb.get("Attack",  80)
                my_def  = my_def_d  or fb.get("Defense", 80)
                opp_att = opp_att_d or fb2.get("Attack",  80)
                opp_def = opp_def_d or fb2.get("Defense", 80)

                opp_habit = best_formation(opp5)
                my_habit  = best_formation(my5)

                # Score all 17 formations
                best_prob, best_form = 0, ""
                all_scores = {}
                for fc_code, fc_name in formations_map.items():
                    test = pd.DataFrame({"Formation":[fc_code],"Team_Attack":[my_att],
                                         "Team_Defense":[my_def],"Opp_Attack":[opp_att],
                                         "Opp_Defense":[opp_def]})
                    prob = model.predict_proba(test)[0][1] * 100
                    if my_habit  and fc_name == my_habit:  prob += 5
                    if opp_habit and opp_habit[0].isdigit():
                        opp_backs = int(opp_habit.split("-")[0])
                        if opp_backs >= 5 and fc_name.startswith("3"): prob -= 5
                    all_scores[fc_name] = round(prob, 1)
                    if prob > best_prob: best_prob, best_form = prob, fc_name

                # ── Results ──────────────────────────────────────────────────
                st.markdown("---")
                r1, r2, r3 = st.columns(3)
                r1.metric("✅ Recommended Formation", best_form)
                r2.metric("🤖 AI Win Probability",    f"{best_prob:.1f}%")
                r3.metric("📐 Opp. Usual Formation",  opp_habit or "Unknown")

                # Form tables
                st.markdown("### 📋 Last 5 Matches")
                fc1, fc2 = st.columns(2)

                def render_form(tname, last5, att, dfn, cached):
                    label = "📦 cached" if cached else "🔴 live"
                    st.markdown(f"**{tname}** <span style='font-size:12px;color:#6b8f72'>({label})</span>",
                                unsafe_allow_html=True)
                    st.caption(f"⚔️ Attack: {att} | 🛡️ Defence: {dfn}")
                    for m in last5:
                        col = {"W":"#22c55e","D":"#f59e0b","L":"#ef4444"}.get(m["result"],"#6b8f72")
                        badge = f"<span style='background:{col};color:#000;padding:1px 7px;border-radius:4px;font-weight:700;font-size:11px'>{m['result']}</span>"
                        st.markdown(
                            f"{badge} &nbsp;vs <b>{m['opponent']}</b> &nbsp;"
                            f"{m['scored']}–{m['conceded']} &nbsp;"
                            f"<code style='font-size:11px;background:#0d1f10;padding:2px 5px;border-radius:3px'>{m['formation']}</code> "
                            f"<span style='font-size:11px;color:#6b8f72'>{m['competition']}</span>",
                            unsafe_allow_html=True)

                with fc1: render_form(my_team,  my5,  my_att,  my_def,  my_cached)
                with fc2: render_form(opp_team, opp5, opp_att, opp_def, opp_cached)

                # Formation leaderboard
                st.markdown("### 🏆 Formation Win-Probability Ranking")
                for rank, (fname, score) in enumerate(
                        sorted(all_scores.items(), key=lambda x:x[1], reverse=True)[:5], 1):
                    medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][rank-1]
                    bar   = int((score/100)*280)
                    st.markdown(
                        f"{medal} **{fname}** &nbsp;"
                        f"<span style='display:inline-block;width:{bar}px;height:8px;"
                        f"background:linear-gradient(90deg,#22c55e,#4ade80);border-radius:4px;vertical-align:middle'></span>"
                        f"&nbsp; {score}%", unsafe_allow_html=True)

                # Starting XI
                st.markdown(f"### 👕 Recommended Starting XI — {best_form}")
                xi = select_starting_xi(my_team, best_form)
                if xi:
                    for p in xi:
                        warn = " ⚠️" if p.get("fallback") else ""
                        ga   = f"{p['G_A']:.2f}" if isinstance(p['G_A'], float) else str(p['G_A'])
                        # Use specific position for badge if informative (e.g. RM, LW, ST, CDM)
                        spec = str(p.get("SpecPos","")).strip()
                        role = p.get("_role", classify_player(p))
                        # Show specific pos if it adds detail beyond generic GK/DF/MF/FW
                        badge = spec if spec and spec.upper() not in ("G","D","M","F") else role
                        name  = str(p.get("Name","")).strip()
                        if not name or name in ("None","null"): continue  # skip null players
                        st.markdown(
                            f"<div class='player-card'>"
                            f"<span><span class='pos-badge'>{badge}</span>{name}{warn}</span>"
                            f"<span class='stat-text'>⏱ {p.get('Min',0)} mins &nbsp;⚽ {ga} G+A</span>"
                            f"</div>", unsafe_allow_html=True)
                else:
                    st.info(f"No player data loaded for **{my_team}**. "
                            f"Go to **Coach's Sandbox**, select {my_team}, and click "
                            f"'Fetch Squad from BSD' — it loads instantly and saves for all modules.")

# =============================================================================
# MODULE 2: OPPONENT ANALYSIS
# =============================================================================
elif app_mode == "📊 Pre-Match Opponent Analysis":
    st.markdown("## 📊 Pre-Match Opponent Analysis")
    st.write("Scout your opponent. Compare strengths and receive a tactical briefing.")

    col1, col2 = st.columns(2)
    with col1: my_team  = st.selectbox("Your Team", DROPDOWN_TEAMS, key="scout_my")
    with col2: opp_team = st.selectbox("Opponent",  DROPDOWN_TEAMS, index=1, key="scout_opp")

    if my_team == opp_team:
        st.error("🚨 A team cannot face itself!")
    else:
        fb  = teams_db.get(my_team,  {"Attack":80,"Defense":80})
        fb2 = teams_db.get(opp_team, {"Attack":80,"Defense":80})
        my_att, my_def   = fb.get("Attack",80),  fb.get("Defense",80)
        opp_att, opp_def = fb2.get("Attack",80), fb2.get("Defense",80)

        st.markdown("### ⚔️ Head-to-Head Comparison")
        c1, c2, c3 = st.columns([2,1,2])
        with c1:
            st.metric(f"{my_team} Attack",  my_att,  f"{my_att-opp_def:+d} vs Opp Def")
            st.metric(f"{my_team} Defence", my_def,  f"{my_def-opp_att:+d} vs Opp Att")
        with c2:
            st.markdown("<div style='text-align:center;margin-top:30px'><span style='font-family:Rajdhani,sans-serif;font-size:2rem;color:#6b8f72;font-weight:700'>VS</span></div>", unsafe_allow_html=True)
        with c3:
            st.metric(f"{opp_team} Attack",  opp_att,  f"{opp_att-my_def:+d} vs Our Def",  delta_color="inverse")
            st.metric(f"{opp_team} Defence", opp_def,  f"{opp_def-my_att:+d} vs Our Att", delta_color="inverse")

        st.markdown("---")
        st.markdown("### 📋 AI Pre-Match Briefing")
        lines = []
        if my_att > opp_def + 10:
            lines.append("🎯 **Offensive Dominance:** Their defence is significantly weaker than our attack. High line, press triggers early — suffocate them.")
        elif my_att < opp_def:
            lines.append("🧱 **Tough Defence Ahead:** Centrally they're solid. Exploit wide areas, overlaps, and set-pieces to unlock them.")
        if opp_att > my_def + 10:
            lines.append("⚠️ **Defensive Vulnerability:** Their attack is lethal vs our defence. Double pivot to screen the backline. No high line.")
        elif my_def > opp_att:
            lines.append("🛡️ **Defensive Superiority:** We handle their forwards comfortably. Full-backs can push aggressively without fear of the counter.")
        if abs(my_att-opp_att) <= 5 and abs(my_def-opp_def) <= 5:
            lines.append("⚖️ **Even Matchup:** This is decided in midfield transitions. Retain the ball, be patient, capitalise on unforced errors.")
        st.info("\n\n".join(lines) if lines else "No extreme mismatches detected. Play to your standard strengths and maintain structural discipline.")

# =============================================================================
# MODULE 3: COACH'S SANDBOX
# =============================================================================
elif app_mode == "🧠 Coach's Sandbox":
    import requests as _req, time as _time

    st.markdown("## 🧠 Coach's Sandbox")
    st.write("Pick your formation, draft your XI, and get an AI second opinion.")

    col1, col2 = st.columns(2)
    with col1: my_team  = st.selectbox("Your Team", DROPDOWN_TEAMS, key="sb_my")
    with col2: opp_team = st.selectbox("Opponent",  DROPDOWN_TEAMS, index=1, key="sb_opp")

    # ── On-demand squad fetcher for teams not in players.json ────────────────
    SQUAD_CACHE = "squad_cache.json"

    def load_squad_cache():
        try: return json.load(open(SQUAD_CACHE, encoding="utf-8"))
        except: return {}

    def save_squad_cache(d):
        try: json.dump(d, open(SQUAD_CACHE,"w",encoding="utf-8"), indent=2)
        except: pass

    SPEC_MAP = {
        "GK":"GK","CB":"DF","RB":"DF","LB":"DF","RWB":"DF","LWB":"DF",
        "CM":"MF","CDM":"MF","DM":"MF","CAM":"MF","AM":"MF",
        "RM":"FW","LM":"FW","RW":"FW","LW":"FW","RWF":"FW","LWF":"FW",
        "ST":"FW","CF":"FW","SS":"FW",
    }
    GEN_MAP = {"G":"GK","D":"DF","M":"MF","F":"FW"}

    def fetch_squad_on_demand(team_name, api_key):
        """
        Fetch squad from BSD for any team not in players.json.
        Caches to squad_cache.json for 7 days to avoid repeat calls.
        Uses GET /api/v2/players/?team_id={id}&limit=100
        """
        cache = load_squad_cache()
        entry = cache.get(team_name, {})
        age   = _time.time() - entry.get("fetched_at", 0)
        if entry and age < 604800:  # 7-day cache
            return entry.get("players", [])

        # Step 1: resolve team_id by name
        team_id, _ = bsd_find_team_id(team_name, api_key)
        if not team_id:
            return []

        # Step 2: fetch players (has specific_position)
        hdrs = {"Authorization": f"Token {api_key}"}
        try:
            r = _req.get(f"{BSD_BASE}/players/",
                         headers=hdrs,
                         params={"team_id": team_id, "limit": 100},
                         timeout=12)
            if r.status_code != 200:
                return []
        except:
            return []

        roster = []
        for p in r.json().get("results", []):
            name = p.get("name") or p.get("short_name","")
            if not name or name.strip() in ("","None","null"):
                continue
            spec = str(p.get("specific_position","")).strip().upper()
            gen  = str(p.get("position","M")).strip().upper()
            pos  = SPEC_MAP.get(spec) or GEN_MAP.get(gen, "MF")
            roster.append({
                "Name":    name.strip(),
                "Pos":     pos,
                "SpecPos": spec or gen,
                "Min":     0,
                "G_A":     0,
            })

        cache[team_name] = {"fetched_at": _time.time(), "players": roster}
        save_squad_cache(cache)

        # Also write into players_db so modules 1 & 5 benefit immediately
        players_db[team_name] = roster
        try:
            with open("players.json","w",encoding="utf-8") as f:
                json.dump(players_db, f, indent=2, ensure_ascii=False)
        except:
            pass

        return roster

    # ── Resolve roster for selected team ─────────────────────────────────────
    api_key    = st.secrets.get("BSD_API_KEY","")
    actual_key = my_team
    roster     = []

    if my_team in players_db:
        roster = [p for p in players_db[my_team]
                  if p.get("Name") and str(p["Name"]).strip() not in ("","None","null")]
    else:
        # Try fuzzy match in existing data first
        m = difflib.get_close_matches(my_team, players_db.keys(), n=1, cutoff=0.6)
        if m:
            actual_key = m[0]
            roster = [p for p in players_db[actual_key]
                      if p.get("Name") and str(p["Name"]).strip() not in ("","None","null")]

    # If still empty, offer on-demand BSD fetch
    if not roster:
        if api_key:
            st.info(f"ℹ️ No local data for **{my_team}**. Click below to fetch their squad from BSD now.")
            if st.button(f"📥 Fetch {my_team} Squad from BSD", key="fetch_squad_btn"):
                with st.spinner(f"Fetching {my_team} squad from BSD API..."):
                    roster = fetch_squad_on_demand(my_team, api_key)
                if roster:
                    st.success(f"✅ {len(roster)} players loaded for {my_team}!")
                    st.rerun()
                else:
                    st.error(f"🚨 Could not find {my_team} in BSD. Try a slightly different spelling.")
        else:
            st.warning(f"No player data for {my_team} and BSD_API_KEY not set. Add it to Streamlit Secrets.")

    if my_team == opp_team:
        st.error("🚨 A team cannot face itself!")
    elif roster:
        roster_names = [p["Name"] for p in roster]

        fc_col, sq_col = st.columns(2)
        with fc_col:
            coach_form = st.selectbox("Your Preferred Formation", list(formations_map.values()))
        with sq_col:
            coach_xi = st.multiselect("Draft Your Starting XI (max 11)", roster_names, max_selections=11)

        if st.button("⚙️ Analyze My Gameplan", use_container_width=True, type="primary"):
            if len(coach_xi) < 11:
                st.warning(f"⚠️ Only {len(coach_xi)}/11 players drafted.")
            fb  = teams_db.get(my_team,  {"Attack":80,"Defense":80})
            fb2 = teams_db.get(opp_team, {"Attack":80,"Defense":80})
            fc_code = list(formations_map.keys())[list(formations_map.values()).index(coach_form)]
            test = pd.DataFrame({"Formation":[fc_code],"Team_Attack":[fb["Attack"]],
                                  "Team_Defense":[fb["Defense"]],"Opp_Attack":[fb2["Attack"]],
                                  "Opp_Defense":[fb2["Defense"]]})
            prob = model.predict_proba(test)[0][1] * 100
            c1, c2 = st.columns(2)
            c1.metric("Your Formation", coach_form)
            c2.metric("AI Win Probability", f"{prob:.1f}%")
            if coach_xi:
                st.markdown("### 👕 Your Drafted XI")
                for name in coach_xi:
                    p_data = next((p for p in roster if p["Name"] == name), None)
                    # Use SpecPos badge if informative, else fall back to resolved Pos
                    if p_data:
                        spec  = str(p_data.get("SpecPos","")).strip()
                        pos   = spec if spec and spec.upper() not in ("G","D","M","F") else p_data.get("Pos","?")
                        ga    = p_data.get("G_A", 0)
                        mins  = p_data.get("Min", 0)
                    else:
                        pos, ga, mins = "?", 0, 0
                    st.markdown(
                        f"<div class='player-card'>"
                        f"<span><span class='pos-badge'>{pos}</span>{name}</span>"
                        f"<span class='stat-text'>⏱ {mins} mins &nbsp;⚽ {ga} G+A</span>"
                        f"</div>", unsafe_allow_html=True)

# =============================================================================
# MODULE 4: LIVE MATCH SIMULATOR
# =============================================================================
elif app_mode == "⏱️ Live Match Simulator":
    st.markdown("## ⏱️ Live Match Simulator")
    st.write("Simulate match scenarios in real time. Adjust your formation and see how the AI responds.")

    col1, col2 = st.columns(2)
    with col1: sim_my  = st.selectbox("Your Team",  DROPDOWN_TEAMS, key="sim_my")
    with col2: sim_opp = st.selectbox("Opponent",   DROPDOWN_TEAMS, index=1, key="sim_opp")

    if sim_my == sim_opp:
        st.error("🚨 A team cannot face itself!")
    else:
        s1, s2, s3 = st.columns(3)
        sim_min   = s1.slider("Match Minute", 0, 90, 45)
        home_g    = s2.number_input("Your Goals", 0, 20, 0)
        away_g    = s3.number_input("Opp Goals",  0, 20, 0)
        sim_form  = st.selectbox("Your Current Formation", list(formations_map.values()), key="sim_form")

        if st.button("🎯 Simulate & Get AI Recommendation", use_container_width=True, type="primary"):
            fb  = teams_db.get(sim_my,  {"Attack":80,"Defense":80})
            fb2 = teams_db.get(sim_opp, {"Attack":80,"Defense":80})
            fc_code = list(formations_map.keys())[list(formations_map.values()).index(sim_form)]
            prob = model.predict_proba(pd.DataFrame({"Formation":[fc_code],"Team_Attack":[fb["Attack"]],
                                                      "Team_Defense":[fb["Defense"]],"Opp_Attack":[fb2["Attack"]],
                                                      "Opp_Defense":[fb2["Defense"]]}))[0][1]*100

            if home_g > away_g:
                status_msg = f"🟢 Winning {home_g}–{away_g} at minute {sim_min}. Protect the lead."
            elif home_g < away_g:
                status_msg = f"🔴 Losing {home_g}–{away_g} at minute {sim_min}. Need to respond."
            else:
                status_msg = f"🟡 Level {home_g}–{away_g} at minute {sim_min}. Push for the winner."

            st.markdown(f"<div class='live-suggestion'><b>{status_msg}</b><br>Current formation win probability: <b>{prob:.1f}%</b></div>",
                        unsafe_allow_html=True)

            advice = []
            if home_g < away_g and sim_min > 60:
                advice.append("🔄 **Tactical Switch Needed:** You're behind with limited time. Switch to a more attacking formation (4-3-3 or 3-4-3) and sacrifice a midfielder for an extra forward.")
            if home_g > away_g and sim_min > 75:
                advice.append("🧱 **Hold the Line:** You're ahead late on. Drop to a 5-4-1 or 5-3-2. Absorb pressure, hit on the counter.")
            if home_g == away_g and sim_min > 80:
                advice.append("⚡ **Push for the Win:** All square with under 10 minutes left. Commit your fullbacks forward. High press. Take risks — a draw achieves little.")
            if not advice:
                advice.append(f"✅ **Maintain Shape:** At minute {sim_min} with the score at {home_g}–{away_g}, your current {sim_form} is well-suited. No change needed yet.")

            for a in advice:
                st.info(a)

# =============================================================================
# MODULE 5: ASSISTANT MANAGER CHAT
# =============================================================================
elif app_mode == "💬 Assistant Manager Chat":
    import requests as _req, time as _time

    st.markdown("## 💬 Assistant Manager Chat")
    st.write("Select your teams, sync live match data from any competition worldwide, "
             "then chat with your AI assistant manager.")

    col1, col2 = st.columns(2)
    with col1:
        chat_my  = st.selectbox("Your Team", DROPDOWN_TEAMS, key="chat_my",
                                index=DROPDOWN_TEAMS.index("Arsenal") if "Arsenal" in DROPDOWN_TEAMS else 0)
    with col2:
        chat_opp = st.selectbox("Opponent",  DROPDOWN_TEAMS, key="chat_opp",
                                index=DROPDOWN_TEAMS.index("Chelsea") if "Chelsea" in DROPDOWN_TEAMS else 1)

    st.markdown("---")
    st.markdown("### 📡 Live Match Intel")

    LIVE_CACHE = "live_match_cache.json"

    def load_lc():
        try: return json.load(open(LIVE_CACHE, encoding="utf-8"))
        except: return {}

    def save_lc(d):
        try: json.dump(d, open(LIVE_CACHE,"w",encoding="utf-8"), indent=2)
        except: pass

    def ck(a, b): return f"{a.lower().strip()}__vs__{b.lower().strip()}"

    lc      = load_lc()
    ckey    = ck(chat_my, chat_opp)
    cached  = lc.get(ckey, {})
    c_age   = _time.time() - cached.get("fetched_at", 0)
    c_fresh = c_age < 300

    if c_fresh and cached.get("match_found"):
        d = cached
        st.markdown(
            f"<div class='live-suggestion'>"
            f"<b>✅ LIVE: {d['home_name']} {d['home_goals']} – {d['away_goals']} {d['away_name']}</b>"
            f"&nbsp;|&nbsp; ⏱️ {d['minute']}' &nbsp;|&nbsp; 🏆 {d['competition']}"
            f"<br><span style='font-size:12px;color:#6b8f72'>Data cached {int(c_age/60)} min ago</span>"
            f"</div>", unsafe_allow_html=True)
        st.session_state.live_context = (
            f"LIVE ({d['competition']}): Minute {d['minute']}'. "
            f"Score: {d['home_name']} {d['home_goals']} – {d['away_goals']} {d['away_name']}."
        )
    elif cached and not cached.get("match_found"):
        st.info(f"ℹ️ Last sync: {chat_my} not in a live fixture. Hit Sync to recheck.")
    else:
        st.info("No live data cached for this fixture. Hit **Sync** to scan all live competitions.")
        if "live_context" not in st.session_state:
            st.session_state.live_context = "No live data. Provide general pre-match tactical advice."

    # ── Sync button ─────────────────────────────────────────────────────────
    last_sync  = st.session_state.get("last_bsd_sync", 0)
    secs_since = _time.time() - last_sync
    COOLDOWN   = 30  # BSD Redis TTL is 30s — no point polling faster
    sync_ready = secs_since >= COOLDOWN
    btn_label  = ("🔄 Sync Live Data (All Competitions Worldwide)"
                  if sync_ready else
                  f"🔄 Sync (cooldown: {max(1, int(COOLDOWN-secs_since))}s)")

    if st.button(btn_label, use_container_width=True, disabled=not sync_ready):
        bsd_key = st.secrets.get("BSD_API_KEY")
        if not bsd_key:
            st.error("🚨 BSD_API_KEY missing from Streamlit Secrets!")
        else:
            with st.spinner("🌐 Scanning all live fixtures worldwide..."):
                hdrs = {"Authorization": f"Token {bsd_key}"}
                try:
                    # GET /api/v2/events/live/
                    # Response: {"count": N, "events": [...]}
                    # Fields: home_team, away_team, home_score, away_score,
                    #         current_minute, league_name, status
                    res = _req.get(f"{BSD_BASE}/events/live/", headers=hdrs, timeout=12)
                    st.session_state.last_bsd_sync = _time.time()

                    if res.status_code != 200:
                        st.error(f"🚨 BSD API error {res.status_code}. Check your key in Streamlit Secrets.")
                    else:
                        live_data   = res.json().get("events", [])
                        match_found = False

                        for match in live_data:
                            hn = match.get("home_team","")
                            an = match.get("away_team","")
                            my_hit  = (chat_my.lower() in hn.lower() or hn.lower() in chat_my.lower() or
                                       chat_my.lower() in an.lower() or an.lower() in chat_my.lower())
                            opp_hit = (chat_opp.lower() in hn.lower() or hn.lower() in chat_opp.lower() or
                                       chat_opp.lower() in an.lower() or an.lower() in chat_opp.lower())

                            if my_hit and opp_hit:
                                minute     = match.get("current_minute") or 0
                                home_goals = match.get("home_score") or 0
                                away_goals = match.get("away_score") or 0
                                competition= match.get("league_name","Unknown Competition")
                                status_txt = match.get("status","inprogress")

                                entry = {"fetched_at":_time.time(),"match_found":True,
                                         "home_name":hn,"away_name":an,
                                         "home_goals":home_goals,"away_goals":away_goals,
                                         "minute":minute,"competition":competition}
                                lc[ckey] = entry; save_lc(lc)

                                st.session_state.live_context = (
                                    f"LIVE ({competition}): Minute {minute}'. "
                                    f"Score: {hn} {home_goals} – {away_goals} {an}. "
                                    f"Status: {status_txt}.")
                                st.markdown(
                                    f"<div class='live-suggestion'>"
                                    f"<b>✅ LIVE: {hn} {home_goals} – {away_goals} {an}</b>"
                                    f"&nbsp;|&nbsp; ⏱️ {minute}' &nbsp;|&nbsp; 🏆 {competition}"
                                    f"</div>", unsafe_allow_html=True)
                                match_found = True; break

                        if not match_found:
                            lc[ckey] = {"fetched_at":_time.time(),"match_found":False}
                            save_lc(lc)
                            st.session_state.live_context = "No live match. Provide pre-match tactical advice."
                            st.warning(f"⚠️ No live fixture found for **{chat_my}** vs **{chat_opp}**. "
                                       f"Checked {len(live_data)} live matches worldwide.")
                except Exception as e:
                    st.error(f"🚨 Connection error: {e}")

    if cached:
        if st.button("🗑️ Clear cached data for this fixture", key="clr_live"):
            lc.pop(ckey,None); save_lc(lc)
            st.session_state.pop("live_context",None)
            st.rerun()

    st.markdown("---")

    # ── AI Chat ──────────────────────────────────────────────────────────────
    st.markdown("### 🧠 Assistant Manager")

    chat_key = f"msgs__{ckey}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(f"Ask your assistant... e.g. 'How do we beat {chat_opp}?'"):
        st.session_state[chat_key].append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        roster      = players_db.get(chat_my, [])
        live_status = st.session_state.get("live_context", "No live data. Provide pre-match tactical advice.")
        history     = "\n".join(
            f"{'Coach' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in st.session_state[chat_key][:-1]
        )

        system_prompt = f"""You are an elite AI Assistant Football Manager.
You assist the Head Coach of {chat_my}, currently facing {chat_opp}.

LIVE MATCH STATUS:
{live_status}

OUR SQUAD (Name | Pos | Minutes | Goals+Assists):
{json.dumps(roster, ensure_ascii=False)}

CONVERSATION HISTORY:
{history or "Start of briefing."}

INSTRUCTIONS:
- Speak directly to the Head Coach. Concise, tactical, professional.
- If LIVE MATCH DATA present, anchor ALL advice to the current score and minute.
- No live data → sharp pre-match tactical advice only.
- Only reference players from our squad above. Never invent names.
- Use football terminology: press triggers, half-spaces, double pivot, low block, etc.
- 3–6 sentences unless a detailed breakdown is explicitly requested.
"""
        if gemini_api_key:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                try:
                    resp  = ai_model.generate_content(f"{system_prompt}\n\nCoach: {prompt}")
                    reply = resp.text
                    placeholder.markdown(reply)
                    st.session_state[chat_key].append({"role":"assistant","content":reply})
                except Exception as e:
                    placeholder.error(f"🚨 Gemini error: {e}")
        else:
            st.error("🚨 GEMINI_API_KEY missing from Streamlit Secrets.")

    if st.session_state.get(chat_key):
        if st.button("🔁 Reset Chat", key="reset_chat"):
            st.session_state[chat_key] = []
            st.rerun()
