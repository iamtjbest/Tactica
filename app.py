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

# ── ALL EUROPEAN TEAMS ─────────────────────────────────────────────────────────
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

DROPDOWN_TEAMS = sorted(set(ALL_EUROPEAN_TEAMS) | set(teams_db.keys()))

# ── Position helpers ──────────────────────────────────────────────────────────
WIDE_ATT_SPECS = {"RM","LM","RW","LW","RWF","LWF","AM","CAM","SS","WF","W"}

def classify_player(p):
    pos  = str(p.get("Pos", "MF")).upper()
    spec = str(p.get("SpecPos", "")).strip().upper()
    if spec in WIDE_ATT_SPECS: return "WIDE_ATT"
    if "GK" in pos: return "GK"
    if "DF" in pos: return "DF"
    if "FW" in pos: return "FW"
    return "MF"

def get_display_pos(p, tactical_role):
    return tactical_role if tactical_role else classify_player(p)

def select_starting_xi(team_name, formation):
    if team_name not in players_db:
        close = difflib.get_close_matches(team_name, players_db.keys(), n=1, cutoff=0.6)
        if close: team_name = close[0]
        else: return None

    parts     = [int(x) for x in re.findall(r"\d+", formation)]
    def_count = parts[0]
    att_count = parts[-1]
    mid_count = sum(parts[1:-1]) if len(parts) > 2 else parts[1]

    clean = [p for p in players_db[team_name]
             if p.get("Name") and str(p["Name"]).strip() not in ("", "None", "null")]

    SPEC_OVERRIDE = {
        "GK":"GK", "CB":"DF","RB":"DF","LB":"DF","RWB":"DF","LWB":"DF",
        "CM":"MF","CDM":"MF","DM":"MF","CAM":"MF","AM":"MF",
        "RM":"FW","LM":"FW","RW":"FW","LW":"FW","RWF":"FW","LWF":"FW",
        "ST":"FW","CF":"FW","SS":"FW",
    }
    resolved = []
    for p in clean:
        p2 = dict(p)
        spec = str(p2.get("SpecPos","")).strip().upper()
        if spec and spec in SPEC_OVERRIDE: p2["Pos"] = SPEC_OVERRIDE[spec]
        resolved.append(p2)

    roster = sorted(resolved, key=lambda x: (x.get("Min",0), x.get("G_A",0)), reverse=True)

    buckets = {"GK":[], "DF":[], "MF":[], "FW":[], "WIDE_ATT":[]}
    for p in roster: buckets[classify_player(p)].append(p)

    xi, used = [], set()

    def pick(pool, n, role):
        drafted = 0
        for p in pool:
            if drafted >= n: break
            if p["Name"] not in used:
                tagged = dict(p); tagged["_role"] = role
                xi.append(tagged); used.add(p["Name"]); drafted += 1
        return drafted

    if pick(buckets["GK"], 1, "GK") < 1: pick(roster, 1, "GK")

    df_pool = sorted(buckets["DF"], key=lambda x: (x["Min"], x["G_A"]), reverse=True)
    n = pick(df_pool, def_count, "DF")
    if n < def_count:
        df_hybrid = [p for p in roster if "DF" in p.get("Pos","") and p["Name"] not in used]
        pick(df_hybrid, def_count - n, "DF")

    if att_count >= 3:
        fw_pool = sorted(buckets["FW"] + buckets["WIDE_ATT"], key=lambda x: (x["Min"], x["G_A"]), reverse=True)
        mf_wide_reserve = []
    else:
        fw_pool         = sorted(buckets["FW"], key=lambda x: (x["Min"], x["G_A"]), reverse=True)
        mf_wide_reserve = sorted(buckets["WIDE_ATT"], key=lambda x: (x["Min"], x["G_A"]), reverse=True)

    n_fw = pick(fw_pool, att_count, "FW")
    if n_fw < att_count:
        remaining_wide = [p for p in buckets["WIDE_ATT"] if p["Name"] not in used]
        pick(sorted(remaining_wide, key=lambda x: (x["Min"], x["G_A"]), reverse=True), att_count - n_fw, "FW")

    mf_pool = sorted(buckets["MF"] + mf_wide_reserve, key=lambda x: (x["Min"], x["G_A"]), reverse=True)
    n_mf = pick(mf_pool, mid_count, "MF")
    if n_mf < mid_count:
        remaining_wide = [p for p in buckets["WIDE_ATT"] if p["Name"] not in used]
        pick(sorted(remaining_wide, key=lambda x: (x["Min"], x["G_A"]), reverse=True), mid_count - n_mf, "MF")

    for p in roster:
        if len(xi) >= 11: break
        if p["Name"] not in used:
            tagged = dict(p); tagged["_role"] = "?"; tagged["fallback"] = True
            xi.append(tagged); used.add(p["Name"])

    return xi

# ── BSD helpers ───────────────────────────────────────────────────────────────
BSD_BASE = "https://sports.bzzoiro.com/api/v2"

# Updated Mapping
LEAGUE_NAMES = {
    1: "Premier League", 2: "Champions League", 3: "Europa League", 
    4: "Ligue 1", 5: "Bundesliga", 8: "La Liga", 11: "Serie A", 
    17: "Premier League", 39: "Scottish Premiership", 40: "FA Cup / EFL Cup",
    88: "Eredivisie", 94: "Primeira Liga", 144: "Belgian Pro League"
}

def bsd_find_team_id(team_name, api_key):
    import requests as _r
    hdrs = {"Authorization": f"Token {api_key}"}
    try:
        res = _r.get(f"{BSD_BASE}/teams/", headers=hdrs, params={"name": team_name, "limit": 3}, timeout=10)
        if res.status_code == 200:
            results = res.json().get("results", [])
            if results:
                best = difflib.get_close_matches(team_name.lower(), [t["name"].lower() for t in results], n=1, cutoff=0.4)
                if best:
                    for t in results:
                        if t["name"].lower() == best[0]: return t["id"], t["name"]
                return results[0]["id"], results[0]["name"]
    except Exception: pass
    return None, None

def clear_cache_silently():
    for f in ["nat_form_cache.json", "squad_cache.json", "form_cache.json"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## NAVIGATION")
app_mode = st.sidebar.radio("Select Module:", [
    "🤖 Pre-Match Auto-Tactics",
    "📊 Pre-Match Opponent Analysis",
    "🧠 Coach's Sandbox",
    "⏱️ Live Match Simulator",
    "💬 Assistant Manager Chat",
    "🏆 World Cup Scout"
])

# =============================================================================
# MODULE 1: PRE-MATCH AUTO-TACTICS
# =============================================================================
if app_mode == "🤖 Pre-Match Auto-Tactics":
    import requests as _req, time as _time

    st.markdown("## 🤖 Pre-Match Auto-Tactics")
    st.write("Select any two European teams. The engine fetches their last 5 matches via BSD API, extracts real formations, calculates live attack/defence ratings, then recommends the optimal game plan.")

    col1, col2 = st.columns(2)
    with col1:
        my_team  = st.selectbox("Your Team",  DROPDOWN_TEAMS, index=DROPDOWN_TEAMS.index("Arsenal") if "Arsenal" in DROPDOWN_TEAMS else 0)
    with col2:
        opp_team = st.selectbox("Opponent",   DROPDOWN_TEAMS, index=DROPDOWN_TEAMS.index("Chelsea") if "Chelsea" in DROPDOWN_TEAMS else 1)

    def fetch_last5(team_name, api_key):
        team_id, matched_name = bsd_find_team_id(team_name, api_key)
        if not team_id: return [], False

        hdrs = {"Authorization": f"Token {api_key}"}
        from datetime import datetime, timedelta
        date_from = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%dT00:00:00Z")
        
        try:
            # Fetch limit 50 to ensure we grab enough recent history, then slice the last 5
            r = _req.get(f"{BSD_BASE}/teams/{team_id}/fixtures/", headers=hdrs,
                         params={"status":"finished", "limit":50, "date_from":date_from}, timeout=12)
        except: return [], False
        if r.status_code != 200: return [], False

        fixtures = r.json().get("results", [])
        
        # Take the most recent 5 and reverse to show newest first
        fixtures = fixtures[-5:]
        fixtures.reverse()
        
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
            lname = fix.get("league", {}).get("name") or fix.get("league_name")
            competition = lname if lname else LEAGUE_NAMES.get(league_id, f"Competition {league_id}")
            result     = "W" if scored > conceded else ("D" if scored == conceded else "L")

            formation_used = "Unknown"
            try:
                lr = _req.get(f"{BSD_BASE}/events/{fid}/lineups/", headers=hdrs, timeout=12)
                _time.sleep(0.2)
                if lr.status_code == 200:
                    ld = lr.json()
                    lu_status = ld.get("lineup_status", "unavailable")
                    lineups   = ld.get("lineups")
                    if lu_status != "unavailable" and lineups:
                        side = "home" if is_home else "away"
                        formation_used = (lineups.get(side) or {}).get("formation") or "Unknown"
            except: pass

            results.append({
                "fixture_id":  fid, "formation": formation_used, "scored": scored,
                "conceded": conceded, "result": result, "competition": competition, "opponent": opp_name,
            })
        return results, False

    def compute_ratings(last5):
        if not last5: return None, None
        avg_s = sum(m["scored"] for m in last5) / len(last5)
        avg_c = sum(m["conceded"] for m in last5) / len(last5)
        return min(99, int(60 + avg_s * 9.75)), max(60, min(99, int(99 - avg_c * 9.75)))

    def best_formation(last5):
        counts = {}
        for m in last5:
            f = m.get("formation","Unknown")
            if f and f != "Unknown": counts[f] = counts.get(f,0)+1
        return max(counts, key=counts.get) if counts else None

    api_key = st.secrets.get("BSD_API_KEY", "")

    with st.expander("ℹ️ API Usage Info"):
        st.markdown("**Fresh fetch costs ~12 BSD API calls** (name lookup + fixtures + 5 lineups per team).")

    if st.button("🔍 Fetch Last 5 Matches & Generate Optimal Tactics", use_container_width=True, type="primary"):
        clear_cache_silently()
        if my_team == opp_team:
            st.error("🚨 A team cannot face itself!")
        elif not api_key:
            st.error("🚨 BSD_API_KEY missing from Streamlit Secrets.")
        else:
            with st.spinner(f"📡 Fetching last 5 matches for {my_team} and {opp_team}..."):
                my5,  my_cached  = fetch_last5(my_team,  api_key)
                opp5, opp_cached = fetch_last5(opp_team, api_key)

            if not my5 and not opp5:
                st.error("🚨 No match data returned.")
            else:
                my_att_d,  my_def_d  = compute_ratings(my5)
                opp_att_d, opp_def_d = compute_ratings(opp5)

                fb = teams_db.get(my_team,  {"Attack":80,"Defense":80})
                fb2= teams_db.get(opp_team, {"Attack":80,"Defense":80})
                my_att  = my_att_d  or fb.get("Attack",  80)
                my_def  = my_def_d  or fb.get("Defense", 80)
                opp_att = opp_att_d or fb2.get("Attack",  80)
                opp_def = opp_def_d or fb2.get("Defense", 80)

                opp_habit = best_formation(opp5)
                my_habit  = best_formation(my5)

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

                st.markdown("---")
                r1, r2, r3 = st.columns(3)
                r1.metric("✅ Recommended Formation", best_form)
                r2.metric("🤖 AI Win Probability",    f"{best_prob:.1f}%")
                r3.metric("📐 Opp. Usual Formation",  opp_habit or "Unknown")

                st.markdown("### 📋 Last 5 Matches")
                fc1, fc2 = st.columns(2)

                def render_form(tname, last5, att, dfn, cached):
                    st.markdown(f"**{tname}** <span style='font-size:12px;color:#6b8f72'>(🔴 live)</span>", unsafe_allow_html=True)
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

                with fc1: render_form(my_team,  my5,  my_att,  my_def,  False)
                with fc2: render_form(opp_team, opp5, opp_att, opp_def, False)

                st.markdown("### 🏆 Formation Win-Probability Ranking")
                for rank, (fname, score) in enumerate(sorted(all_scores.items(), key=lambda x:x[1], reverse=True)[:5], 1):
                    medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][rank-1]
                    bar   = int((score/100)*280)
                    st.markdown(f"{medal} **{fname}** &nbsp;<span style='display:inline-block;width:{bar}px;height:8px;background:linear-gradient(90deg,#22c55e,#4ade80);border-radius:4px;vertical-align:middle'></span>&nbsp; {score}%", unsafe_allow_html=True)

                st.markdown(f"### 👕 Recommended Starting XI — {best_form}")
                xi = select_starting_xi(my_team, best_form)
                if xi:
                    for p in xi:
                        warn = " ⚠️" if p.get("fallback") else ""
                        ga   = f"{p['G_A']:.2f}" if isinstance(p['G_A'], float) else str(p['G_A'])
                        spec = str(p.get("SpecPos","")).strip()
                        role = p.get("_role", classify_player(p))
                        badge = spec if spec and spec.upper() not in ("G","D","M","F") else role
                        name  = str(p.get("Name","")).strip()
                        if not name or name in ("None","null"): continue
                        st.markdown(f"<div class='player-card'><span><span class='pos-badge'>{badge}</span>{name}{warn}</span><span class='stat-text'>⏱ {p.get('Min',0)} mins &nbsp;⚽ {ga} G+A</span></div>", unsafe_allow_html=True)
                else:
                    st.info(f"No player data loaded for **{my_team}**.")

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

    def fetch_squad_on_demand(team_name, api_key):
        team_id, _ = bsd_find_team_id(team_name, api_key)
        if not team_id: return []
        hdrs = {"Authorization": f"Token {api_key}"}
        try:
            r = _req.get(f"{BSD_BASE}/players/", headers=hdrs, params={"team_id": team_id, "limit": 100}, timeout=12)
            if r.status_code != 200: return []
        except: return []

        roster = []
        for p in r.json().get("results", []):
            name = p.get("name") or p.get("short_name","")
            if not name or name.strip() in ("","None","null"): continue
            spec = str(p.get("specific_position","")).strip().upper()
            gen  = str(p.get("position","M")).strip().upper()
            pos  = SPEC_MAP.get(spec) or GEN_MAP.get(gen, "MF")
            roster.append({"Name": name.strip(), "Pos": pos, "SpecPos": spec or gen, "Min": 0, "G_A": 0})
        
        players_db[team_name] = roster
        try:
            with open("players.json","w",encoding="utf-8") as f: json.dump(players_db, f, indent=2, ensure_ascii=False)
        except: pass
        return roster

    api_key    = st.secrets.get("BSD_API_KEY","")
    actual_key = my_team
    roster     = []

    if my_team in players_db:
        roster = [p for p in players_db[my_team] if p.get("Name") and str(p["Name"]).strip() not in ("","None","null")]
    else:
        m = difflib.get_close_matches(my_team, players_db.keys(), n=1, cutoff=0.6)
        if m:
            actual_key = m[0]
            roster = [p for p in players_db[actual_key] if p.get("Name") and str(p["Name"]).strip() not in ("","None","null")]

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
                    st.error(f"🚨 Could not find {my_team} in BSD.")
        else:
            st.warning(f"No player data for {my_team} and BSD_API_KEY not set.")

    if my_team == opp_team:
        st.error("🚨 A team cannot face itself!")
    elif roster:
        roster_names = [p["Name"] for p in roster]
        fc_col, sq_col = st.columns(2)
        with fc_col: coach_form = st.selectbox("Your Preferred Formation", list(formations_map.values()))
        with sq_col: coach_xi = st.multiselect("Draft Your Starting XI (max 11)", roster_names, max_selections=11)

        if st.button("⚙️ Analyze My Gameplan", use_container_width=True, type="primary"):
            if len(coach_xi) < 11: st.warning(f"⚠️ Only {len(coach_xi)}/11 players drafted.")
            fb  = teams_db.get(my_team,  {"Attack":80,"Defense":80})
            fb2 = teams_db.get(opp_team, {"Attack":80,"Defense":80})
            fc_code = list(formations_map.keys())[list(formations_map.values()).index(coach_form)]
            test = pd.DataFrame({"Formation":[fc_code],"Team_Attack":[fb["Attack"]], "Team_Defense":[fb["Defense"]],"Opp_Attack":[fb2["Attack"]], "Opp_Defense":[fb2["Defense"]]})
            prob = model.predict_proba(test)[0][1] * 100
            c1, c2 = st.columns(2)
            c1.metric("Your Formation", coach_form)
            c2.metric("AI Win Probability", f"{prob:.1f}%")
            if coach_xi:
                st.markdown("### 👕 Your Drafted XI")
                for name in coach_xi:
                    p_data = next((p for p in roster if p["Name"] == name), None)
                    if p_data:
                        spec  = str(p_data.get("SpecPos","")).strip()
                        pos   = spec if spec and spec.upper() not in ("G","D","M","F") else p_data.get("Pos","?")
                        ga, mins = p_data.get("G_A", 0), p_data.get("Min", 0)
                    else: pos, ga, mins = "?", 0, 0
                    st.markdown(f"<div class='player-card'><span><span class='pos-badge'>{pos}</span>{name}</span><span class='stat-text'>⏱ {mins} mins &nbsp;⚽ {ga} G+A</span></div>", unsafe_allow_html=True)

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
            prob = model.predict_proba(pd.DataFrame({"Formation":[fc_code],"Team_Attack":[fb["Attack"]], "Team_Defense":[fb["Defense"]],"Opp_Attack":[fb2["Attack"]], "Opp_Defense":[fb2["Defense"]]}))[0][1]*100

            if home_g > away_g: status_msg = f"🟢 Winning {home_g}–{away_g} at minute {sim_min}. Protect the lead."
            elif home_g < away_g: status_msg = f"🔴 Losing {home_g}–{away_g} at minute {sim_min}. Need to respond."
            else: status_msg = f"🟡 Level {home_g}–{away_g} at minute {sim_min}. Push for the winner."

            st.markdown(f"<div class='live-suggestion'><b>{status_msg}</b><br>Current formation win probability: <b>{prob:.1f}%</b></div>", unsafe_allow_html=True)

            advice = []
            if home_g < away_g and sim_min > 60: advice.append("🔄 **Tactical Switch Needed:** You're behind with limited time. Switch to a more attacking formation (4-3-3 or 3-4-3) and sacrifice a midfielder for an extra forward.")
            if home_g > away_g and sim_min > 75: advice.append("🧱 **Hold the Line:** You're ahead late on. Drop to a 5-4-1 or 5-3-2. Absorb pressure, hit on the counter.")
            if home_g == away_g and sim_min > 80: advice.append("⚡ **Push for the Win:** All square with under 10 minutes left. Commit your fullbacks forward. High press. Take risks — a draw achieves little.")
            if not advice: advice.append(f"✅ **Maintain Shape:** At minute {sim_min} with the score at {home_g}–{away_g}, your current {sim_form} is well-suited.")

            for a in advice: st.info(a)

# =============================================================================
# MODULE 5: ASSISTANT MANAGER CHAT
# =============================================================================
elif app_mode == "💬 Assistant Manager Chat":
    import requests as _req, time as _time

    st.markdown("## 💬 Assistant Manager Chat")
    st.write("Select your teams, sync live match data from any competition worldwide, then chat with your AI assistant manager.")

    col1, col2 = st.columns(2)
    with col1: chat_my  = st.selectbox("Your Team", DROPDOWN_TEAMS, key="chat_my", index=DROPDOWN_TEAMS.index("Arsenal") if "Arsenal" in DROPDOWN_TEAMS else 0)
    with col2: chat_opp = st.selectbox("Opponent",  DROPDOWN_TEAMS, key="chat_opp", index=DROPDOWN_TEAMS.index("Chelsea") if "Chelsea" in DROPDOWN_TEAMS else 1)

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
        st.markdown(f"<div class='live-suggestion'><b>✅ LIVE: {d['home_name']} {d['home_goals']} – {d['away_goals']} {d['away_name']}</b>&nbsp;|&nbsp; ⏱️ {d['minute']}' &nbsp;|&nbsp; 🏆 {d['competition']}<br><span style='font-size:12px;color:#6b8f72'>Data cached {int(c_age/60)} min ago</span></div>", unsafe_allow_html=True)
        st.session_state.live_context = f"LIVE ({d['competition']}): Minute {d['minute']}'. Score: {d['home_name']} {d['home_goals']} – {d['away_goals']} {d['away_name']}."
    elif cached and not cached.get("match_found"):
        st.info(f"ℹ️ Last sync: {chat_my} not in a live fixture. Hit Sync to recheck.")
    else:
        st.info("No live data cached for this fixture. Hit **Sync** to scan all live competitions.")
        if "live_context" not in st.session_state: st.session_state.live_context = "No live data. Provide general pre-match tactical advice."

    last_sync  = st.session_state.get("last_bsd_sync", 0)
    secs_since = _time.time() - last_sync
    COOLDOWN   = 30
    sync_ready = secs_since >= COOLDOWN
    btn_label  = "🔄 Sync Live Data (All Competitions Worldwide)" if sync_ready else f"🔄 Sync (cooldown: {max(1, int(COOLDOWN-secs_since))}s)"

    if st.button(btn_label, use_container_width=True, disabled=not sync_ready):
        bsd_key = st.secrets.get("BSD_API_KEY")
        if not bsd_key:
            st.error("🚨 BSD_API_KEY missing from Streamlit Secrets!")
        else:
            with st.spinner("🌐 Scanning all live fixtures worldwide..."):
                hdrs = {"Authorization": f"Token {bsd_key}"}
                try:
                    res = _req.get(f"{BSD_BASE}/events/live/", headers=hdrs, timeout=12)
                    st.session_state.last_bsd_sync = _time.time()
                    if res.status_code != 200:
                        st.error(f"🚨 BSD API error {res.status_code}.")
                    else:
                        live_data   = res.json().get("events", [])
                        match_found = False
                        for match in live_data:
                            hn, an = match.get("home_team",""), match.get("away_team","")
                            my_hit  = (chat_my.lower() in hn.lower() or hn.lower() in chat_my.lower() or chat_my.lower() in an.lower() or an.lower() in chat_my.lower())
                            opp_hit = (chat_opp.lower() in hn.lower() or hn.lower() in chat_opp.lower() or chat_opp.lower() in an.lower() or an.lower() in chat_opp.lower())
                            if my_hit and opp_hit:
                                minute, home_goals, away_goals = match.get("current_minute") or 0, match.get("home_score") or 0, match.get("away_score") or 0
                                competition, status_txt = match.get("league_name","Unknown Competition"), match.get("status","inprogress")
                                lc[ckey] = {"fetched_at":_time.time(),"match_found":True, "home_name":hn,"away_name":an, "home_goals":home_goals,"away_goals":away_goals, "minute":minute,"competition":competition}
                                save_lc(lc)
                                st.session_state.live_context = f"LIVE ({competition}): Minute {minute}'. Score: {hn} {home_goals} – {away_goals} {an}. Status: {status_txt}."
                                st.markdown(f"<div class='live-suggestion'><b>✅ LIVE: {hn} {home_goals} – {away_goals} {an}</b>&nbsp;|&nbsp; ⏱️ {minute}' &nbsp;|&nbsp; 🏆 {competition}</div>", unsafe_allow_html=True)
                                match_found = True; break
                        if not match_found:
                            lc[ckey] = {"fetched_at":_time.time(),"match_found":False}
                            save_lc(lc)
                            st.session_state.live_context = "No live match. Provide pre-match tactical advice."
                            st.warning(f"⚠️ No live fixture found for **{chat_my}** vs **{chat_opp}**. Checked {len(live_data)} matches.")
                except Exception as e: st.error(f"🚨 Connection error: {e}")

    if cached and st.button("🗑️ Clear cached data for this fixture"):
        lc.pop(ckey,None); save_lc(lc)
        st.session_state.pop("live_context",None); st.rerun()

    st.markdown("---")
    st.markdown("### 🧠 Assistant Manager")

    chat_key = f"msgs__{ckey}"
    if chat_key not in st.session_state: st.session_state[chat_key] = []
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input(f"Ask your assistant... e.g. 'How do we beat {chat_opp}?'"):
        st.session_state[chat_key].append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)

        roster      = players_db.get(chat_my, [])
        live_status = st.session_state.get("live_context", "No live data. Provide pre-match tactical advice.")
        history     = "\n".join(f"{'Coach' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in st.session_state[chat_key][:-1])

        system_prompt = f"You are an elite AI Assistant Football Manager. You assist the Head Coach of {chat_my}, currently facing {chat_opp}.\nLIVE MATCH STATUS:\n{live_status}\nOUR SQUAD:\n{json.dumps(roster, ensure_ascii=False)}\nHISTORY:\n{history}\nINSTRUCTIONS: Speak directly to the Head Coach. Concise, tactical, professional. Anchor advice to current score if live. Only use real players from the roster."
        
        if gemini_api_key:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                try:
                    resp  = ai_model.generate_content(f"{system_prompt}\n\nCoach: {prompt}")
                    reply = resp.text
                    placeholder.markdown(reply)
                    st.session_state[chat_key].append({"role":"assistant","content":reply})
                except Exception as e: placeholder.error(f"🚨 Gemini error: {e}")
        else: st.error("🚨 GEMINI_API_KEY missing from Streamlit Secrets.")

    if st.session_state.get(chat_key) and st.button("🔁 Reset Chat"):
        st.session_state[chat_key] = []; st.rerun()

# =============================================================================
# MODULE 6: WORLD Cup Scout
# =============================================================================
elif app_mode == "🏆 World Cup Scout":
    import requests as _req, time as _time
    
    st.markdown("## 🏆 World Cup 2026 Scout Engine")
    st.write("Analyze international matchups. Extracts real formations from recent fixtures (including friendlies) to recommend the optimal game plan and starting XI.")

    WC_NATIONS = sorted([
        "Mexico", "South Africa", "South Korea", "Czechia", "Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland",
        "Brazil", "Morocco", "Haiti", "Scotland", "United States", "Paraguay", "Australia", "Türkiye",
        "Germany", "Curaçao", "Ivory Coast", "Ecuador", "Netherlands", "Japan", "Sweden", "Tunisia",
        "Belgium", "Egypt", "Iran", "New Zealand", "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
        "France", "Senegal", "Iraq", "Norway", "Argentina", "Algeria", "Austria", "Jordan",
        "Portugal", "DR Congo", "Uzbekistan", "Colombia", "England", "Croatia", "Ghana", "Panama"
    ])
    NATION_DROPDOWN = sorted(set(WC_NATIONS) | {k for k in teams_db.keys() if k in WC_NATIONS})

    col1, col2 = st.columns(2)
    with col1: home_team = st.selectbox("🏠 Your Nation", NATION_DROPDOWN, index=NATION_DROPDOWN.index("Nigeria") if "Nigeria" in NATION_DROPDOWN else 0)
    with col2: away_team = st.selectbox("✈️ Opponent Nation", NATION_DROPDOWN, index=NATION_DROPDOWN.index("South Africa") if "South Africa" in NATION_DROPDOWN else 1)
        
    st.markdown("---")

    def get_strict_nation_id(team_name, api_key):
        hdrs = {"Authorization": f"Token {api_key}"}
        try:
            res = _req.get(f"{BSD_BASE}/teams/", headers=hdrs, params={"name": team_name, "limit": 10}, timeout=10)
            if res.status_code == 200:
                results = res.json().get("results", [])
                for t in results:
                    if t["name"].strip().lower() == team_name.lower(): return t["id"], t["name"]
                for t in results:
                    n_lower = t["name"].lower()
                    if team_name.lower() in n_lower and " u" not in n_lower and " w" not in n_lower: return t["id"], t["name"]
                if results: return results[0]["id"], results[0]["name"]
        except: pass
        return None, None

    def fetch_nation_last5(team_name, api_key):
        team_id, matched_name = get_strict_nation_id(team_name, api_key)
        if not team_id: return [], False, None

        hdrs = {"Authorization": f"Token {api_key}"}
        try:
            from datetime import datetime, timedelta
            # Pull 2 full years to guarantee we find their most recent 5 international matches
            date_from = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%dT00:00:00Z")
            r = _req.get(f"{BSD_BASE}/teams/{team_id}/fixtures/", headers=hdrs,
                         params={"status":"finished", "limit":50, "date_from": date_from}, timeout=15)
            if r.status_code != 200: return [], False, team_id
        except: return [], False, team_id

        fixtures_data = r.json().get("results", [])
        
        # Take the most recent 5 and reverse to show newest first
        fixtures_data = fixtures_data[-5:]
        fixtures_data.reverse()
        
        results = []
        for fix in fixtures_data:
            fid        = fix.get("id", 0)
            home_id    = fix.get("home_team_id", 0)
            home_goals = fix.get("home_score") or 0
            away_goals = fix.get("away_score") or 0
            is_home    = (home_id == team_id)
            scored     = home_goals if is_home else away_goals
            conceded   = away_goals if is_home else home_goals
            opp_name   = fix.get("away_team","?") if is_home else fix.get("home_team","?")
            
            league_id  = fix.get("league_id", 0)
            lname = fix.get("league", {}).get("name") or fix.get("league_name")
            competition = lname if lname else LEAGUE_NAMES.get(league_id, "International")
            result     = "W" if scored > conceded else ("D" if scored == conceded else "L")

            formation_used = "Unknown"
            try:
                lr = _req.get(f"{BSD_BASE}/events/{fid}/lineups/", headers=hdrs, timeout=10)
                if lr.status_code == 200:
                    ld = lr.json()
                    if ld.get("lineup_status") != "unavailable" and ld.get("lineups"):
                        side = "home" if is_home else "away"
                        formation_used = (ld.get("lineups").get(side) or {}).get("formation", "Unknown")
            except: pass

            results.append({
                "fixture_id": fid, "formation": formation_used, "scored": scored,
                "conceded": conceded, "result": result, "competition": competition, "opponent": opp_name
            })
        return results, False, team_id

    def compute_nat_ratings(last5):
        if not last5: return None, None
        avg_s = sum(m["scored"] for m in last5) / len(last5)
        avg_c = sum(m["conceded"] for m in last5) / len(last5)
        return min(99, int(60 + avg_s * 10.5)), max(60, min(99, int(99 - avg_c * 10.5)))

    def best_nat_formation(last5):
        counts = {}
        for m in last5:
            f = m.get("formation","Unknown")
            if f and f != "Unknown": counts[f] = counts.get(f,0)+1
        return max(counts, key=counts.get) if counts else None

    def load_nation_squad(team_name, team_id, api_key):
        roster = []
        if team_name in players_db and len(players_db[team_name]) >= 11: return True
        if team_id:
            hdrs = {"Authorization": f"Token {api_key}"}
            try:
                r = _req.get(f"{BSD_BASE}/players/", headers=hdrs, params={"team_id": team_id, "limit": 100}, timeout=12)
                if r.status_code == 200:
                    SPEC_MAP = {
                        "GK":"GK","CB":"DF","RB":"DF","LB":"DF","RWB":"DF","LWB":"DF",
                        "CM":"MF","CDM":"MF","DM":"MF","CAM":"MF","AM":"MF",
                        "RM":"FW","LM":"FW","RW":"FW","LW":"FW","RWF":"FW","LWF":"FW","ST":"FW","CF":"FW","SS":"FW",
                    }
                    GEN_MAP = {"G":"GK","D":"DF","M":"MF","F":"FW"}
                    for p in r.json().get("results", []):
                        name = p.get("name") or p.get("short_name","")
                        if not name or name.strip() in ("","None","null"): continue
                        spec = str(p.get("specific_position","")).strip().upper()
                        gen  = str(p.get("position","M")).strip().upper()
                        pos  = SPEC_MAP.get(spec) or GEN_MAP.get(gen, "MF")
                        roster.append({"Name": name.strip(), "Pos": pos, "SpecPos": spec or gen, "Min": 0, "G_A": 0})
            except: pass

        if len(roster) < 11:
            roster = []
            generic_positions = ["GK", "RB", "RCB", "LCB", "LB", "RDM", "LDM", "CAM", "RW", "ST", "LW", "SUB1", "SUB2"]
            for pos in generic_positions:
                if pos in ["RW", "LW", "ST"]: gen_pos = "FW"
                elif pos in ["RDM", "LDM", "CAM"]: gen_pos = "MF"
                elif pos in ["RB", "RCB", "LCB", "LB"]: gen_pos = "DF"
                else: gen_pos = "GK"
                roster.append({"Name": f"{team_name} {pos}", "Pos": gen_pos, "SpecPos": pos.replace("R","").replace("L","") if len(pos)>2 else pos, "Min": 0, "G_A": 0, "fallback": True})

        players_db[team_name] = roster
        try:
            with open("players.json","w",encoding="utf-8") as f: json.dump(players_db, f, indent=2, ensure_ascii=False)
        except: pass
        return True

    if st.button("🔍 Fetch Form & Generate Optimal Tactics", type="primary", use_container_width=True):
        clear_cache_silently()
        if home_team == away_team:
            st.error("🚨 Invalid Matchup: A nation cannot play against itself.")
        else:
            bsd_key = st.secrets.get("BSD_API_KEY", "")
            if not bsd_key:
                st.error("🚨 BSD_API_KEY missing from Streamlit Secrets.")
            else:
                with st.spinner(f"📡 Fetching latest forms for {home_team} and {away_team}..."):
                    h_matches, h_cached, h_id = fetch_nation_last5(home_team, bsd_key)
                    a_matches, a_cached, a_id = fetch_nation_last5(away_team, bsd_key)

                    h_att_d, h_def_d = compute_nat_ratings(h_matches)
                    a_att_d, a_def_d = compute_nat_ratings(a_matches)

                    h_fb = teams_db.get(home_team, {"Attack": 82, "Defense": 80})
                    a_fb = teams_db.get(away_team, {"Attack": 82, "Defense": 80})

                    h_att = h_att_d or h_fb.get("Attack", 82)
                    h_def = h_def_d or h_fb.get("Defense", 80)
                    a_att = a_att_d or a_fb.get("Attack", 82)
                    a_def = a_def_d or a_fb.get("Defense", 80)

                    h_habit = best_nat_formation(h_matches)
                    a_habit = best_nat_formation(a_matches)

                    best_prob, best_form = 0, ""
                    for fc_code, fc_name in formations_map.items():
                        test = pd.DataFrame({"Formation": [fc_code], "Team_Attack": [h_att], "Team_Defense": [h_def], "Opp_Attack": [a_att], "Opp_Defense": [a_def]})
                        prob = float(model.predict_proba(test)[0][1] * 100)
                        if h_habit and fc_name == h_habit: prob += 5.0
                        if a_habit and a_habit[0].isdigit():
                            opp_backs = int(a_habit.split("-")[0])
                            if opp_backs >= 5 and fc_name.startswith("3"): prob -= 5.0
                        if prob > best_prob:
                            best_prob = prob
                            best_form = fc_name
                            
                    st.markdown("---")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("✅ Recommended Formation", best_form)
                    r2.metric("🤖 AI Win Probability", f"{best_prob:.1f}%")
                    r3.metric("📐 Opp. Usual Formation", a_habit or "Unknown")

                    st.markdown("### 📋 Last 5 Matches")
                    fc1, fc2 = st.columns(2)

                    def render_nat_form(tname, last5, att, dfn, cached):
                        st.markdown(f"**{tname}** <span style='font-size:12px;color:#6b8f72'>(🔴 live)</span>", unsafe_allow_html=True)
                        st.caption(f"⚔️ Attack: {att} | 🛡️ Defence: {dfn}")
                        if not last5: st.write("No recent matches found.")
                        for m in last5:
                            col = {"W":"#22c55e","D":"#f59e0b","L":"#ef4444"}.get(m["result"],"#6b8f72")
                            badge = f"<span style='background:{col};color:#000;padding:1px 7px;border-radius:4px;font-weight:700;font-size:11px'>{m['result']}</span>"
                            st.markdown(f"{badge} &nbsp;vs <b>{m['opponent']}</b> &nbsp;{m['scored']}–{m['conceded']} &nbsp;<code style='font-size:11px;background:#0d1f10;padding:2px 5px;border-radius:3px'>{m['formation']}</code> <span style='font-size:11px;color:#6b8f72'>{m['competition']}</span>", unsafe_allow_html=True)

                    with fc1: render_nat_form(home_team, h_matches, h_att, h_def, False)
                    with fc2: render_nat_form(away_team, a_matches, a_att, a_def, False)

                    st.markdown(f"### 👕 Recommended Starting XI — {best_form}")
                    
                    load_nation_squad(home_team, h_id, bsd_key)
                    
                    xi = select_starting_xi(home_team, best_form)
                    if xi:
                        for p in xi:
                            warn = " ⚠️" if p.get("fallback") else ""
                            ga   = f"{p['G_A']:.2f}" if isinstance(p['G_A'], float) else str(p['G_A'])
                            spec = str(p.get("SpecPos","")).strip()
                            role = p.get("_role", classify_player(p))
                            badge = spec if spec and spec.upper() not in ("G","D","M","F") else role
                            name  = str(p.get("Name","")).strip()
                            if not name or name in ("None","null"): continue
                            st.markdown(f"<div class='player-card'><span><span class='pos-badge'>{badge}</span>{name}{warn}</span><span class='stat-text'>⏱ {p.get('Min',0)} mins &nbsp;⚽ {ga} G+A</span></div>", unsafe_allow_html=True)
                    else:
                        st.info(f"Could not load any data for **{home_team}**.")
