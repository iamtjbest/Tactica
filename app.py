import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import joblib
import google.generativeai as genai
import os
import difflib
import random # NEW: Added for dynamic response generation
from sklearn.ensemble import RandomForestClassifier

# Fetch the API key from Streamlit secrets
gemini_api_key = st.secrets.get("GEMINI_API_KEY")

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    # Using the fast, current flash model
    ai_model = genai.GenerativeModel('gemini-2.5-flash')

st.set_page_config(page_title="Tactical AI", page_icon="⚽", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0b210e; background-image: gradient(0deg, #0b210e, #0b210e 60px, #0f2b13 60px, #0f2b13 120px); }
h1 { color: #22c55e !important; text-shadow: 0px 0px 10px rgba(34, 197, 94, 0.4); text-transform: uppercase; }
div[data-testid="metric-container"] { background: rgba(0, 0, 0, 0.6); padding: 15px; border-radius: 8px; border-left: 3px solid #22c55e; }
.player-card { background: rgba(0, 0, 0, 0.7); border: 1px solid #22c55e; border-radius: 5px; padding: 10px; margin-bottom: 5px; color: white; display: flex; justify-content: space-between;}
.stat-text { color: #8ca892; font-size: 14px; }
.live-alert { background: rgba(220, 38, 38, 0.2); border-left: 4px solid #dc2626; padding: 15px; border-radius: 5px; color: #f87171; margin-bottom: 15px; }
.live-suggestion { background: rgba(34, 197, 94, 0.2); border-left: 4px solid #22c55e; padding: 15px; border-radius: 5px; color: #86efac; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Tactical AI Engine")

# --- 1. THE UPGRADED AI TACTICAL BRAIN (TRUE ML) ---
@st.cache_resource
def load_true_model():
    formations_map_internal = {
        0: "3-4-3", 1: "3-5-2", 2: "3-4-1-2", 3: "3-2-4-1", 4: "3-4-2-1", 5: "3-3-1-3",
        6: "4-2-3-1", 7: "4-3-3", 8: "4-4-2", 9: "4-4-2 Diamond", 10: "4-1-4-1", 11: "4-3-2-1", 12: "4-2-2-2",
        13: "5-3-2", 14: "5-4-1", 15: "5-2-2-1", 16: "5-2-3"
    }
    
    try:
        # Load the real historical brain
        model = joblib.load('tactical_model.pkl')
        return model, formations_map_internal
    except:
        st.warning("⚠️ True ML brain not found! Defaulting to synthetic tactical engine. Run true_ml_trainer.py to generate tactical_model.pkl.")
        # Fallback to the synthetic generator if the pickle file is missing
        data = {
            'Formation': np.random.randint(0, 17, 1000), 
            'Team_Attack': np.random.randint(50, 99, 1000), 'Team_Defense': np.random.randint(50, 99, 1000),
            'Opp_Attack': np.random.randint(50, 99, 1000), 'Opp_Defense': np.random.randint(50, 99, 1000),
            'Win': np.random.randint(0, 2, 1000)
        }
        df = pd.DataFrame(data)
        model = RandomForestClassifier().fit(df[['Formation', 'Team_Attack', 'Team_Defense', 'Opp_Attack', 'Opp_Defense']], df['Win'])
        return model, formations_map_internal

model, formations_map = load_true_model()

# --- 2. LOAD DATABASES ---
try:
    with open('teams.json', 'r', encoding='utf-8') as f: 
        teams_db = json.load(f)
except Exception as e:
    st.error(f"🚨 Error loading teams.json: {e}")
    teams_db = {}

try:
    with open('players.json', 'r', encoding='utf-8') as f: 
        players_db = json.load(f)
except Exception as e: 
    st.error(f"🚨 JSON ERROR: Your players.json file is broken! Details: {e}")
    players_db = {}


# --- 3. PLAYER SELECTION ALGORITHM ---
def get_primary_pos(pos_string):
    """Return first position from compound string e.g. 'MF,FW' → 'MF'."""
    return (pos_string or "MF").split(",")[0].strip()

def select_starting_xi(team_name, formation):
    """
    Select best 11 for a formation using strict primary-position drafting.
    BSD positions: GK, DF, MF, FW (single, clean — no hybrid strings).
    """
    if team_name not in players_db:
        close_matches = difflib.get_close_matches(team_name, players_db.keys(), n=1, cutoff=0.6)
        if close_matches:
            team_name = close_matches[0]
        else:
            return None

    parts = [int(x) for x in re.findall(r'\d+', formation)]
    def_count = parts[0]
    att_count = parts[-1]
    mid_count = sum(parts[1:-1]) if len(parts) > 2 else parts[1]

    roster        = players_db[team_name]
    sorted_roster = sorted(roster, key=lambda x: (x['Min'], x['G_A']), reverse=True)
    starting_xi   = []
    drafted_names = set()

    def draft_by_primary(primary_pos, count):
        drafted = 0
        for p in sorted_roster:
            if drafted >= count: break
            if get_primary_pos(p['Pos']) == primary_pos and p['Name'] not in drafted_names:
                starting_xi.append(p)
                drafted_names.add(p['Name'])
                drafted += 1
        return drafted

    def draft_by_any(pos_keyword, count):
        drafted = 0
        for p in sorted_roster:
            if drafted >= count: break
            if pos_keyword in p['Pos'] and p['Name'] not in drafted_names:
                starting_xi.append(p)
                drafted_names.add(p['Name'])
                drafted += 1
        return drafted

    # GK
    if draft_by_primary("GK", 1) < 1: draft_by_any("GK", 1)
    # Defenders
    n = draft_by_primary("DF", def_count)
    if n < def_count: draft_by_any("DF", def_count - n)
    # Midfielders
    n = draft_by_primary("MF", mid_count)
    if n < mid_count: draft_by_any("MF", mid_count - n)
    # Forwards
    n = draft_by_primary("FW", att_count)
    if n < att_count: draft_by_any("FW", att_count - n)

    # Emergency fill (data gap) — flag with fallback key
    for p in sorted_roster:
        if len(starting_xi) >= 11: break
        if p['Name'] not in drafted_names:
            p = dict(p); p["fallback"] = True
            starting_xi.append(p)
            drafted_names.add(p['Name'])

    return starting_xi

# --- 4. SIDEBAR NAVIGATION & UI ---
if teams_db:
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Module:", [
        "🤖 Pre-Match Auto-Tactics", 
        "📊 Pre-Match Opponent Analysis", 
        "🧠 Coach's Sandbox", 
        "⏱️ Live Match Simulator",
        "💬 Assistant Manager Chat"
    ])

    # ---------------------------------------------------------
    # MODULE 1: PRE-MATCH AUTO-TACTICS
    # ---------------------------------------------------------
    if app_mode == "🤖 Pre-Match Auto-Tactics":
        import requests as _req, time as _time

        st.markdown("## 🤖 Pre-Match Auto-Tactics")
        st.write("Select your teams. The engine reads each team's last 5 matches via BSD API, "
                 "extracts real formations used, calculates dynamic attack/defence ratings "
                 "from actual results, then recommends the optimal game plan.")

        col1, col2 = st.columns(2)
        with col1: my_team  = st.selectbox("Your Team", list(teams_db.keys()), index=0)
        with col2: opp_team = st.selectbox("Opponent",  list(teams_db.keys()), index=1 if len(teams_db) > 1 else 0)

        FORM_CACHE_FILE = "form_cache.json"
        FORM_CACHE_TTL  = 86400  # 24 hours

        def load_form_cache():
            try:
                with open(FORM_CACHE_FILE, "r", encoding="utf-8") as f: return json.load(f)
            except Exception: return {}

        def save_form_cache(data):
            try:
                with open(FORM_CACHE_FILE, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
            except Exception: pass

        # BSD Team IDs — same as auto_updater.py
        TEAM_IDS = {
            "Manchester City": 267, "Arsenal": 2,     "Liverpool": 10,
            "Aston Villa": 24,      "Tottenham": 6,   "Manchester Utd": 8,
            "Chelsea": 4,           "Newcastle": 19,  "Brighton": 36,
            "West Ham": 20,         "Crystal Palace": 31, "Everton": 14,
            "Fulham": 43,           "Brentford": 189, "Bournemouth": 91,
            "Nott'm Forest": 17,   "Wolves": 39,     "Leicester": 26,
            "Southampton": 57,      "Ipswich": 40,
        }

        def fetch_last5(team_name, api_key):
            """
            BSD API v2 — fetch last 5 finished fixtures for a team.
            Endpoints:
              GET /api/v2/teams/{id}/fixtures/?status=finished&limit=5
                Response: {"results": [event objects]}
                Fields: id, home_team_id, home_team, away_team,
                        home_score, away_score, league_name

              GET /api/v2/events/{id}/lineups/
                Response: {
                  "lineup_status": "confirmed|predicted|unavailable",
                  "lineups": {        ← null when status=="unavailable"
                    "home": {"formation": "4-3-3", ...},
                    "away": {"formation": "4-4-2", ...}
                  }
                }
            Cached 24h in form_cache.json — free repeat calls.
            """
            cache     = load_form_cache()
            entry     = cache.get(team_name, {})
            cache_age = _time.time() - entry.get("fetched_at", 0)
            if entry and cache_age < FORM_CACHE_TTL:
                return entry.get("matches", []), True  # True = from cache

            team_id = TEAM_IDS.get(team_name)
            if not team_id: return [], False

            hdrs = {"Authorization": f"Token {api_key}"}

            try:
                r = _req.get(
                    f"https://sports.bzzoiro.com/api/v2/teams/{team_id}/fixtures/",
                    headers=hdrs,
                    params={"status": "finished", "limit": 5},
                    timeout=12
                )
            except Exception: return [], False

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
                opp_name   = fix.get("away_team", "?") if is_home else fix.get("home_team", "?")
                competition = fix.get("league_name", "Unknown")
                result     = "W" if scored > conceded else ("D" if scored == conceded else "L")

                # Lineup/formation
                formation_used = "Unknown"
                try:
                    lr = _req.get(
                        f"https://sports.bzzoiro.com/api/v2/events/{fid}/lineups/",
                        headers=hdrs, timeout=12
                    )
                    _time.sleep(0.2)
                    if lr.status_code == 200:
                        ld = lr.json()
                        lu_status = ld.get("lineup_status", "unavailable")
                        # lineups is null when status == "unavailable" — MUST check!
                        if lu_status != "unavailable" and ld.get("lineups"):
                            side = "home" if is_home else "away"
                            formation_used = ld["lineups"].get(side, {}).get("formation") or "Unknown"
                except Exception: pass

                results.append({
                    "fixture_id": fid, "formation": formation_used,
                    "scored": scored,  "conceded": conceded,
                    "result": result,  "competition": competition, "opponent": opp_name,
                })

            cache[team_name] = {"fetched_at": _time.time(), "matches": results}
            save_form_cache(cache)
            return results, False

        def compute_dynamic_ratings(last5):
            if not last5: return None, None
            avg_scored   = sum(m["scored"]   for m in last5) / len(last5)
            avg_conceded = sum(m["conceded"] for m in last5) / len(last5)
            return min(99, int(60 + avg_scored * 9.75)), max(60, min(99, int(99 - avg_conceded * 9.75)))

        def most_used_formation(last5):
            counts = {}
            for m in last5:
                f = m.get("formation", "Unknown")
                if f and f != "Unknown": counts[f] = counts.get(f, 0) + 1
            return max(counts, key=counts.get) if counts else None

        # ── UI ──────────────────────────────────────────────────────────────
        api_key    = st.secrets.get("BSD_API_KEY", "")
        form_cache = load_form_cache()
        my_cached  = form_cache.get(my_team, {})
        opp_cached = form_cache.get(opp_team, {})

        if my_cached and opp_cached:
            my_h  = int((_time.time() - my_cached.get("fetched_at", 0))  / 3600)
            opp_h = int((_time.time() - opp_cached.get("fetched_at", 0)) / 3600)
            st.caption(f"📦 Cached form data — {my_team}: {my_h}h ago | {opp_team}: {opp_h}h ago")

        with st.expander("ℹ️ API Usage"):
            st.markdown(
                "Fetching fresh data costs ~12 BSD API calls per run (no rate limits). "
                "Results are **cached 24 hours** — subsequent clicks cost zero calls."
            )

        btn_label = ("♻️ Generate Tactics (Cached) / Re-Fetch Last 5"
                     if my_cached and opp_cached
                     else "🔍 Fetch Last 5 Matches & Generate Optimal Tactics")

        if st.button(btn_label, use_container_width=True, type="primary"):
            if my_team == opp_team:
                st.error("🚨 Tactical Error: A team cannot play against itself!")
            elif not api_key:
                st.error("🚨 BSD_API_KEY missing from Streamlit Secrets.")
            else:
                with st.spinner(f"📡 Fetching last 5 matches for {my_team} and {opp_team}..."):
                    my_last5,  my_from_cache  = fetch_last5(my_team,  api_key)
                    opp_last5, opp_from_cache = fetch_last5(opp_team, api_key)

                if not my_last5 and not opp_last5:
                    st.error("🚨 No match data returned. Check BSD_API_KEY and team IDs.")
                else:
                    my_att_d,  my_def_d  = compute_dynamic_ratings(my_last5)
                    opp_att_d, opp_def_d = compute_dynamic_ratings(opp_last5)
                    my_att  = my_att_d  or teams_db.get(my_team,  {}).get("Attack",  80)
                    my_def  = my_def_d  or teams_db.get(my_team,  {}).get("Defense", 80)
                    opp_att = opp_att_d or teams_db.get(opp_team, {}).get("Attack",  80)
                    opp_def = opp_def_d or teams_db.get(opp_team, {}).get("Defense", 80)

                    opp_habit = most_used_formation(opp_last5)
                    my_habit  = most_used_formation(my_last5)

                    # Score all formations through ML model
                    best_prob, best_form = 0, ""
                    all_scores = {}
                    for f_code, f_name in formations_map.items():
                        test = pd.DataFrame({"Formation": [f_code], "Team_Attack": [my_att],
                                             "Team_Defense": [my_def], "Opp_Attack": [opp_att],
                                             "Opp_Defense": [opp_def]})
                        prob = model.predict_proba(test)[0][1] * 100
                        if my_habit and f_name == my_habit: prob += 5   # familiarity bonus
                        if opp_habit:
                            opp_backs = int(opp_habit.split("-")[0]) if opp_habit[0].isdigit() else 4
                            if opp_backs >= 5 and f_name.startswith("3"): prob -= 5
                        all_scores[f_name] = round(prob, 1)
                        if prob > best_prob: best_prob, best_form = prob, f_name

                    # Results
                    st.markdown("---")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("✅ Recommended Formation", best_form)
                    r2.metric("🤖 AI Win Probability", f"{best_prob:.1f}%")
                    r3.metric("📐 Opp. Usual Formation", opp_habit or "Unknown")

                    # Form tables
                    st.markdown("### 📋 Last 5 Matches")
                    fc1, fc2 = st.columns(2)
                    def render_form(tname, last5, att, dfn, cached):
                        label = "📦 cached" if cached else "🔴 live"
                        st.markdown(f"**{tname}** <span style='font-size:12px;color:#8ca892'>({label})</span>", unsafe_allow_html=True)
                        st.caption(f"⚔️ Attack: {att} | 🛡️ Defence: {dfn}")
                        for m in last5:
                            col = {"W":"#22c55e","D":"#f59e0b","L":"#ef4444"}.get(m["result"],"#8ca892")
                            badge = f"<span style='background:{col};color:#000;padding:1px 6px;border-radius:3px;font-weight:bold;font-size:12px'>{m['result']}</span>"
                            st.markdown(
                                f"{badge} vs **{m['opponent']}** {m['scored']}–{m['conceded']} "
                                f"<code style='font-size:11px'>{m['formation']}</code> "
                                f"<span style='font-size:11px;color:#8ca892'>{m['competition']}</span>",
                                unsafe_allow_html=True)
                    with fc1: render_form(my_team,  my_last5,  my_att,  my_def,  my_from_cache)
                    with fc2: render_form(opp_team, opp_last5, opp_att, opp_def, opp_from_cache)

                    # Formation leaderboard
                    st.markdown("### 🏆 Formation Ranking")
                    for rank, (fname, score) in enumerate(sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:5], 1):
                        medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][rank-1]
                        bar   = int((score/100)*300)
                        st.markdown(
                            f"{medal} **{fname}** "
                            f"<span style='display:inline-block;width:{bar}px;height:10px;"
                            f"background:#22c55e;border-radius:3px;vertical-align:middle'></span>"
                            f" {score}%", unsafe_allow_html=True)

                    # Starting XI
                    st.markdown(f"### 👕 AI Recommended Starting XI — {best_form}")
                    xi = select_starting_xi(my_team, best_form)
                    if xi:
                        for p in xi:
                            warn = " ⚠️" if p.get("fallback") else ""
                            g_a  = f"{p['G_A']:.2f}" if isinstance(p['G_A'], float) else str(p['G_A'])
                            st.markdown(
                                f"<div class='player-card'>"
                                f"<span><b>{get_primary_pos(p['Pos'])}</b> | {p['Name']}{warn}</span>"
                                f"<span class='stat-text'>⏱️ {p['Min']} mins | ⚽ {g_a} G+A</span>"
                                f"</div>", unsafe_allow_html=True)
                    else:
                        st.warning(f"No player data for '{my_team}' in players.json.")

    # ---------------------------------------------------------
    # MODULE 2: PRE-MATCH OPPONENT ANALYSIS
    # ---------------------------------------------------------
    elif app_mode == "📊 Pre-Match Opponent Analysis":
        st.markdown("## 📊 Pre-Match Opponent Analysis")
        st.write("Scout your opponent before kickoff. Compare team strengths and generate a tactical briefing.")

        col1, col2 = st.columns(2)
        with col1: my_team = st.selectbox("Your Team", list(teams_db.keys()), index=0, key="scout_team")
        with col2: opp_team = st.selectbox("Opponent", list(teams_db.keys()), index=1 if len(teams_db) > 1 else 0, key="scout_opp")

        if my_team == opp_team:
            st.error("🚨 Tactical Error: A team cannot play against itself!")
        else:
            my_att, my_def = teams_db[my_team]["Attack"], teams_db[my_team]["Defense"]
            opp_att, opp_def = teams_db[opp_team]["Attack"], teams_db[opp_team]["Defense"]

            st.markdown("### ⚔️ Head-to-Head Comparison")
            
            comp_col1, comp_col2, comp_col3 = st.columns([2, 1, 2])
            
            with comp_col1:
                st.metric(f"{my_team} Attack", my_att, f"{my_att - opp_def} vs Opp Def")
                st.metric(f"{my_team} Defense", my_def, f"{my_def - opp_att} vs Opp Att")
                
            with comp_col2:
                st.markdown("<h1 style='text-align: center; color: #8ca892; margin-top: 20px;'>VS</h1>", unsafe_allow_html=True)
                
            with comp_col3:
                st.metric(f"{opp_team} Attack", opp_att, f"{opp_att - my_def} vs Our Def", delta_color="inverse")
                st.metric(f"{opp_team} Defense", opp_def, f"{opp_def - my_att} vs Our Att", delta_color="inverse")

            st.markdown("---")
            st.markdown("### 📋 AI Pre-Match Briefing")
            briefing = ""

            if my_att > opp_def + 10:
                briefing += "🎯 **Offensive Dominance:** Their defense is significantly weaker than our attack. Instruct forwards to play a high line and exploit the gaps. A high-pressing possession game will suffocate them.\n\n"
            elif my_att < opp_def:
                briefing += "🧱 **Tough Defense:** They have a rock-solid defense. Breaking them down centrally will be difficult. Focus on wide areas, overlaps, and set-pieces to create chances.\n\n"

            if opp_att > my_def + 10:
                briefing += "⚠️ **Defensive Vulnerability:** Their attack is lethal compared to our defense. Avoid playing a high line. Consider deploying a double pivot (two defensive midfielders) to screen the backline and limit their space.\n\n"
            elif my_def > opp_att:
                briefing += "🛡️ **Defensive Superiority:** We are well-equipped to handle their attackers. Fullbacks have the license to push forward aggressively without constantly fearing the counter-attack.\n\n"

            if abs(my_att - opp_att) <= 5 and abs(my_def - opp_def) <= 5:
                briefing += "⚖️ **Even Matchup:** This is a tightly contested fixture. The game will likely be won or lost in the midfield transitions. Focus on ball retention and capitalize on unforced errors.\n\n"

            st.info(briefing if briefing else "No extreme tactical mismatches detected. Play to your standard strengths and maintain structural discipline.")

    # ---------------------------------------------------------
    # MODULE 3: COACH'S SANDBOX
    # ---------------------------------------------------------
    elif app_mode == "🧠 Coach's Sandbox":
        st.markdown("## 🧠 Coach's Sandbox")
        st.write("Set up your own formation and squad, then see how it scores before asking the AI for a second opinion.")
        
        col1, col2 = st.columns(2)
        with col1: my_team = st.selectbox("Your Team", list(teams_db.keys()), index=0, key="coach_team") 
        with col2: opp_team = st.selectbox("Opponent", list(teams_db.keys()), index=1 if len(teams_db) > 1 else 0, key="coach_opp")

        if my_team == opp_team:
            st.error("🚨 Tactical Error: A team cannot play against itself!")
        else:
            form_col, squad_col = st.columns(2)
            with form_col:
                coach_formation = st.selectbox("Your Preferred Formation", list(formations_map.values()))
            
            roster_names = []
            actual_team_key = my_team
            if my_team in players_db:
                roster_names = [p['Name'] for p in players_db[my_team]]
            else:
                matches = difflib.get_close_matches(my_team, players_db.keys(), n=1, cutoff=0.6)
                if matches:
                    actual_team_key = matches[0]
                    roster_names = [p['Name'] for p in players_db[actual_team_key]]

            with squad_col:
                if roster_names:
                    coach_xi = st.multiselect("Draft Your Starting XI (Max 11)", roster_names, max_selections=11)
                else:
                    st.warning(f"No player data loaded for {my_team}.")
                    coach_xi = []

            if st.button("Analyze My Gameplan", use_container_width=True, type="primary"):
                if len(coach_xi) < 11:
                    st.warning(f"⚠️ You only drafted {len(coach_xi)} players. A full squad requires 11 on the pitch!")
                
                my_att, my_def = teams_db[my_team]["Attack"], teams_db[my_team]["Defense"]
                opp_att, opp_def = teams_db[opp_team]["Attack"], teams_db[opp_team]["Defense"]
                
                f_code = list(formations_map.keys())[list(formations_map.values()).index(coach_formation)]
                test_match = pd.DataFrame({'Formation': [f_code], 'Team_Attack': [my_att], 'Team_Defense': [my_def], 'Opp_Attack': [opp_att], 'Opp_Defense': [opp_def]})
                coach_prob = model.predict_proba(test_match)[0][1] * 100
                
                st.metric("Your Strategy's Win Probability", f"{coach_prob:.1f}%")
                
                st.markdown("---")
                with st.expander("🤖 Ask AI for a Second Opinion (Reveal Optimal Setup)"):
                    best_prob, best_form = 0, ""
                    for fc, fn in formations_map.items():
                        tm = pd.DataFrame({'Formation': [fc], 'Team_Attack': [my_att], 'Team_Defense': [my_def], 'Opp_Attack': [opp_att], 'Opp_Defense': [opp_def]})
                        prob = model.predict_proba(tm)[0][1] * 100
                        if prob > best_prob: best_prob, best_form = prob, fn
                    
                    diff = best_prob - coach_prob
                    if diff > 0.1:
                        st.success(f"The AI found a more optimized tactical setup! (+{diff:.1f}% Win Prob)")
                    else:
                        st.info("Great minds think alike. The AI agrees your formation is highly optimal.")

                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("AI's Best Formation", best_form)
                    res_col2.metric("AI's Win Probability", f"{best_prob:.1f}%")
                    
                    st.markdown("#### AI Recommended Starting XI")
                    xi = select_starting_xi(actual_team_key, best_form)
                    if xi:
                        for p in xi:
                            star = "⭐" if p['Name'] not in coach_xi else "✅"
                            st.markdown(f"<div class='player-card'><span>{star} <b>{p['Pos']}</b> | {p['Name']}</span> <span class='stat-text'>⏱️ {p['Min']} mins | ⚽ {p['G_A']} G+A</span></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # MODULE 4: LIVE MATCH SIMULATOR (EXPANDED DYNAMIC LOGIC)
    # ---------------------------------------------------------
    elif app_mode == "⏱️ Live Match Simulator":
        st.markdown("## ⏱️ Live Match Simulator")
        st.write("Input current match conditions to receive real-time tactical adjustments and substitution alerts.")
        
        col1, col2 = st.columns(2)
        with col1: 
            my_team = st.selectbox("Your Team", list(teams_db.keys()), index=0, key="live_team")
            match_min = st.slider("Match Minute", 1, 90, 60)
            score_diff = st.selectbox("Current Scoreline", ["Winning Comfortably (+2 goals)", "Leading by 1", "Tied", "Trailing by 1", "Losing Badly (-2 goals)"], index=2)
            
        with col2: 
            opp_team = st.selectbox("Opponent", list(teams_db.keys()), index=1 if len(teams_db) > 1 else 0, key="live_opp")
            current_form = st.selectbox("Your Current Formation", list(formations_map.values()), index=6)
            
            # EXPANDED TACTICAL PROBLEMS
            match_event = st.selectbox("Current Tactical Problem", [
                "None - Game is balanced", 
                "Midfield is being overrun", 
                "Unable to break down deep block", 
                "Vulnerable to counter-attacks", 
                "Attackers look fatigued",
                "Defenders struggling with opponent pace",
                "Losing the aerial battle in the box",
                "Opponent's high press is suffocating us"
            ])

        # --- THE DYNAMIC TACTICAL DATABASE ---
        tactical_db = {
            "Midfield is being overrun": {
                "tactics": [
                    "⚠️ **Overrun Midfield:** The opponent is dominating possession centrally. Shift to a formation with 4 or 5 midfielders to regain control.",
                    "⚠️ **Numerical Disadvantage:** We are losing the midfield battle. Instruct your wingers to invert and create a box midfield.",
                    "⚠️ **Central Overload:** They are playing right through us. Drop the defensive line slightly and compress the space between midfield and defense."
                ],
                "subs": [
                    "🔄 **Sub Alert:** Introduce a fresh defensive midfielder (CDM) to break up play and disrupt their rhythm.",
                    "🔄 **Sub Alert:** Sacrifice a striker for an energetic box-to-box midfielder to win second balls.",
                    "🔄 **Sub Alert:** Bring on a deep-lying playmaker to help dictate the tempo and retain possession."
                ]
            },
            "Unable to break down deep block": {
                "tactics": [
                    "⚠️ **Deep Block Detected:** Opponent is parking the bus. Suggest moving to a wider formation to stretch their defense.",
                    "⚠️ **Low Block Frustration:** Stop forcing it centrally. Instruct fullbacks to overlap aggressively and hit early crosses.",
                    "⚠️ **Compact Defense:** We need to move their block. Increase passing tempo and encourage center-backs to step into midfield."
                ],
                "subs": [
                    "🔄 **Sub Alert:** Substitute a fatigued central midfielder for a tricky winger who can beat a man 1v1.",
                    "🔄 **Sub Alert:** Bring on a tall target-man (FW) to provide an aerial threat for crosses.",
                    "🔄 **Sub Alert:** Introduce an attacking fullback to provide width and overload the wide areas."
                ]
            },
            "Vulnerable to counter-attacks": {
                "tactics": [
                    "⚠️ **Counter-Attack Risk:** We are overcommitting forward. Keep one holding midfielder anchored at all times.",
                    "⚠️ **Transition Danger:** Instruct your fullbacks to invert rather than overlap, providing extra bodies in the middle if we lose the ball.",
                    "⚠️ **High Line Exposed:** Drop the defensive line back 10 yards. We are leaving too much space behind."
                ],
                "subs": [
                    "🔄 **Sub Alert:** Consider substituting a slow center-back for a faster alternative to cover the channels.",
                    "🔄 **Sub Alert:** Bring on a dedicated defensive fullback to lock down the flank they are exposing.",
                    "🔄 **Sub Alert:** Introduce an energetic midfielder specifically instructed to commit tactical fouls high up the pitch."
                ]
            },
            "Attackers look fatigued": {
                "tactics": [
                    "⚠️ **Attacking Fatigue:** Pressing intensity has dropped. Switch from a high press to a mid-block to conserve energy.",
                    "⚠️ **Lethargic Movement:** We are static in the final third. Shift to a counter-attacking style and let the opponent have the ball.",
                    "⚠️ **Tired Legs:** Instruct the team to play shorter passes to feet rather than playing into channels for forwards to chase."
                ],
                "subs": [
                    "🔄 **Sub Alert:** Immediate substitution required for your starting forwards (FW) to restore high-press energy.",
                    "🔄 **Sub Alert:** Introduce a pacey winger against their tired fullbacks.",
                    "🔄 **Sub Alert:** Bring on a fresh shadow striker to exploit the spaces opening up as the game stretches."
                ]
            },
            "Defenders struggling with opponent pace": {
                "tactics": [
                    "⚠️ **Pace Mismatch:** Their forwards are too quick for our high line. Drop into a deeper defensive block immediately.",
                    "⚠️ **Exposed Channels:** Double up on the flanks. Instruct wingers to track back and support the fullbacks.",
                    "⚠️ **Speed Threat:** Play more conservatively in possession. Avoid risky passes that could lead to quick turnovers."
                ],
                "subs": [
                    "🔄 **Sub Alert:** Hook your slowest center-back for a more agile defender.",
                    "🔄 **Sub Alert:** Introduce a defensive midfielder to screen the backline and cut out through-balls.",
                    "🔄 **Sub Alert:** Bring on fresh fullbacks to cope with their wide speedsters."
                ]
            },
            "Losing the aerial battle in the box": {
                "tactics": [
                    "⚠️ **Aerial Weakness:** We are being bullied in the air. Instruct players to stop crosses at the source by pressing wide players aggressively.",
                    "⚠️ **Set-Piece Danger:** Avoid conceding cheap fouls around the penalty area. Transition to a zonal marking system on corners.",
                    "⚠️ **Long Ball Threat:** Force them to play through the middle. Show their defenders inside so they can't hit diagonal long balls."
                ],
                "subs": [
                    "🔄 **Sub Alert:** Introduce a taller, more physical center-back to handle crosses.",
                    "🔄 **Sub Alert:** Bring on a robust midfielder to win the second balls dropping off the target man.",
                    "🔄 **Sub Alert:** Substitute a small fullback for a taller option to defend back-post crosses."
                ]
            },
            "Opponent's high press is suffocating us": {
                "tactics": [
                    "⚠️ **Suffocating Press:** We can't build from the back. Instruct the goalkeeper to go long and bypass their midfield press.",
                    "⚠️ **Trapped in Defense:** Stretch the pitch. Instruct wingers to stay as wide and high as possible to pin their fullbacks back.",
                    "⚠️ **High Turnover Risk:** Stop playing short goal kicks. Use a target man to win the first ball in their half."
                ],
                "subs": [
                    "🔄 **Sub Alert:** Introduce a physical target man (FW) to aim long clearances toward.",
                    "🔄 **Sub Alert:** Bring on a highly technical, press-resistant midfielder to help navigate out of tight spaces.",
                    "🔄 **Sub Alert:** Swap to a goalkeeper with better distribution statistics."
                ]
            }
        }

        if st.button("Generate Live Instructions", use_container_width=True, type="primary"):
            if my_team == opp_team:
                st.error("🚨 Tactical Error: A team cannot play against itself!")
            else:
                my_att, my_def = teams_db[my_team]["Attack"], teams_db[my_team]["Defense"]
                opp_att, opp_def = teams_db[opp_team]["Attack"], teams_db[opp_team]["Defense"]
                
                adjusted_att = my_att
                adjusted_def = my_def
                
                # Dynamic Logic Selection
                tactic_advice = ""
                sub_advice = ""
                
                # 1. Time & Scoreline Logic (Randomized variations)
                if match_min > 70:
                    if "Trailing" in score_diff or "Losing" in score_diff:
                        adjusted_att += 15 
                        adjusted_def -= 10
                        tactic_advice += random.choice([
                            "⏳ **Late Game Scenario:** Abandon structural discipline. Transition to an ultra-attacking overload. Push fullbacks extremely high.\n\n",
                            "⏳ **Chasing the Game:** Go direct. Bypass the midfield and load the penalty box with extra bodies.\n\n",
                            "⏳ **Desperation Phase:** Throw caution to the wind. Leave only two defenders back and commit everyone else forward.\n\n"
                        ])
                    elif "Leading" in score_diff or "Winning" in score_diff:
                        adjusted_def += 15 
                        adjusted_att -= 10
                        tactic_advice += random.choice([
                            "🛡️ **Protect the Lead:** Drop the defensive line deeper, tighten spaces between lines, and waste time where possible.\n\n",
                            "🛡️ **Lock it Down:** Shift to a back five. Clog the center of the pitch and force them into low-percentage crosses.\n\n",
                            "🛡️ **Game Management:** Focus entirely on shape. Do not commit numbers forward on the counter-attack.\n\n"
                        ])
                
                # 2. Match Event Logic (Pulls from Database)
                if match_event != "None - Game is balanced":
                    event_data = tactical_db.get(match_event)
                    if event_data:
                        tactic_advice += random.choice(event_data["tactics"])
                        sub_advice = random.choice(event_data["subs"])

                # Recalculate best formation with adjusted stats
                best_prob, best_form = 0, ""
                for f_code, f_name in formations_map.items():
                    test_match = pd.DataFrame({'Formation': [f_code], 'Team_Attack': [adjusted_att], 'Team_Defense': [adjusted_def], 'Opp_Attack': [opp_att], 'Opp_Defense': [opp_def]})
                    prob = model.predict_proba(test_match)[0][1] * 100
                    
                    # AI Penalties for bad formation matchups based on the specific problem
                    if match_event == "Midfield is being overrun" and f_name in ["4-2-4", "5-2-3", "4-4-2"]: prob -= 15 
                    if match_event == "Unable to break down deep block" and f_name.startswith("5"): prob -= 15 
                    if match_event == "Vulnerable to counter-attacks" and f_name in ["3-4-3", "4-2-4"]: prob -= 15
                    if match_event == "Defenders struggling with opponent pace" and f_name.startswith("3"): prob -= 15
                    
                    if prob > best_prob: best_prob, best_form = prob, f_name

                st.markdown("---")
                if current_form != best_form:
                    st.markdown(f"<div class='live-alert'><b>🚨 TACTICAL SHIFT REQUIRED</b><br>Current formation ({current_form}) is suboptimal for this match state. Shift immediately to <b>{best_form}</b>.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='live-suggestion'><b>✅ FORMATION OPTIMAL</b><br>Maintain your current shape ({current_form}), but execute the tactical tweaks below.</div>", unsafe_allow_html=True)
                
                colA, colB = st.columns(2)
                colA.markdown("### 📋 AI Tactical Briefing")
                colA.write(tactic_advice if tactic_advice else "Maintain standard operating procedures. The game is currently balanced.")
                
                colB.markdown("### 🔄 Substitution Protocol")
                colB.write(sub_advice if sub_advice else "No emergency substitutions required based on current data. Monitor stamina levels.")

    # ---------------------------------------------------------
    # MODULE 5: ASSISTANT MANAGER CHAT (BSD Live Intel)
    # ---------------------------------------------------------
    elif app_mode == "💬 Assistant Manager Chat":
        import requests as _req, time as _time

        st.markdown("## 💬 Assistant Manager Chat")
        st.write("Select your teams, sync live match data from any competition, then chat with your AI assistant manager.")

        col1, col2 = st.columns(2)
        with col1: chat_my_team  = st.selectbox("Your Team", list(teams_db.keys()), index=0, key="chat_team")
        with col2: chat_opp_team = st.selectbox("Opponent",  list(teams_db.keys()), index=1 if len(teams_db) > 1 else 0, key="chat_opp")

        st.markdown("---")
        st.markdown("### 📡 Live Match Intel")

        LIVE_CACHE_FILE = "live_match_cache.json"
        CACHE_TTL       = 300  # show cached data for 5 min before offering re-sync

        def load_live_cache():
            try:
                return json.load(open(LIVE_CACHE_FILE, encoding="utf-8"))
            except Exception: return {}

        def save_live_cache(data):
            try:
                json.dump(data, open(LIVE_CACHE_FILE, "w", encoding="utf-8"), indent=2)
            except Exception: pass

        def cache_key(t1, t2):
            return f"{t1.lower().strip()}__vs__{t2.lower().strip()}"

        live_cache = load_live_cache()
        ck         = cache_key(chat_my_team, chat_opp_team)
        cached     = live_cache.get(ck, {})
        cache_age  = _time.time() - cached.get("fetched_at", 0)
        cache_ok   = cache_age < CACHE_TTL

        # Display cached result if fresh
        if cache_ok and cached.get("match_found"):
            d = cached
            st.markdown(
                f"<div class='live-suggestion'>"
                f"<b>✅ LIVE: {d['home_name']} {d['home_goals']} – {d['away_goals']} {d['away_name']}</b>"
                f" | ⏱️ {d['minute']}' | 🏆 {d['competition']}"
                f"<br><span style='font-size:12px;color:#86efac'>Cached {int(cache_age/60)} min ago</span>"
                f"</div>", unsafe_allow_html=True)
            st.session_state.live_context = (
                f"LIVE ({d['competition']}): Minute {d['minute']}'. "
                f"Score: {d['home_name']} {d['home_goals']} – {d['away_goals']} {d['away_name']}."
            )
        elif cached and not cached.get("match_found"):
            st.info(f"ℹ️ Last sync: {chat_my_team} not in a live fixture. Hit Sync to check again.")
        else:
            st.info("No live data cached. Hit **Sync** to search all live competitions worldwide.")
            if "live_context" not in st.session_state:
                st.session_state.live_context = "No live data. Provide general pre-match tactical advice."

        # Sync button — BSD has no rate limits so cooldown is just 30s (Redis TTL)
        last_sync  = st.session_state.get("last_bsd_sync", 0)
        secs_since = _time.time() - last_sync
        COOLDOWN   = 30  # matches BSD Redis TTL of 30s
        sync_ready = secs_since >= COOLDOWN
        btn_label  = ("🔄 Sync Live Data (All Competitions)"
                      if sync_ready else
                      f"🔄 Sync (cooldown: {max(1, int(COOLDOWN - secs_since))}s)")

        if st.button(btn_label, use_container_width=True, disabled=not sync_ready):
            bsd_key = st.secrets.get("BSD_API_KEY")
            if not bsd_key:
                st.error("🚨 BSD_API_KEY missing from Streamlit Secrets!")
            else:
                with st.spinner("🌐 Scanning all live fixtures worldwide..."):
                    hdrs = {"Authorization": f"Token {bsd_key}"}
                    try:
                        # BSD Live window: GET /api/v2/events/live/
                        # Response: {"count": N, "events": [...]}
                        # Each event: home_team, away_team, home_score, away_score,
                        #             current_minute, league_name, status
                        res = _req.get(
                            "https://sports.bzzoiro.com/api/v2/events/live/",
                            headers=hdrs, timeout=12
                        )
                        st.session_state.last_bsd_sync = _time.time()

                        if res.status_code != 200:
                            st.error(f"🚨 BSD API error {res.status_code}. Check your key.")
                        else:
                            live_data   = res.json().get("events", [])
                            match_found = False

                            for match in live_data:
                                home_name  = match.get("home_team", "")
                                away_name  = match.get("away_team", "")

                                # Fuzzy match: works even if BSD name differs slightly
                                my_hit  = (chat_my_team.lower()  in home_name.lower() or
                                           home_name.lower() in chat_my_team.lower()  or
                                           chat_my_team.lower()  in away_name.lower() or
                                           away_name.lower() in chat_my_team.lower())
                                opp_hit = (chat_opp_team.lower() in home_name.lower() or
                                           home_name.lower() in chat_opp_team.lower() or
                                           chat_opp_team.lower() in away_name.lower() or
                                           away_name.lower() in chat_opp_team.lower())

                                if my_hit and opp_hit:
                                    minute      = match.get("current_minute") or 0
                                    home_goals  = match.get("home_score") or 0
                                    away_goals  = match.get("away_score") or 0
                                    competition = match.get("league_name", "Unknown Competition")
                                    status_txt  = match.get("status", "inprogress")

                                    entry = {
                                        "fetched_at": _time.time(), "match_found": True,
                                        "home_name": home_name, "away_name": away_name,
                                        "home_goals": home_goals, "away_goals": away_goals,
                                        "minute": minute, "competition": competition,
                                    }
                                    live_cache[ck] = entry
                                    save_live_cache(live_cache)

                                    st.session_state.live_context = (
                                        f"LIVE ({competition}): Minute {minute}'. "
                                        f"Score: {home_name} {home_goals} – {away_goals} {away_name}. "
                                        f"Status: {status_txt}."
                                    )
                                    st.markdown(
                                        f"<div class='live-suggestion'>"
                                        f"<b>✅ LIVE: {home_name} {home_goals} – {away_goals} {away_name}</b>"
                                        f" | ⏱️ {minute}' | 🏆 {competition}"
                                        f"</div>", unsafe_allow_html=True)
                                    match_found = True
                                    break

                            if not match_found:
                                live_cache[ck] = {"fetched_at": _time.time(), "match_found": False}
                                save_live_cache(live_cache)
                                st.session_state.live_context = "No live match. Provide pre-match tactical advice."
                                st.warning(
                                    f"⚠️ No live fixture found for **{chat_my_team}** vs **{chat_opp_team}**. "
                                    f"Checked {len(live_data)} live matches worldwide."
                                )
                    except Exception as e:
                        st.error(f"🚨 Connection error: {e}")

        # Clear cache button
        if cached:
            if st.button("🗑️ Clear cached data for this fixture", key="clear_live_cache"):
                live_cache.pop(ck, None)
                save_live_cache(live_cache)
                st.session_state.pop("live_context", None)
                st.rerun()

        st.markdown("---")

        # ── Chat Interface ────────────────────────────────────────────────
        st.markdown("### 🧠 Assistant Manager")

        chat_key = f"messages__{ck}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input(f"Ask your assistant... e.g., 'How do we beat {chat_opp_team}?'"):
            st.session_state[chat_key].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            my_roster   = players_db.get(chat_my_team, [])
            live_status = st.session_state.get("live_context", "No live data. Provide general pre-match tactical advice.")

            history_text = ""
            for msg in st.session_state[chat_key][:-1]:
                role_label = "Coach" if msg["role"] == "user" else "Assistant"
                history_text += f"{role_label}: {msg['content']}\n"

            system_prompt = f"""You are an elite AI Assistant Football Manager.
You are helping the Head Coach of {chat_my_team}, currently facing {chat_opp_team}.

LIVE MATCH STATUS:
{live_status}

OUR SQUAD ROSTER (Name | Position | Minutes Played | Goals+Assists):
{json.dumps(my_roster, ensure_ascii=False)}

CONVERSATION HISTORY:
{history_text if history_text else "This is the start of the briefing."}

INSTRUCTIONS:
- Speak directly to the Head Coach. Be concise, tactical, professional.
- If LIVE MATCH DATA is present, anchor ALL advice to the current score and minute.
- If no live data, give sharp pre-match tactical advice.
- Only reference players from our squad roster above. Never invent names.
- Use football terminology (press triggers, half-spaces, double pivot, low block, etc.)
- 3–6 sentences unless a detailed breakdown is requested.
"""
            if gemini_api_key:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    try:
                        response       = ai_model.generate_content(f"{system_prompt}\n\nCoach: {prompt}")
                        assistant_reply = response.text
                        placeholder.markdown(assistant_reply)
                        st.session_state[chat_key].append({"role": "assistant", "content": assistant_reply})
                    except Exception as e:
                        placeholder.error(f"🚨 Gemini error: {e}")
            else:
                st.error("🚨 GEMINI_API_KEY missing from Streamlit Secrets.")

        if st.session_state.get(chat_key):
            if st.button("🔁 Reset Chat", key="reset_chat"):
                st.session_state[chat_key] = []
                st.rerun()

else:
    st.warning("No teams loaded. Please check your teams.json file.")
