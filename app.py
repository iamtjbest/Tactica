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
.stApp { background-color: #0b210e; background-image: repeating-linear-gradient(0deg, #0b210e, #0b210e 60px, #0f2b13 60px, #0f2b13 120px); }
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


# --- 3. ADVANCED PLAYER SELECTION ALGORITHM ---
def select_starting_xi(team_name, formation):
    if team_name not in players_db:
        close_matches = difflib.get_close_matches(team_name, players_db.keys(), n=1, cutoff=0.6)
        if close_matches:
            team_name = close_matches[0]
        else:
            return None
    
    parts = [int(x) for x in re.findall(r'\d+', formation)]
    def_count, att_count = parts[0], parts[-1]
    mid_count = sum(parts[1:-1]) if len(parts) > 2 else parts[1]

    roster = players_db[team_name]
    sorted_roster = sorted(roster, key=lambda x: (x['Min'], x['G_A']), reverse=True)
    
    starting_xi = []
    drafted_names = set()

    def draft_players(pos_keyword, count):
        drafted = 0
        for p in sorted_roster:
            if drafted >= count:
                break
            if pos_keyword in p['Pos'] and p['Name'] not in drafted_names:
                starting_xi.append(p)
                drafted_names.add(p['Name'])
                drafted += 1

    draft_players('GK', 1)
    draft_players('DF', def_count)
    draft_players('MF', mid_count)
    draft_players('FW', att_count)
    
    for p in sorted_roster:
        if len(starting_xi) >= 11:
            break
        if p['Name'] not in drafted_names:
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
        st.markdown("## 🤖 Pre-Match Auto-Tactics")
        st.write("Let the engine analyze the matchup and generate the optimal starting game plan.")
        
        col1, col2 = st.columns(2)
        with col1: my_team = st.selectbox("Your Team", list(teams_db.keys()), index=0) 
        with col2: opp_team = st.selectbox("Opponent", list(teams_db.keys()), index=1 if len(teams_db) > 1 else 0)

        if st.button("Generate Optimal Tactics", use_container_width=True, type="primary"):
            if my_team == opp_team:
                st.error("🚨 Tactical Error: A team cannot play against itself!")
            else:
                my_att, my_def = teams_db[my_team]["Attack"], teams_db[my_team]["Defense"]
                opp_att, opp_def = teams_db[opp_team]["Attack"], teams_db[opp_team]["Defense"]
                
                best_prob, best_form = 0, ""
                for f_code, f_name in formations_map.items():
                    test_match = pd.DataFrame({'Formation': [f_code], 'Team_Attack': [my_att], 'Team_Defense': [my_def], 'Opp_Attack': [opp_att], 'Opp_Defense': [opp_def]})
                    prob = model.predict_proba(test_match)[0][1] * 100
                    if prob > best_prob: best_prob, best_form = prob, f_name
                        
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Recommended Formation", best_form)
                res_col2.metric("AI Win Probability", f"{best_prob:.1f}%")

                st.markdown("### AI Recommended Starting XI")
                xi = select_starting_xi(my_team, best_form)
                if xi:
                    for p in xi:
                        st.markdown(f"<div class='player-card'><span><b>{p['Pos']}</b> | {p['Name']}</span> <span class='stat-text'>⏱️ {p['Min']} mins | ⚽ {p['G_A']} G+A</span></div>", unsafe_allow_html=True)
                else:
                    st.warning(f"No player data found in players.json for '{my_team}'.")

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
    # MODULE 5: ASSISTANT MANAGER CHAT (with Live Match Intel)
    # ---------------------------------------------------------
    elif app_mode == "💬 Assistant Manager Chat":
        import time as _time
        import requests as _requests

        st.markdown("## 💬 Assistant Manager Chat")
        st.write("Select your teams, sync live match data from any competition, then chat with your AI assistant manager.")

        # ── Team selectors ────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            chat_my_team = st.selectbox("Your Team", list(teams_db.keys()), index=0, key="chat_team")
        with col2:
            chat_opp_team = st.selectbox("Opponent", list(teams_db.keys()), index=1 if len(teams_db) > 1 else 0, key="chat_opp")

        st.markdown("---")

        # ── Live Match Intel Panel ────────────────────────────────
        st.markdown("### 📡 Live Match Intel")

        LIVE_CACHE_FILE = "live_match_cache.json"
        CACHE_TTL_SECONDS = 300  # 5 minutes — read from cache before hitting API again

        def load_live_cache():
            """Load the on-disk live match cache."""
            try:
                with open(LIVE_CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}

        def save_live_cache(data: dict):
            """Persist live match data to disk so the sync button reads cache first."""
            try:
                with open(LIVE_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass  # Don't crash the app over a cache write failure

        def build_cache_key(team_a: str, team_b: str) -> str:
            return f"{team_a.lower().strip()}__vs__{team_b.lower().strip()}"

        # Check cache first
        live_cache = load_live_cache()
        cache_key = build_cache_key(chat_my_team, chat_opp_team)
        cached_entry = live_cache.get(cache_key)

        cache_age_seconds = None
        if cached_entry:
            cache_age_seconds = _time.time() - cached_entry.get("fetched_at", 0)
            cache_is_fresh = cache_age_seconds < CACHE_TTL_SECONDS
        else:
            cache_is_fresh = False

        # Resolve what live context to use
        if cache_is_fresh and cached_entry.get("match_found"):
            # Fresh cache hit — display without an API call
            d = cached_entry
            mins_old = int(cache_age_seconds / 60)
            st.markdown(
                f"""<div class='live-suggestion'>
                <b>✅ LIVE: {d['home_name']} {d['home_goals']} – {d['away_goals']} {d['away_name']}</b>
                &nbsp;|&nbsp; ⏱️ Minute {d['minute']}' &nbsp;|&nbsp; 🏆 {d['competition']}
                <br><span style='font-size:12px;color:#86efac'>Data cached {mins_old} min ago — sync to refresh</span>
                </div>""",
                unsafe_allow_html=True,
            )
            live_context_for_ai = (
                f"LIVE MATCH DATA ({d['competition']}): "
                f"Match Minute {d['minute']}'. "
                f"Current Scoreline: {d['home_name']} {d['home_goals']} – {d['away_goals']} {d['away_name']}. "
                f"Venue: {d.get('venue', 'Unknown')}."
            )
            st.session_state.live_context = live_context_for_ai

        elif cached_entry and not cached_entry.get("match_found"):
            mins_old = int(cache_age_seconds / 60) if cache_age_seconds else "?"
            st.info(f"ℹ️ Last sync ({mins_old} min ago): {chat_my_team} was not in a live fixture across any competition.")
            st.session_state.live_context = "No live match currently in progress for this team. Advise based on pre-match context."

        else:
            st.info("No live data cached for this fixture. Hit **Sync** to search all live competitions.")
            if "live_context" not in st.session_state:
                st.session_state.live_context = "No live data synced yet. Provide general tactical advice."

        # ── Sync button with cooldown ─────────────────────────────
        last_sync_time = st.session_state.get("last_live_sync", 0)
        seconds_since_sync = _time.time() - last_sync_time
        SYNC_COOLDOWN = 300  # 5 min between API hits

        sync_ready = seconds_since_sync >= SYNC_COOLDOWN
        if sync_ready:
            btn_label = "🔄 Sync Live Data (All Competitions)"
        else:
            mins_left = max(1, int((SYNC_COOLDOWN - seconds_since_sync) / 60))
            btn_label = f"🔄 Sync Live Data (cooldown: ~{mins_left} min)"

        if st.button(btn_label, use_container_width=True, disabled=not sync_ready):
            api_sports_key = st.secrets.get("API_SPORTS_KEY")
            if not api_sports_key:
                st.error("🚨 API_SPORTS_KEY is missing from Streamlit Secrets! Add it under Settings → Secrets.")
            else:
                with st.spinner("🌐 Scanning all live fixtures across global competitions..."):
                    headers = {
                        "x-apisports-key": api_sports_key,
                        "x-apisports-host": "v3.football.api-sports.io"
                    }
                    try:
                        # Single API call — fetches ALL live matches worldwide (UCL, EPL, La Liga, etc.)
                        res = _requests.get(
                            "https://v3.football.api-sports.io/fixtures?live=all",
                            headers=headers,
                            timeout=12
                        )
                        st.session_state.last_live_sync = _time.time()

                        if res.status_code == 429:
                            st.error("🚨 Daily API quota exhausted. Cache will be used until midnight UTC resets your limit.")
                        elif res.status_code != 200:
                            st.error(f"🚨 API returned status {res.status_code}. Verify your API key in Streamlit Secrets.")
                        else:
                            live_data = res.json().get("response", [])
                            match_found = False

                            for match in live_data:
                                home_name  = match["teams"]["home"]["name"]
                                away_name  = match["teams"]["away"]["name"]
                                home_id    = match["teams"]["home"]["id"]
                                away_id    = match["teams"]["away"]["id"]

                                # Fuzzy match — works for partial names and alternate spellings
                                my_team_hits  = (chat_my_team.lower() in home_name.lower() or
                                                 chat_my_team.lower() in away_name.lower() or
                                                 home_name.lower() in chat_my_team.lower() or
                                                 away_name.lower() in chat_my_team.lower())
                                opp_team_hits = (chat_opp_team.lower() in home_name.lower() or
                                                 chat_opp_team.lower() in away_name.lower() or
                                                 home_name.lower() in chat_opp_team.lower() or
                                                 away_name.lower() in chat_opp_team.lower())

                                if my_team_hits and opp_team_hits:
                                    minute      = match["fixture"]["status"].get("elapsed") or 0
                                    home_goals  = match["goals"]["home"] if match["goals"]["home"] is not None else 0
                                    away_goals  = match["goals"]["away"] if match["goals"]["away"] is not None else 0
                                    competition = match.get("league", {}).get("name", "Unknown Competition")
                                    venue       = match.get("fixture", {}).get("venue", {}).get("name", "Unknown Venue")
                                    status_long = match["fixture"]["status"].get("long", "Live")

                                    # Persist to cache
                                    entry = {
                                        "fetched_at":  _time.time(),
                                        "match_found": True,
                                        "home_name":   home_name,
                                        "away_name":   away_name,
                                        "home_goals":  home_goals,
                                        "away_goals":  away_goals,
                                        "minute":      minute,
                                        "competition": competition,
                                        "venue":       venue,
                                        "status":      status_long,
                                    }
                                    live_cache[cache_key] = entry
                                    save_live_cache(live_cache)

                                    live_context_str = (
                                        f"LIVE MATCH DATA ({competition}): "
                                        f"Match Minute {minute}'. "
                                        f"Current Scoreline: {home_name} {home_goals} – {away_goals} {away_name}. "
                                        f"Venue: {venue}. Status: {status_long}."
                                    )
                                    st.session_state.live_context = live_context_str

                                    st.markdown(
                                        f"""<div class='live-suggestion'>
                                        <b>✅ LIVE: {home_name} {home_goals} – {away_goals} {away_name}</b>
                                        &nbsp;|&nbsp; ⏱️ Minute {minute}' &nbsp;|&nbsp; 🏆 {competition}
                                        <br><span style='font-size:12px;color:#86efac'>📍 {venue} &nbsp;·&nbsp; {status_long} &nbsp;·&nbsp; Saved to cache</span>
                                        </div>""",
                                        unsafe_allow_html=True,
                                    )
                                    match_found = True
                                    break

                            if not match_found:
                                # Cache the "no match" result too — avoids repeat API calls
                                live_cache[cache_key] = {
                                    "fetched_at":  _time.time(),
                                    "match_found": False,
                                }
                                save_live_cache(live_cache)
                                st.session_state.live_context = "No live match in progress for this fixture. Advise on pre-match / general context."
                                st.warning(
                                    f"⚠️ No live fixture found for **{chat_my_team}** vs **{chat_opp_team}** "
                                    f"across any competition right now. Checked {len(live_data)} live matches worldwide."
                                )

                    except Exception as e:
                        st.error(f"🚨 Connection error: {e}")

        # ── Clear cache button (small, secondary) ─────────────────
        if cached_entry:
            if st.button("🗑️ Clear cached data for this fixture", key="clear_cache"):
                live_cache.pop(cache_key, None)
                save_live_cache(live_cache)
                st.session_state.pop("live_context", None)
                st.rerun()

        st.markdown("---")

        # ── Chat Interface ────────────────────────────────────────
        st.markdown("### 🧠 Assistant Manager")

        # Per-tab isolated chat history (keyed to the team pairing)
        chat_session_key = f"messages__{cache_key}"
        if chat_session_key not in st.session_state:
            st.session_state[chat_session_key] = []

        # Render existing messages
        for message in st.session_state[chat_session_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input(f"Ask your assistant... e.g., 'How do we beat {chat_opp_team}?'"):
            st.session_state[chat_session_key].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            my_roster   = players_db.get(chat_my_team, [])
            live_status = st.session_state.get("live_context", "No live data. Provide general pre-match tactical advice.")

            # Build full conversation history for the AI (true multi-turn memory)
            history_text = ""
            for msg in st.session_state[chat_session_key][:-1]:  # exclude the message just added
                role_label = "Coach" if msg["role"] == "user" else "Assistant"
                history_text += f"{role_label}: {msg['content']}\n"

            system_instruction = f"""You are an elite, world-class Assistant Football Manager AI.
You are assisting the Head Coach of {chat_my_team}, currently facing {chat_opp_team}.

LIVE MATCH STATUS:
{live_status}

OUR SQUAD ROSTER (Name | Position | Minutes Played | Goals+Assists):
{json.dumps(my_roster, ensure_ascii=False)}

CONVERSATION HISTORY (for context — do not repeat already-given advice):
{history_text if history_text else "This is the start of the briefing."}

INSTRUCTIONS:
- Speak directly to the Head Coach. Be concise, tactical, and professional.
- If LIVE MATCH DATA is present, anchor ALL advice to the current scoreline and match minute.
- If no live data, give sharp pre-match tactical advice based on known team strengths.
- Reference ONLY players from our squad roster above. Never invent player names.
- Keep responses focused — 3 to 6 sentences unless the coach asks for a detailed breakdown.
- Use football terminology (press triggers, half-spaces, double pivot, low block, etc.).
"""

            if gemini_api_key:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    try:
                        full_prompt = f"{system_instruction}\n\nCoach's Question: {prompt}"
                        response = ai_model.generate_content(full_prompt)
                        assistant_reply = response.text
                        message_placeholder.markdown(assistant_reply)
                        st.session_state[chat_session_key].append({"role": "assistant", "content": assistant_reply})
                    except Exception as e:
                        message_placeholder.error(f"🚨 AI error: {e}")
            else:
                st.error("🚨 GEMINI_API_KEY is missing from Streamlit Secrets. The AI chat requires it.")

        # Reset chat button
        if st.session_state.get(chat_session_key):
            if st.button("🔁 Reset Chat", key="reset_chat"):
                st.session_state[chat_session_key] = []
                st.rerun()


else:
    st.warning("No teams loaded. Please check your teams.json file.")
