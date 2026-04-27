import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import difflib
import random
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Tactical AI", page_icon="⚽", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0b210e; background-image: linear-gradient(0deg, #0b210e, #0b210e 60px, #0f2b13 60px, #0f2b13 120px); }
h1 { color: #22c55e !important; text-shadow: 0px 0px 10px rgba(34, 197, 94, 0.4); text-transform: uppercase; }
div[data-testid="metric-container"] { background: rgba(0, 0, 0, 0.6); padding: 15px; border-radius: 8px; border-left: 3px solid #22c55e; }
.player-card { background: rgba(0, 0, 0, 0.7); border: 1px solid #22c55e; border-radius: 5px; padding: 10px; margin-bottom: 5px; color: white; display: flex; justify-content: space-between;}
.stat-text { color: #8ca892; font-size: 14px; }
.live-alert { background: rgba(220, 38, 38, 0.2); border-left: 4px solid #dc2626; padding: 15px; border-radius: 5px; color: #f87171; margin-bottom: 15px; }
.live-suggestion { background: rgba(34, 197, 94, 0.2); border-left: 4px solid #22c55e; padding: 15px; border-radius: 5px; color: #86efac; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Tactical AI Engine")

# --- 1. THE UPGRADED AI TACTICAL BRAIN ---
@st.cache_resource
def train_model():
    np.random.seed(42)
    formations_map_internal = {
        0: "3-4-3", 1: "3-5-2", 2: "3-4-1-2", 3: "3-2-4-1", 4: "3-4-2-1", 5: "3-3-1-3",
        6: "4-2-3-1", 7: "4-3-3", 8: "4-4-2", 9: "4-4-2 Diamond", 10: "4-1-4-1", 11: "4-3-2-1", 12: "4-2-2-2",
        13: "5-3-2", 14: "5-4-1", 15: "5-2-2-1", 16: "5-2-3"
    }

    data = {
        'Formation': np.random.randint(0, 17, 4000), 
        'Team_Attack': np.random.randint(50, 99, 4000), 'Team_Defense': np.random.randint(50, 99, 4000),
        'Opp_Attack': np.random.randint(50, 99, 4000), 'Opp_Defense': np.random.randint(50, 99, 4000)
    }
    df = pd.DataFrame(data)

    def get_winner(row):
        score = (row['Team_Attack'] + row['Team_Defense']) - (row['Opp_Attack'] + row['Opp_Defense'])
        f_name = formations_map_internal[row['Formation']]
        
        # Elite vs Elite (Go toe-to-toe)
        if row['Team_Attack'] >= 85 and row['Opp_Attack'] >= 85:
            if f_name in ["4-2-3-1", "4-3-3", "3-2-4-1"]: score += 20
        # Weak vs Elite (Park the bus)
        elif row['Team_Defense'] < 75 and row['Opp_Attack'] >= 85:
            if f_name.startswith('5'): score += 20 
        # Elite vs Weak (Crush them)
        elif row['Team_Attack'] >= 85 and row['Opp_Defense'] < 75:
            if f_name in ["3-4-3", "3-3-1-3", "4-2-4"]: score += 20 
        # Balanced Matchups (Structure wins)
        else:
            if f_name in ["4-4-2", "4-1-4-1", "4-3-2-1"]: score += 15 

        return 1 if score > 0 else 0

    df['Win'] = df.apply(get_winner, axis=1)
    model = RandomForestClassifier().fit(df[['Formation', 'Team_Attack', 'Team_Defense', 'Opp_Attack', 'Opp_Defense']], df['Win'])
    return model, formations_map_internal

model, formations_map = train_model()

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
    st.sidebar.markdown("### NAVIGATION")
    app_mode = st.sidebar.radio(
        "Select Module:",
        (
            "🤖 Pre-Match Auto-Tactics", 
            "📊 Pre-Match Opponent Analysis", 
            "🧠 Coach's Sandbox", 
            "⏱️ Live Match Simulator",
            "💬 Assistant Manager AI"
        )
    )

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
    # MODULE 4: LIVE MATCH SIMULATOR
    # ---------------------------------------------------------
    elif app_mode == "⏱️ Live Match Simulator":
        st.markdown("## ⏱️ Live Match Simulator")
        st.write("Input current match conditions to receive real-time tactical adjustments and substitution alerts.")
        
        col1, col2 = st.columns(2)
        with col1: 
            my_team = st.selectbox("Your Team", list(teams_db.keys()), index=0, key="sim_team")
            match_min = st.slider("Match Minute", 1, 90, 60)
            score_diff = st.selectbox("Current Scoreline", ["Winning Comfortably (+2 goals)", "Leading by 1", "Tied", "Trailing by 1", "Losing Badly (-2 goals)"], index=2)
            
        with col2: 
            opp_team = st.selectbox("Opponent", list(teams_db.keys()), index=1 if len(teams_db) > 1 else 0, key="sim_opp")
            current_form = st.selectbox("Your Current Formation", list(formations_map.values()), index=6)
            match_event = st.selectbox("Current Tactical Problem", [
                "None - Game is balanced", "Midfield is being overrun", "Unable to break down deep block", 
                "Vulnerable to counter-attacks", "Attackers look fatigued", "Defenders struggling with opponent pace",
                "Losing the aerial battle in the box", "Opponent's high press is suffocating us"
            ])

        tactical_db = {
            "Midfield is being overrun": { "tactics": ["⚠️ **Overrun Midfield:** Shift to a formation with 4 or 5 midfielders.", "⚠️ **Numerical Disadvantage:** Instruct your wingers to invert."], "subs": ["🔄 **Sub Alert:** Introduce a fresh defensive midfielder (CDM)."] },
            "Unable to break down deep block": { "tactics": ["⚠️ **Deep Block Detected:** Suggest moving to a wider formation.", "⚠️ **Low Block Frustration:** Instruct fullbacks to overlap aggressively."], "subs": ["🔄 **Sub Alert:** Bring on a tall target-man (FW)."] },
            "Vulnerable to counter-attacks": { "tactics": ["⚠️ **Counter-Attack Risk:** Keep one holding midfielder anchored at all times.", "⚠️ **High Line Exposed:** Drop the defensive line back 10 yards."], "subs": ["🔄 **Sub Alert:** Bring on a dedicated defensive fullback."] },
            "Attackers look fatigued": { "tactics": ["⚠️ **Attacking Fatigue:** Switch from a high press to a mid-block.", "⚠️ **Tired Legs:** Play shorter passes to feet."], "subs": ["🔄 **Sub Alert:** Immediate substitution required for your starting forwards (FW)."] },
            "Defenders struggling with opponent pace": { "tactics": ["⚠️ **Pace Mismatch:** Drop into a deeper defensive block immediately."], "subs": ["🔄 **Sub Alert:** Hook your slowest center-back for a more agile defender."] },
            "Losing the aerial battle in the box": { "tactics": ["⚠️ **Aerial Weakness:** Transition to a zonal marking system on corners."], "subs": ["🔄 **Sub Alert:** Introduce a taller, more physical center-back."] },
            "Opponent's high press is suffocating us": { "tactics": ["⚠️ **Suffocating Press:** Instruct the goalkeeper to go long."], "subs": ["🔄 **Sub Alert:** Introduce a physical target man (FW) to aim long clearances toward."] }
        }

        if st.button("Generate Live Instructions", use_container_width=True, type="primary"):
            if my_team == opp_team:
                st.error("🚨 Tactical Error: A team cannot play against itself!")
            else:
                my_att, my_def = teams_db[my_team]["Attack"], teams_db[my_team]["Defense"]
                opp_att, opp_def = teams_db[opp_team]["Attack"], teams_db[opp_team]["Defense"]
                adjusted_att, adjusted_def = my_att, my_def
                tactic_advice, sub_advice = "", ""
                
                if match_min > 70:
                    if "Trailing" in score_diff or "Losing" in score_diff:
                        adjusted_att += 15; adjusted_def -= 10
                        tactic_advice += "⏳ **Late Game Scenario:** Abandon structural discipline. Transition to an ultra-attacking overload. Push fullbacks extremely high.\n\n"
                    elif "Leading" in score_diff or "Winning" in score_diff:
                        adjusted_def += 15; adjusted_att -= 10
                        tactic_advice += "🛡️ **Protect the Lead:** Drop the defensive line deeper, tighten spaces between lines, and waste time where possible.\n\n"
                
                if match_event != "None - Game is balanced":
                    event_data = tactical_db.get(match_event)
                    if event_data:
                        tactic_advice += random.choice(event_data["tactics"])
                        sub_advice = random.choice(event_data["subs"])

                best_prob, best_form = 0, ""
                for f_code, f_name in formations_map.items():
                    test_match = pd.DataFrame({'Formation': [f_code], 'Team_Attack': [adjusted_att], 'Team_Defense': [adjusted_def], 'Opp_Attack': [opp_att], 'Opp_Defense': [opp_def]})
                    prob = model.predict_proba(test_match)[0][1] * 100
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
    # MODULE 5: ASSISTANT MANAGER AI (GEMINI ENGINE)
    # ---------------------------------------------------------
    elif app_mode == "💬 Assistant Manager AI":
        st.markdown("## 💬 Tactical AI Assistant")
        st.write("Select your matchup. The AI will draw the latest stats from the database to generate customized advice.")
        
        # Dedicated matchup selectors for the AI chat
        col1, col2 = st.columns(2)
        with col1: 
            ai_my_team = st.selectbox("Your Team (AI Focus)", list(teams_db.keys()), index=0, key="ai_my_team")
        with col2: 
            ai_opp_team = st.selectbox("Opponent", list(teams_db.keys()), index=1 if len(teams_db) > 1 else 0, key="ai_opp_team")
            
        st.markdown("---")

        # The explicit "Sync" button requirement
        if st.button("📡 Sync Live Match Data", use_container_width=True, type="primary"):
            st.session_state.ai_synced = True
            st.success(f"✅ Live tactical data for {ai_my_team} and {ai_opp_team} successfully synced to the AI Engine! You may now brief the Assistant Manager.")

        # The chat interface ONLY opens after the button is pressed
        if st.session_state.get("ai_synced", False):
            # Initialize Gemini Setup safely
            gemini_ready = False
            try:
                if "GEMINI_API_KEY" in st.secrets:
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                    # Changed from gemini-1.5-flash to the ultra-stable gemini-pro
                    model_ai = genai.GenerativeModel('gemini-pro')
                    gemini_ready = True
                else:
                    st.warning("⚠️ GEMINI_API_KEY not found in Streamlit Secrets. The AI is offline.")
            except Exception as e:
                st.error(f"Failed to load AI: {e}")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input(f"E.g., How do I set up {ai_my_team} to counter {ai_opp_team}?"):
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Generate AI response 
                with st.chat_message("assistant"):
                    if gemini_ready:
                        # Pull real data from your databases to feed the AI context
                        my_att = teams_db.get(ai_my_team, {}).get("Attack", 80)
                        my_def = teams_db.get(ai_my_team, {}).get("Defense", 80)
                        opp_att = teams_db.get(ai_opp_team, {}).get("Attack", 80)
                        opp_def = teams_db.get(ai_opp_team, {}).get("Defense", 80)
                        
                        # Create the invisible system prompt framing the match state
                        system_prompt = f"""
                        You are an elite football tactical Assistant Manager. Provide direct, tactical advice. 
                        We are managing {ai_my_team} (Attack Rating: {my_att}, Defense Rating: {my_def}).
                        We are playing against {ai_opp_team} (Attack Rating: {opp_att}, Defense Rating: {opp_def}).
                        The Manager asks: "{prompt}"
                        """
                        
                        try:
                            response = model_ai.generate_content(system_prompt)
                            ai_reply = response.text
                            st.markdown(ai_reply)
                            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                        except Exception as e:
                            st.error(f"Error communicating with Gemini: {e}")
                    else:
                        st.error("The Gemini Engine is offline. Please check your API keys.")

else:
    st.warning("No teams loaded. Please check your teams.json file.")
