import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import difflib
import random

# Page setup
st.set_page_config(page_title="Tactical AI", page_icon="⚽", layout="wide")

# Custom CSS for the green football theme
st.markdown("""
<style>
.stApp {
    background-color: #0b210e;
    background-image: repeating-linear-gradient(0deg, #0b210e, #0b210e 60px, #0f2b13 60px, #0f2b13 120px);
}
h1, h2, h3 {
    color: #22c55e !important;
    font-family: 'Courier New', Courier, monospace;
}
.stButton>button {
    background-color: #22c55e !important;
    color: white !important;
    border-radius: 20px;
    border: 2px solid #15803d;
}
.live-suggestion {
    background-color: #1e3a1e;
    padding: 15px;
    border-left: 5px solid #22c55e;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    try:
        with open('teams.json', 'r', encoding='utf-8') as f:
            teams_data = json.load(f)
    except FileNotFoundError:
        teams_data = {}
        
    try:
        with open('players.json', 'r', encoding='utf-8') as f:
            players_data = json.load(f)
    except FileNotFoundError:
        players_data = {}
        
    return teams_data, players_data

teams_db, players_db = load_data()

# Load trained ML model if available
try:
    import pickle
    with open('tactical_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        formations_map = model_data['formations_map']
except Exception:
    model, formations_map = None, {}

# --- FORMATION-AWARE POSITION-DRAFTING LOGIC ---
def select_starting_xi(team_name, formation_string, players_db):
    if team_name not in players_db or not players_db[team_name]:
        return []
        
    players_list = players_db[team_name]
    # Parse numbers out of formation string (e.g., "4-3-3" -> [4, 3, 3])
    formation_numbers = [int(n) for n in re.findall(r'\d+', formation_string)]
    
    if len(formation_numbers) == 3:
        num_df, num_mf, num_fw = formation_numbers
    elif len(formation_numbers) == 4:  # e.g., 4-2-3-1 structure
        num_df = formation_numbers[0]
        num_mf = formation_numbers[1] + formation_numbers[2]
        num_fw = formation_numbers[3]
    else:
        num_df, num_mf, num_fw = 4, 4, 2  # Standard safety fallback
        
    requirements = {'GK': 1, 'DF': num_df, 'MF': num_mf, 'FW': num_fw}
    
    # Sort squad primarily by total goal contributions (G_A) and minutes played
    sorted_squad = sorted(players_list, key=lambda x: x.get('G_A', 0) + (x.get('Mins', 0) / 900), reverse=True)
    
    starting_xi = []
    assigned_ids = set()
    
    # Check if current formation relies on true wingers (e.g., 3-back or 4-3-3 attacking shapes)
    is_winger_formation = (formation_string in ["4-3-3", "3-4-3", "3-5-2"])
    
    # Phase 1: Smart Contextual Assignment (Handles flexible profiles like Saka)
    for pos in ['GK', 'DF', 'MF', 'FW']:
        for p in sorted_squad:
            if p['Name'] in assigned_ids:
                continue
                
            pos_tags = [t.strip().upper() for t in p.get('Pos', '').split(',')]
            primary_pos = pos_tags[0] if pos_tags else 'MF'
            
            # If the player is an MF,FW hybrid and we need a forward in a winger setup, slide them to FW
            if pos == 'FW' and is_winger_formation and 'FW' in pos_tags and 'MF' in pos_tags:
                if requirements['FW'] > 0:
                    starting_xi.append({'Name': p['Name'], 'Pos': 'FW', 'G_A': p.get('G_A', 0)})
                    assigned_ids.add(p['Name'])
                    requirements['FW'] -= 1
                    continue
            
            # Standard matching based on their primary position tag
            if primary_pos == pos and requirements[pos] > 0:
                starting_xi.append({'Name': p['Name'], 'Pos': pos, 'G_A': p.get('G_A', 0)})
                assigned_ids.add(p['Name'])
                requirements[pos] -= 1

    # Phase 2: Structural Backup Assignment (Fill remainder using secondary tags)
    for pos, needed in requirements.items():
        if needed <= 0:
            continue
        for p in sorted_squad:
            if p['Name'] in assigned_ids:
                continue
            pos_tags = [t.strip().upper() for t in p.get('Pos', '').split(',')]
            if pos in pos_tags and requirements[pos] > 0:
                starting_xi.append({'Name': p['Name'], 'Pos': pos, 'G_A': p.get('G_A', 0)})
                assigned_ids.add(p['Name'])
                requirements[pos] -= 1

    # Phase 3: Roster Padding (Prevents blank spots if data records run low)
    for pos, needed in requirements.items():
        while requirements[pos] > 0:
            available = [p for p in sorted_squad if p['Name'] not in assigned_ids]
            if not available:
                break
            p = available[0]
            starting_xi.append({'Name': p['Name'], 'Pos': f"{pos}⚠️", 'G_A': p.get('G_A', 0)})
            assigned_ids.add(p['Name'])
            requirements[pos] -= 1
            
    return starting_xi


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("NAVIGATION")
app_mode = st.sidebar.radio(
    "Select Module:",
    (
        "🤖 Pre-Match Auto-Tactics", 
        "📊 Pre-Match Opponent Analysis", 
        "🧠 Coach's Sandbox", 
        "⏱️ Live Match Simulator",
        "💬 Assistant Manager Chat"
    )
)

if not teams_db:
    st.warning("Please ensure teams.json and players.json are generated and valid.")
    st.stop()

# Get available teams list
available_teams = list(teams_db.keys())

# ---------------------------------------------------------
# MODULE 1: PRE-MATCH AUTO-TACTICS
# ---------------------------------------------------------
if app_mode == "🤖 Pre-Match Auto-Tactics":
    st.title("🤖 Pre-Match Auto-Tactics")
    st.write("Let the ML model analyze ratings and auto-generate the optimal lineup and strategy.")
    
    col1, col2 = st.columns(2)
    with col1:
        my_team = st.selectbox("Select Your Team:", available_teams, index=0)
    with col2:
        opp_team = st.selectbox("Select Opponent:", available_teams, index=1 if len(available_teams)>1 else 0)
        
    if st.button("Generate Optimal Tactics", use_container_width=True):
        if not model:
            st.error("No trained model found. Please train the model first using the training script.")
        else:
            my_stats = teams_db[my_team]
            opp_stats = teams_db[opp_team]
            
            # Predict best formation using random feature sample format for testing
            sample_features = [[6, my_stats['Attack'], my_stats['Defense'], opp_stats['Attack'], opp_stats['Defense']]]
            pred_class = model.predict(sample_features)[0]
            best_formation = formations_map.get(pred_class, "4-4-2")
            
            st.success(f"🎯 Recommended Formation: **{best_formation}**")
            
            # Select Starting XI using our updated formation-aware selector
            starting_xi = select_starting_xi(my_team, best_formation, players_db)
            
            if starting_xi:
                st.subheader(f"📋 Suggested Starting XI ({best_formation})")
                df_xi = pd.DataFrame(starting_xi)
                st.table(df_xi)
            else:
                st.warning("Could not automatically generate lineup. Check your squad depth.")

# ---------------------------------------------------------
# MODULE 2: PRE-MATCH OPPONENT ANALYSIS
# ---------------------------------------------------------
elif app_mode == "📊 Pre-Match Opponent Analysis":
    st.title("📊 Pre-Match Opponent Analysis")
    st.write("Analyze your opponent's structural flaws, key threats, and vulnerabilities.")
    
    target_opp = st.selectbox("Select Opponent to Analyze:", available_teams)
    opp_stats = teams_db[target_opp]
    
    st.subheader(f"Team Stats for {target_opp}")
    st.metric("Attack Rating", opp_stats['Attack'])
    st.metric("Defense Rating", opp_stats['Defense'])
    
    st.markdown("### ⚠️ Strategic Vulnerabilities")
    if opp_stats['Defense'] < 75:
        st.write("🔴 Low defensive rating. Recommend heavy high-press and vertical counter-attacks.")
    elif opp_stats['Attack'] > 85:
        st.write("🟠 Dangerous attacking capacity. Recommend low block defensive containment strategies.")
    else:
        st.write("🟢 Balanced team structure. Control possession and look for overlapping wing play.")

# ---------------------------------------------------------
# MODULE 3: COACH'S SANDBOX
# ---------------------------------------------------------
elif app_mode == "🧠 Coach's Sandbox":
    st.title("🧠 Coach's Sandbox")
    st.write("Manually override setups and test structural win probabilities against opponents.")
    
    col1, col2 = st.columns(2)
    with col1:
        my_team = st.selectbox("Your Team:", available_teams, key="sb_my")
        chosen_formation = st.selectbox("Select Formation:", ["4-4-2", "4-3-3", "4-2-3-1", "3-5-2", "5-3-2"])
    with col2:
        opp_team = st.selectbox("Opponent Team:", available_teams, key="sb_opp")
        
    if st.button("Calculate Tactical Probability"):
        my_stats = teams_db[my_team]
        opp_stats = teams_db[opp_team]
        
        # Calculate mock win probability based on structural stats gap
        base_prob = 50 + (my_stats['Attack'] - opp_stats['Defense']) + (my_stats['Defense'] - opp_stats['Attack'])
        win_prob = max(10, min(95, base_prob))
        
        st.metric("Estimated Win Probability", f"{win_prob}%")
        
        # Generate starting XI using our updated formation-aware selector
        xi = select_starting_xi(my_team, chosen_formation, players_db)
        if xi:
            st.subheader("Selected Starting XI")
            st.dataframe(pd.DataFrame(xi))

# ---------------------------------------------------------
# MODULE 4: LIVE MATCH SIMULATOR
# ---------------------------------------------------------
elif app_mode == "⏱️ Live Match Simulator":
    st.title("⏱️ Live Match Simulator")
    st.write("Simulate an artificial interactive 90-minute fixture timeline.")
    
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Home Team:", available_teams, key="sim_a")
    with col2:
        team_b = st.selectbox("Away Team:", available_teams, key="sim_b")
        
    if st.button("Launch Simulation Engine"):
        stats_a = teams_db[team_a]
        stats_b = teams_db[team_b]
        
        score_a = random.randint(0, 3) if stats_a['Attack'] > stats_b['Defense'] else random.randint(0, 1)
        score_b = random.randint(0, 3) if stats_b['Attack'] > stats_a['Defense'] else random.randint(0, 1)
        
        st.header(f"🏆 Final Result: {team_a} {score_a} - {score_b} {team_b}")
        st.balloons()

# ---------------------------------------------------------
# MODULE 5: ASSISTANT MANAGER CHAT
# ---------------------------------------------------------
elif app_mode == "💬 Assistant Manager Chat":
    st.title("💬 Assistant Manager Chat")
    st.write("Debate tactics, ask for substitution advice, or question the starting XI based on live data.")
    
    col1, col2 = st.columns(2)
    with col1:
        ai_my_team = st.selectbox("Your Team", available_teams, key="chat_my")
    with col2:
        ai_opp_team = st.selectbox("Opponent", available_teams, key="chat_opp")
        
    # Standard warning box placeholder
    st.warning(f"⚠️ {ai_my_team} is not currently playing a live professional fixture.")
    
    # Simple message window fallback
    st.chat_input("Ask your assistant... e.g., 'How do we break down the opposition?'")
