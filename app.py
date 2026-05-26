import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import difflib
import random
import requests
import pickle
from datetime import datetime
import google.generativeai as genai

# Page Config
st.set_page_config(page_title="TACTICAL AI ENGINE", page_icon="⚽", layout="wide")

# Theme & Font Injection (Consistent Dark Green Theme with Glow Accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #050d07 0%, #0d1f10 100%) !important;
        font-family: 'Rajdhani', sans-serif !important;
        color: #e2e8f0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #22c55e !important;
        text-shadow: 0 0 10px rgba(34, 197, 94, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #16a34a 0%, #22c55e 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 4px !important;
        box-shadow: 0 0 15px rgba(34, 197, 94, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 25px rgba(34, 197, 94, 0.7);
    }
    
    .live-scoreboard {
        background: rgba(13, 31, 16, 0.8);
        border: 2px solid #22c55e;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.2);
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Load local data
def load_local_data():
    try:
        with open("teams.json", "r") as f:
            teams = json.load(f)
        with open("players.json", "r") as f:
            players = json.load(f)
    except:
        teams, players = {}, {}
    return teams, players

teams_db, players_db = load_local_data()

# Helper: Load ML Model
@st.cache_resource
def load_saved_model():
    try:
        with open("tactical_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

model_package = load_saved_model()

# --- FORMATION-AWARE POSITION-DRAFTING ENGINE ---
def select_starting_xi(team_name, formation_string, players_db):
    if team_name not in players_db or not players_db[team_name]:
        return []
        
    players_list = players_db[team_name]
    # Parse formation numbers (e.g., "4-3-3" -> [4, 3, 3])
    formation_numbers = [int(n) for n in re.findall(r'\d+', formation_string)]
    
    if len(formation_numbers) == 3:
        num_df, num_mf, num_fw = formation_numbers
    elif len(formation_numbers) == 4:  # e.g., 4-2-3-1
        num_df = formation_numbers[0]
        num_mf = formation_numbers[1] + formation_numbers[2]
        num_fw = formation_numbers[3]
    else:
        num_df, num_mf, num_fw = 4, 4, 2  # Default fallback
        
    requirements = {'GK': 1, 'DF': num_df, 'MF': num_mf, 'FW': num_fw}
    
    # Sort players primarily by performance metric
    sorted_squad = sorted(players_list, key=lambda x: x.get('G_A', 0) + (x.get('Mins', 0) / 900), reverse=True)
    
    starting_xi = []
    assigned_ids = set()
    
    # Context Phase: Establish tactical properties of the formation layout
    is_winger_formation = (formation_string in ["4-3-3", "3-4-3", "3-5-2"])
    
    # Phase 1: Contextual Primary Slotting (Handles flexible profiles like Saka)
    for pos in ['GK', 'DF', 'MF', 'FW']:
        for p in sorted_squad:
            if p['Name'] in assigned_ids:
                continue
                
            pos_tags = [t.strip().upper() for t in p.get('Pos', '').split(',')]
            primary_pos = pos_tags[0] if pos_tags else 'MF'
            
            # Smart logic: If a player is MF,FW and we need FW in a winger setup, slide them to FW
            if pos == 'FW' and is_winger_formation and 'FW' in pos_tags and 'MF' in pos_tags:
                if requirements['FW'] > 0:
                    starting_xi.append({'Name': p['Name'], 'Pos': 'FW', 'G_A': p.get('G_A', 0)})
                    assigned_ids.add(p['Name'])
                    requirements['FW'] -= 1
                    continue
            
            # Standard matching based on primary tag listing
            if primary_pos == pos and requirements[pos] > 0:
                starting_xi.append({'Name': p['Name'], 'Pos': pos, 'G_A': p.get('G_A', 0)})
                assigned_ids.add(p['Name'])
                requirements[pos] -= 1

    # Phase 2: Structural Fallback Slotting (Fill leftover roles with multi-position players)
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

    # Phase 3: Pure Padding (Avoid leaving lineup empty due to strict position scarcity)
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

# --- NAVIGATION ROUTING ---
st.sidebar.markdown("<h1>📊 TACTICA MENU</h1>", unsafe_allow_html=True)
app_mode = st.sidebar.radio("Navigate Modules:", [
    "🤖 Pre-Match Auto-Tactics",
    "📊 Pre-Match Opponent Analysis",
    "🧠 Coach's Sandbox",
    "⏱️ Live Match Simulator",
    "💬 Assistant Manager Chat"
])

# Expanded Extended European Coverage Array (~130 Club Coverage Capacity)
european_clubs = sorted([
    "Arsenal", "Aston Villa", "Chelsea", "Liverpool", "Manchester City", "Manchester Utd", "Tottenham", "Newcastle", "West Ham", "Everton",
    "Real Madrid", "Barcelona", "Atletico Madrid", "Real Sociedad", "Villarreal", "Sevilla", "Real Betis", "Girona", "Athletic Club",
    "Bayern Munich", "Borussia Dortmund", "Bayer Leverkusen", "RB Leipzig", "Eintracht Frankfurt", "VfL Wolfsburg", "VfB Stuttgart",
    "Paris Saint-Germain", "Monaco", "Marseille", "Lyon", "Lille", "Lens", "Nice", "Rennes",
    "Inter Milan", "AC Milan", "Juventus", "Napoli", "AS Roma", "Lazio", "Atalanta", "Fiorentina", "Bologna",
    "Benfica", "FC Porto", "Sporting CP", "Ajax", "PSV Eindhoven", "Feyenoord", "Celtic", "Rangers", "Club Brugge", "Galatasaray", "Fenerbahce"
])

# --- MODULE 1: PRE-MATCH AUTO-TACTICS ---
if app_mode == "🤖 Pre-Match Auto-Tactics":
    st.markdown("## 🤖 AUTOMATED TACTICS LOGIC")
    col1, col2 = st.columns(2)
    with col1:
        my_team = st.selectbox("Your Squad Focus", european_clubs, index=0)
    with col2:
        opp_team = st.selectbox("Opponent Squad Focus", european_clubs, index=1)
        
    if st.button("Generate Tactical Lineup", use_container_width=True):
        if model_package:
            # Gather static values
            my_rating = teams_db.get(my_team, {"Attack": 80, "Defense": 80})
            opp_rating = teams_db.get(opp_team, {"Attack": 75, "Defense": 75})
            
            # Predict
            features = np.array([[6, my_rating['Attack'], my_rating['Defense'], opp_rating['Attack'], opp_rating['Defense']]])
            pred_form = model_package["formations_map"].get(model_package["model"].predict(features)[0], "4-3-3")
            
            st.success(f"Recommended System Structure: {pred_form}")
            
            # Generate starting XI using context framework
            xi = select_starting_xi(my_team, pred_form, players_db)
            if xi:
                st.markdown("### Selected Starting Lineup")
                df_xi = pd.DataFrame(xi)
                st.dataframe(df_xi, use_container_width=True)
            else:
                st.warning("Please run auto_updater.py script first to synchronize player data vectors.")
        else:
            st.error("Model array package `tactical_model.pkl` not detected.")

# --- MODULE 5: ASSISTANT MANAGER CHAT (BSD API DATA FETCHING) ---
elif app_mode == "💬 Assistant Manager Chat":
    st.markdown("## 💬 ASSISTANT MANAGER CHAT ENGINE")
    
    col1, col2 = st.columns(2)
    with col1:
        chat_my_team = st.selectbox("Select Your Focus Team", european_clubs, key="cm")
    with col2:
        chat_opp_team = st.selectbox("Select Opponent Target", european_clubs, key="co")
        
    if st.button("📡 Synchronize Live Fixture Streams", use_container_width=True):
        bsd_key = st.secrets.get("BSD_API_KEY")
        if not bsd_key:
            st.error("Critical Token Error: `BSD_API_KEY` configuration variable missing inside system settings.")
        else:
            with st.spinner("Connecting to BSD Global Database Engines..."):
                headers = {"Authorization": f"Bearer {bsd_key}"}
                
                # Fetch dynamically via name payload
                url = f"https://sports.bzzoiro.com/api/v2/events/live/"
                try:
                    res = requests.get(url, headers=headers).json()
                    live_events = res.get("data", [])
                    
                    found_match = None
                    for event in live_events:
                        home_name = event.get("home_team", {}).get("name", "").lower()
                        away_name = event.get("away_team", {}).get("name", "").lower()
                        
                        if chat_my_team.lower() in home_name or chat_my_team.lower() in away_name:
                            found_match = event
                            break
                            
                    if found_match:
                        home = found_match["home_team"]["name"]
                        away = found_match["away_team"]["name"]
                        h_score = found_match.get("home_score", 0)
                        a_score = found_match.get("away_score", 0)
                        minute = found_match.get("minute", "FT")
                        league = found_match.get("league", {}).get("name", "UCL / Domestic Cup")
                        
                        st.session_state.live_match_context = f"{home} {h_score} - {a_score} away | Minute: {minute}' | Competition: {league}"
                        st.session_state.ai_synced = True
                    else:
                        st.session_state.live_match_context = f"Offline Match Matrix: {chat_my_team} vs {chat_opp_team} is not currently running a live professional fixture."
                        st.session_state.ai_synced = True
                except Exception as e:
                    st.error(f"Network Pipeline Blocked: {e}")

    if st.session_state.get("ai_synced", False):
        st.markdown(f"<div class='live-scoreboard'><b>📡 LIVE FEED STATUS:</b> {st.session_state.live_match_context}</div>", unsafe_allow_html=True)
        
        # Chat Framework Engine
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if prompt := st.chat_input("Ask tactical questions..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            gemini_key = st.secrets.get("GEMINI_API_KEY")
            if gemini_key:
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-pro")
                
                sys_prompt = f"Context: {st.session_state.live_match_context}. As an expert football coach, answer: {prompt}"
                try:
                    response = model.generate_content(sys_prompt)
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Gemini API failure: {e}")
            else:
                st.error("Missing `GEMINI_API_KEY` token infrastructure.")
else:
    st.info("Module visualization currently offline. Under construction or waiting initialization parameters.")
