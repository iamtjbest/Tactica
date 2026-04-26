import streamlit as st
import json

# 1. Load the database dynamically
try:
    with open('players.json', 'r') as f:
        players_db = json.load(f)
        # Extract just the team names into a list
        available_teams = list(players_db.keys()) 
except (FileNotFoundError, json.JSONDecodeError):
    available_teams = []
    st.error("Could not load players.json. Please run db_builder.py first!")

# 2. Feed that list directly into your selectboxes
if available_teams:
    col1, col2 = st.columns(2)
    with col1:
        team_name = st.selectbox("Your Team", available_teams)
    with col2:
        opponent_name = st.selectbox("Opponent", available_teams)