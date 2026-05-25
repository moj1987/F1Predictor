import streamlit as st
import pandas as pd
from data_pipeline import get_session_laps, clean_laps, extract_long_runs, calculate_avg_pace

st.set_page_config(page_title="F1 Race Pace Predictor", layout="wide")

st.title("🏎️ F1 Race Pace Predictor")
st.markdown("Analyze Free Practice 2 long-runs to discover the true race pace of every team.")

# Sidebar for user inputs
st.sidebar.header("Select Race Weekend")
year = st.sidebar.selectbox("Year", [2026, 2025, 2024])
event = st.sidebar.text_input("Event Name (e.g., Japan, Australia)", "Australia")

if st.sidebar.button("Analyze FP2 Pace"):
    with st.spinner(f"Fetching {year} {event} FP2 Data... (FastF1 is downloading telemetry)"):
        # 1. Fetch
        laps = get_session_laps(year, event, 'FP2')
        
        if laps is not None:
            # 2. Clean & Extract
            clean_df = clean_laps(laps)
            long_runs = extract_long_runs(clean_df)
            
            # 3. Calculate Pace
            pace_df = calculate_avg_pace(long_runs)
            
            st.success(f"Successfully analyzed {len(long_runs)} long-run laps!")
            
                        # 4. Display beautifully!
            st.subheader("🏁 FP2 Average Race Pace by Compound")
            
            # Find which compounds were actually used in long runs this session
            compounds = pace_df['Compound'].unique()
            
            # Create a separate table and chart for each compound!
            for compound in compounds:
                st.markdown(f"### 🛞 {compound} Tires")
                
                # Filter the dataframe to only show this specific compound
                compound_df = pace_df[pace_df['Compound'] == compound]
                
                display_df = compound_df[['Driver', 'Team', 'Laps_Count', 'FP2_Avg_Pace_Formatted']]
                st.dataframe(display_df, use_container_width=True)
                
                # A separate bar chart for each compound
                # st.bar_chart(data=compound_df, x='Driver', y='FP2_Avg_Pace_s')

        else:
            st.error("Failed to load data. The session might have been rained out or cancelled!")
