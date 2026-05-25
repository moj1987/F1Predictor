import joblib
import fastf1
import streamlit as st
import pandas as pd
from data_pipeline import get_session_laps, clean_laps, extract_long_runs, calculate_avg_pace

st.set_page_config(page_title="F1 Race Pace Predictor", layout="wide")

st.title("🏎️ F1 Race Pace Predictor")
st.markdown("Analyze Free Practice 2 long-runs to discover the true race pace of every team.")

# Sidebar for user inputs
st.sidebar.header("Select Race Weekend")

year = st.sidebar.selectbox("Year", [2026, 2025, 2024])

# Dynamically fetch the F1 schedule for the selected year
schedule = fastf1.get_event_schedule(year)

# We don't want Pre-Season Testing in our dropdown, just actual races!
schedule = schedule[schedule['EventFormat'] != 'testing']

# Get a clean list of all Event Names
event_names = schedule['EventName'].tolist()

# Create a beautiful dropdown menu!
event = st.sidebar.selectbox("Select Grand Prix", event_names)


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
                
                            # --- THE DUMB MODEL PREDICTION ---
            st.markdown("---")
            st.subheader("🔮 Predicted Race Order (The Dumb Model)")
            
            with st.spinner("Fetching historical team data to make prediction..."):
                from data_pipeline import get_race_results
                # Fetch last year's results to know how good the car is
                historical_data = get_race_results(year - 1, event)
                
                if historical_data is not None:
                    # Filter pace_df to only use the fastest tire compound they ran
                    # (To simplify, we take their fastest overall average pace)
                    fastest_pace = pace_df.groupby('Driver').first().reset_index()
                    fastest_pace['Pace_Rank'] = fastest_pace['FP2_Avg_Pace_s'].rank()
                    
                    # Prepare the historical team feature
                    team_history = historical_data.groupby('TeamName')['Race_Position'].mean().reset_index()
                    team_history.rename(columns={'Race_Position': 'Team_Hist_Pos', 'TeamName': 'Team'}, inplace=True)
                    
                    # Combine our features
                    prediction_df = pd.merge(fastest_pace, team_history, on='Team', how='left')
                    prediction_df['Team_Hist_Pos'] = prediction_df['Team_Hist_Pos'].fillna(15.0)
                    
                    # Load our saved model!
                    model = joblib.load('dumb_model.pkl')
                    
                    # Ask it to predict!
                    prediction_df['Predicted_Finish'] = model.predict(prediction_df[['Pace_Rank', 'Team_Hist_Pos']])
                    
                    # Sort by the predicted finish
                    prediction_df = prediction_df.sort_values('Predicted_Finish').reset_index(drop=True)
                    
                    # Display the final prediction
                    st.dataframe(prediction_df[['Driver', 'Team', 'Predicted_Finish']])


        else:
            st.error("Failed to load data. The session might have been rained out or cancelled!")
