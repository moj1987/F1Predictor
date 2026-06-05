import os
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

try: 
    # Dynamically fetch the F1 schedule for the selected year
    schedule = fastf1.get_event_schedule(year)

    # We don't want Pre-Season Testing in our dropdown, just actual races!
    schedule = schedule[schedule['EventFormat'] != 'testing']

    # Get a clean list of all Event Names
    event_names = schedule['EventName'].tolist()

    # Create a beautiful dropdown menu!
    event = st.sidebar.selectbox("Select Grand Prix", event_names)
    # Fetch and display Track Characteristics
    from data_pipeline import get_track_downforce, get_downforce_label
    downforce_lvl = get_track_downforce(event)
    df_label = get_downforce_label(downforce_lvl)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Track Characteristic:**")
    st.sidebar.info(f"🏎️ {df_label}")

except Exception as e: 
    st.sidebar.error(f"Failed to fetch F1 schedule. Check your internet connection. ({e})")
    st.stop() # Stop execution here so the app doesn't crash further down


if st.sidebar.button("Analyze FP2 Pace"):
    with st.spinner(f"Fetching {year} {event} FP2 Data... (FastF1 is downloading telemetry)"):
        # 1. Fetch
        laps = get_session_laps(year, event, 'FP2')
        
        if laps is not None and not laps.empty:
            # 2. Clean & Extract
            clean_df = clean_laps(laps)
            long_runs = extract_long_runs(clean_df)
            
            if long_runs.empty:
                st.warning("No valid long runs detected. The session might have been wet, heavily interrupted by red flags, or no one did long stints.")
            else:
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
                    
                    # Add Degradation_Formatted so we can see it in Streamlit!
                    display_df = compound_df[['Driver', 'Team', 'Laps_Count', 'FP2_Avg_Pace_Formatted', 'Degradation_Formatted']]
                    st.dataframe(display_df, use_container_width=True)

                    
                # --- THE DUMB MODEL PREDICTION ---
                st.markdown("---")
                st.subheader("🔮 Predicted Race Order (The Dumb Model)")
                
                with st.spinner("Fetching historical team data to make prediction..."):
                    from data_pipeline import get_race_results
                    # Fetch last year's results to know how good the car is
                    historical_data = get_race_results(year - 1, event)
                    
                    if historical_data is not None and not historical_data.empty:
                        # Filter pace_df to only use the fastest tire compound they ran
                        # (To simplify, we take their fastest overall average pace)
                        fastest_pace = pace_df.groupby('Driver').first().reset_index()
                        fastest_pace['Pace_Rank'] = fastest_pace['FP2_Avg_Pace_s'].rank()
                        
                        # 1. Driver History (You already have this!)
                        driver_history = historical_data.groupby('Driver')['Race_Position'].mean().reset_index()
                        driver_history.rename(columns={'Race_Position': 'Driver_Hist_Pos'}, inplace=True)
                        prediction_df = pd.merge(fastest_pace, driver_history, on='Driver', how='left')
                        prediction_df['Driver_Hist_Pos'] = prediction_df['Driver_Hist_Pos'].fillna(15.0)

                        # 2. Team History
                        team_history = historical_data.groupby('TeamName')['Race_Position'].mean().reset_index()
                        team_history.rename(columns={'Race_Position': 'Team_Hist_Pos', 'TeamName': 'Team'}, inplace=True)
                        prediction_df = pd.merge(prediction_df, team_history, on='Team', how='left')
                        prediction_df['Team_Hist_Pos'] = prediction_df['Team_Hist_Pos'].fillna(15.0)

                        # 3. Grid Position
                        from data_pipeline import get_qualifying_results
                        qualy_results = get_qualifying_results(year, event)
                        if qualy_results is not None and not qualy_results.empty:
                            prediction_df = pd.merge(prediction_df, qualy_results, on='Driver', how='left')
                        else:
                            prediction_df['GridPosition'] = 20.0
                        prediction_df['GridPosition'] = prediction_df['GridPosition'].fillna(20.0)
                        
                        # 4. Track Characteristic!
                        prediction_df['Track_Type'] = downforce_lvl

                        # Check if the model file actually exists before loading!
                        if not os.path.exists('dumb_model.pkl'):
                            st.warning("⚠️ Model file 'dumb_model.pkl' not found! You need to train the model first before getting predictions.")
                        else:
                            # Load our saved model!
                            model = joblib.load('dumb_model.pkl')
                            
                            # Ask it to predict using our new 6th super-feature!
                            prediction_df['Predicted_Finish'] = model.predict(prediction_df[['Pace_Rank', 'Driver_Hist_Pos', 'Team_Hist_Pos', 'GridPosition', 'Track_Type', 'Tire_Deg_Rate']])

                            
                            # Convert raw scores into an exact 1-N ranking
                            prediction_df['Predicted_Finish'] = prediction_df['Predicted_Finish'].rank()
                            
                            # Sort by the predicted finish initially
                            prediction_df = prediction_df.sort_values('Predicted_Finish').reset_index(drop=True)
                            
                            # Check for actual race results
                            actual_results = get_race_results(year, event)
                            
                            if actual_results is not None and not actual_results.empty:
                                # 1 & 2: Use how='outer' to include everyone, even DNFs who missed FP2
                                prediction_df = pd.merge(prediction_df, actual_results[['Driver', 'Race_Position', 'Status']], on='Driver', how='outer')
                                
                                # Rename for clarity
                                prediction_df.rename(columns={'Race_Position': 'Actual_Finish', 'Status': 'Race_Status'}, inplace=True)
                                
                                # 4: Sort the final table based on the actual race results!
                                prediction_df = prediction_df.sort_values('Actual_Finish').reset_index(drop=True)
                                
                                # 3: Display prediction (hide_index=True removes the weird numbers on the left)
                                st.dataframe(prediction_df[['Driver', 'Team', 'Predicted_Finish', 'Actual_Finish', 'Race_Status']], hide_index=True)
                            else:
                                # Display just the final prediction
                                st.dataframe(prediction_df[['Driver', 'Team', 'Predicted_Finish']], hide_index=True)


                    else:
                        st.warning("Historical data for this event from last year is unavailable (e.g., this is a new track). The Dumb Model cannot make a prediction.")

        else:
            st.error("Failed to load FP2 data. The session might have been rained out, cancelled, or it was a Sprint weekend!")
