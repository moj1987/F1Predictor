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
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                # --- THE DYNAMIC ML PREDICTION ---
                st.markdown("---")
                st.subheader("🤖 The Dynamic Random Forest Model")
                
                with st.spinner("🧠 Booting up Dynamic Training (fetching past 3 races)..."):
                    from model import build_dynamic_model
                    model, driver_form, team_form, track_affinity = build_dynamic_model(year, event)

                
                if model is not None:
                    st.success("✅ Dynamic Model Trained on the latest momentum!")
                    
                    with st.spinner("Running Predictions..."):
                        # Get the fastest pace per driver for the CURRENT race
                        prediction_df = pace_df.groupby('Driver').first().reset_index()
                        
                        # Add Pace Rank
                        prediction_df['Pace_Rank'] = prediction_df['FP2_Avg_Pace_s'].rank()
                        
                        # Add Track Type
                        prediction_df['Track_Type'] = downforce_lvl
                        
                        # Map Recent Form
                        prediction_df = pd.merge(prediction_df, driver_form, on='Driver', how='left')
                        prediction_df = pd.merge(prediction_df, team_form, on='Team', how='left')
                        prediction_df['Team_Recent_Form'] = prediction_df['Team_Recent_Form'].fillna(15.0)
                        prediction_df = pd.merge(prediction_df, track_affinity, on='Driver', how='left')
                        # Fallback to Driver_Recent_Form if they have zero history at this track!
                        prediction_df['Driver_Track_History'] = prediction_df['Driver_Track_History'].fillna(prediction_df['Driver_Recent_Form'])

                        # If a driver is a rookie, give them the car's average form!
                        prediction_df['Driver_Recent_Form'] = prediction_df['Driver_Recent_Form'].fillna(prediction_df['Team_Recent_Form']).fillna(15.0)

                        # Fetch Qualifying
                        from data_pipeline import get_qualifying_results
                        qualy_results = get_qualifying_results(year, event)
                        if qualy_results is not None and not qualy_results.empty:
                            prediction_df = pd.merge(prediction_df, qualy_results, on='Driver', how='left')
                        else:
                            prediction_df['GridPosition'] = 20.0
                        prediction_df['GridPosition'] = prediction_df['GridPosition'].fillna(20.0)


                        # Predict using the LIVE dynamic model!
                        prediction_df['Predicted_Finish'] = model.predict(prediction_df[['Pace_Rank', 'Driver_Recent_Form', 'Team_Recent_Form', 'Driver_Track_History', 'GridPosition', 'Track_Type', 'Tire_Deg_Rate']])

                        
                        # Convert raw scores into an exact 1-N ranking
                        prediction_df['Predicted_Finish'] = prediction_df['Predicted_Finish'].rank(method='first')
                        
                        # Sort by the predicted finish initially
                        prediction_df = prediction_df.sort_values('Predicted_Finish').reset_index(drop=True)
                            
                        # Check for actual race results
                        from data_pipeline import get_race_results
                        actual_results = get_race_results(year, event)
                        
                        if actual_results is not None and not actual_results.empty:
                            # 1 & 2: Use how='outer' to include everyone, even DNFs who missed FP2
                            # Bring TeamName over from Sunday's results!
                            prediction_df = pd.merge(prediction_df, actual_results[['Driver', 'Race_Position', 'Status', 'TeamName']], on='Driver', how='outer')

                            # Fill any blank Teams with the TeamName from Sunday
                            prediction_df['Team'] = prediction_df['Team'].fillna(prediction_df['TeamName'])

                            
                            # Rename for clarity
                            prediction_df.rename(columns={'Race_Position': 'Actual_Finish', 'Status': 'Race_Status'}, inplace=True)
                            
                            # 4: Sort the final table based on the actual race results!
                            prediction_df = prediction_df.sort_values('Actual_Finish').reset_index(drop=True)
                            
                            # 3: Display prediction (hide_index=True removes the weird numbers on the left)
                            st.dataframe(prediction_df[['Driver', 'Team', 'Driver_Track_History', 'Predicted_Finish', 'Actual_Finish', 'Race_Status']], hide_index=True)
                        else:
                            st.dataframe(prediction_df[['Driver', 'Team', 'Driver_Track_History', 'Predicted_Finish']], hide_index=True)

                else:
                    st.warning("⚠️ Welcome to a New Era! Because this is the first race of the new regulations, there is NO historical data to train the ML model.")
                    st.info("🔮 Falling back to predicting based purely on Free Practice 2 Pace!")
                    
                    with st.spinner("Running Heuristic Predictions..."):
                        # Just grab the FP2 pace and rank them exactly as they are
                        prediction_df = pace_df.groupby('Driver').first().reset_index()
                        prediction_df['Predicted_Finish'] = prediction_df['FP2_Avg_Pace_s'].rank(method='first')
                        prediction_df = prediction_df.sort_values('Predicted_Finish').reset_index(drop=True)
                        
                        from data_pipeline import get_race_results
                        actual_results = get_race_results(year, event)
                        
                        if actual_results is not None and not actual_results.empty:
                            # Bring TeamName over from Sunday's results!
                            prediction_df = pd.merge(prediction_df, actual_results[['Driver', 'Race_Position', 'Status', 'TeamName']], on='Driver', how='outer')

                            # Fill any blank Teams with the TeamName from Sunday
                            prediction_df['Team'] = prediction_df['Team'].fillna(prediction_df['TeamName'])

                            prediction_df.rename(columns={'Race_Position': 'Actual_Finish', 'Status': 'Race_Status'}, inplace=True)
                            prediction_df = prediction_df.sort_values('Actual_Finish').reset_index(drop=True)
                            st.dataframe(prediction_df[['Driver', 'Team', 'Predicted_Finish', 'Actual_Finish', 'Race_Status']], hide_index=True)
                        else:
                            st.dataframe(prediction_df[['Driver', 'Team', 'Predicted_Finish']], hide_index=True)
            

        else:
            st.error("Failed to load FP2 data. The session might have been rained out, cancelled, or it was a Sprint weekend!")
