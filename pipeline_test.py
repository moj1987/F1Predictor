import fastf1
from data_pipeline import get_session_laps, clean_laps, extract_long_runs, calculate_avg_pace, get_race_results, get_qualifying_results
from model import build_training_dataset, train_rf_model
def explore_fastf1_api():
    print("--- 1. EXPLORING FASTF1 API ---")
    # Load Data
    session = fastf1.get_session(2026, 'Spain', 'FP2')
    session.load(telemetry=False, weather=True) 

    print(f"\nEvent Name: {session.event['EventName']}")
    print(f"Session Name: {session.name}")
    print(f"Weather Data columns: {session.weather_data.columns.tolist()}")
    
    # Get the fastest lap of the session
    fastest_lap = session.laps.pick_fastest()
    print(f"\nFastest Lap was by: {fastest_lap['Driver']}")
    print(f"Time: {fastest_lap['LapTime']}")
    print(f"Compound used: {fastest_lap['Compound']}")
    print(f"Top Speed trap (SpeedST): {fastest_lap['SpeedST']} km/h")
    
    # What columns are available in a lap?
    print("\nLap Data contains these columns (features we can use!):")
    print(session.laps.columns.tolist())

def test_pipeline():
    print("\n\n--- 2. TESTING OUR DATA PIPELINE ---")
    
    # 1. Fetch
    print("Fetching Japan 2026 FP2 using our pipeline...")
    laps = get_session_laps(2026, 'Japan', 'FP2')
    print(f"Raw laps fetched: {len(laps)}")

    # 2. Clean
    clean = clean_laps(laps)
    print(f"Cleaned laps: {len(clean)} (Removed out-laps, in-laps, safety cars)")

    # 3. Extract Long Runs
    long_runs = extract_long_runs(clean, min_laps=4)
    print(f"Extracted {len(long_runs)} long run laps!")
    
    # 4. EXPORT DATA FOR INSPECTION ---
    csv_filename = "japan_2026_fp2_long_runs.csv"
    long_runs.to_csv(csv_filename, index=False)
    print(f"\n✅ Full dataset saved to {csv_filename}! Open it to inspect every row.")

    # Let's see how many long-run laps each driver did
    print("\nLong run lap count per driver:")
    # Change this line in test_pipeline.py:
    print(long_runs[['Driver', 'Compound', 'TyreLife', 'LapTime_Formatted', 'LapNumber']].head(10))
    
    print("\nPreview of the final data we will send to the ML model:")
    print(long_runs[['Driver', 'Compound', 'TyreLife', 'LapTime_s', 'LapNumber']].head(10))

    # --- 5. CALCULATE AVERAGE PACE ---
    print("\n--- FINAL FP2 PACE RANKINGS ---")
    avg_pace = calculate_avg_pace(long_runs)
    print(avg_pace[['Driver', 'FP2_Avg_Pace_Formatted']].head(10))

def test_ml_pipeline():
    print("\n\n--- 3. TESTING MACHINE LEARNING MODEL ---")
    
    # Piece A: FP2 Pace 
    print("Fetching 2025 FP2 data...")
    fp2_2025 = get_session_laps(2025, 'Spain', 'FP2')
    clean_fp2_2025 = clean_laps(fp2_2025)
    long_runs_2025 = extract_long_runs(clean_fp2_2025)
    pace_2025 = calculate_avg_pace(long_runs_2025)
    
    # Piece B: Actual Race Results (Our Answer Key)
    print("Fetching 2025 Race Results...")
    actual_results_2025 = get_race_results(2025, 'Spain')
    
    # Piece C: Historical Team Performance for 2024 (Our Feature)
    print("Fetching 2024 Race Results (for historical team data)...")
    historical_results_2024 = get_race_results(2024, 'Spain')
    
    print("Fetching 2025 Qualifying Results...")
    qualy_2025 = get_qualifying_results(2025, 'Spain')
    
    print("\nBuilding training dataset...")
    # Pass quali
    training_df = build_training_dataset(pace_2025, actual_results_2025, historical_results_2024, qualy_2025)

    print("\nTraining Data Preview:")
    print(training_df.head())
    
    # Train the Model!
    print("\nTraining Random Forest...")
    model = train_rf_model(training_df)

if __name__ == "__main__":
    # explore_fastf1_api()
    # test_pipeline()
    test_ml_pipeline()
