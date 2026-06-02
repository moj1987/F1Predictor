def test_ml_pipeline():
    print("\n\n--- 3. TESTING MACHINE LEARNING MODEL ---")
    
    # Piece A: FP2 Pace for Japan 2025
    print("Fetching 2025 FP2 data...")
    fp2_2025 = get_session_laps(2025, 'Japan', 'FP2')
    clean_fp2_2025 = clean_laps(fp2_2025)
    long_runs_2025 = extract_long_runs(clean_fp2_2025)
    pace_2025 = calculate_avg_pace(long_runs_2025)
    
    # Piece B: Actual Race Results for Japan 2025 (Our Answer Key)
    print("Fetching 2025 Race Results...")
    actual_results_2025 = get_race_results(2025, 'Japan')
    
    # Piece C: Historical Team Performance for Japan 2024 (Our Feature)
    print("Fetching 2024 Race Results (for historical team data)...")
    historical_results_2024 = get_race_results(2024, 'Japan')
    
    # Build the Training Dataset
    print("\nBuilding training dataset...")
    training_df = build_training_dataset(pace_2025, actual_results_2025, historical_results_2024)
    print("\nTraining Data Preview:")
    print(training_df.head())
    
    # Train the Model!
    print("\nTraining Random Forest...")
    model = train_rf_model(training_df)
