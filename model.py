import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def build_training_dataset(fp2_pace_df, actual_race_results_df, historical_team_results_df, qualy_results_df, event_name):
    # Calculate historical finish position per DRIVER
    driver_history = historical_team_results_df.groupby('Driver')['Race_Position'].mean().reset_index()
    driver_history.rename(columns={'Race_Position': 'Driver_Hist_Pos'}, inplace=True)
    df = pd.merge(fp2_pace_df, driver_history, on='Driver', how='left')
    df['Driver_Hist_Pos'] = df['Driver_Hist_Pos'].fillna(15.0)

    # Team History
    team_history = historical_team_results_df.groupby('TeamName')['Race_Position'].mean().reset_index()
    team_history.rename(columns={'Race_Position': 'Team_Hist_Pos', 'TeamName': 'Team'}, inplace=True)
    df = pd.merge(df, team_history, on='Team', how='left')
    df['Team_Hist_Pos'] = df['Team_Hist_Pos'].fillna(15.0)

    # Grid position
    df = pd.merge(df, qualy_results_df, on='Driver', how='left')
    df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce').fillna(20.0)

    actual_results = actual_race_results_df[['Driver', 'Race_Position']]
    df = pd.merge(df, actual_results, on='Driver', how='inner')
    df['Race_Position'] = pd.to_numeric(df['Race_Position'], errors='coerce').fillna(20.0)
    
    # Create the track-agnostic Pace Rank!
    df['Pace_Rank'] = df['FP2_Avg_Pace_s'].rank()
    
    # Track type 
    from data_pipeline import get_track_downforce
    df['Track_Type'] = get_track_downforce(event_name)

    return df

def train_rf_model(training_df):
    features = ['Pace_Rank', 'Driver_Hist_Pos', 'Team_Hist_Pos', 'GridPosition', 'Track_Type', 'Tire_Deg_Rate']
    target = 'Race_Position'
    
    X = training_df[features]
    y = training_df[target]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    predictions = model.predict(X)
    error = mean_absolute_error(y, predictions)
    print(f"Model Trained! Mean Absolute Error: {error:.2f} positions")
    
    # NEW: Save the model to a file!
    joblib.dump(model, 'dumb_model.pkl')
    print("✅ Model successfully saved to dumb_model.pkl!")
    
    return model

def get_recent_events(target_year, target_event_name, num_races=3):
    """Finds the N races immediately preceding the target race."""
    import fastf1
    schedule = fastf1.get_event_schedule(target_year)
    target_event = schedule[schedule['EventName'] == target_event_name]
    if target_event.empty: return []
    
    target_round = target_event.iloc[0]['RoundNumber']
    past_events = []
    curr_round = target_round - 1
    curr_year = target_year
    curr_schedule = schedule
    
    while len(past_events) < num_races:
        if curr_round < 1:
            curr_year -= 1
            curr_schedule = fastf1.get_event_schedule(curr_year)
            curr_round = curr_schedule[curr_schedule['RoundNumber'] > 0]['RoundNumber'].max()
            
        event_row = curr_schedule[curr_schedule['RoundNumber'] == curr_round].iloc[0]
        if event_row['EventFormat'] != 'testing':
            past_events.append({'year': curr_year, 'event': event_row['EventName']})
        curr_round -= 1
        
    return past_events

def build_dynamic_model(target_year, target_event_name):
    from data_pipeline import get_session_laps, clean_laps, extract_long_runs, calculate_avg_pace, get_race_results, get_qualifying_results, get_track_downforce
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    
    past_events = get_recent_events(target_year, target_event_name, num_races=3)
    
    all_training_data = []
    recent_driver_results = []
    recent_team_results = []
    
    for pe in past_events:
        year, event = pe['year'], pe['event']
        
        # 1. Get Race Results
        actual = get_race_results(year, event)
        if actual is None or actual.empty: continue
            
        # NEW FIX: Treat DNFs as 20th place so they don't get an artificially good average!
        actual['Race_Position'] = pd.to_numeric(actual['Race_Position'], errors='coerce').fillna(20.0)
            
        # Store for Step 4 (Recent Form)
        recent_driver_results.append(actual[['Driver', 'Race_Position']])
        recent_team_results.append(actual[['TeamName', 'Race_Position']])

        # 2. Get Pace
        laps = get_session_laps(year, event, 'FP2')
        if laps is None or laps.empty: continue
        clean = clean_laps(laps)
        long_runs = extract_long_runs(clean)
        pace_df = calculate_avg_pace(long_runs)
        if pace_df.empty: continue
        pace_df = pace_df.groupby('Driver').first().reset_index()
        
        # 3. Get Qualy
        qualy = get_qualifying_results(year, event)
        
        # Merge it all for this single past race!
        df = pace_df.copy()
        if qualy is not None and not qualy.empty:
            df = pd.merge(df, qualy, on='Driver', how='left')
        df['GridPosition'] = pd.to_numeric(df.get('GridPosition', 20), errors='coerce').fillna(20.0)
        
        df = pd.merge(df, actual[['Driver', 'Race_Position']], on='Driver', how='inner')
        df['Race_Position'] = pd.to_numeric(df['Race_Position'], errors='coerce').fillna(20.0)
        
        df['Track_Type'] = get_track_downforce(event)
        df['Pace_Rank'] = df['FP2_Avg_Pace_s'].rank()
        
        all_training_data.append(df)
        
    if not all_training_data:
        return None, None, None
        
    # Combine all training data
    training_df = pd.concat(all_training_data, ignore_index=True)
    
    # Calculate Recent Form! (Step 4)
    all_driver_res = pd.concat(recent_driver_results, ignore_index=True)
    driver_form = all_driver_res.groupby('Driver')['Race_Position'].mean().reset_index()
    driver_form.rename(columns={'Race_Position': 'Driver_Recent_Form'}, inplace=True)
    
    all_team_res = pd.concat(recent_team_results, ignore_index=True)
    team_form = all_team_res.groupby('TeamName')['Race_Position'].mean().reset_index()
    team_form.rename(columns={'Race_Position': 'Team_Recent_Form', 'TeamName': 'Team'}, inplace=True)
    
    # Map Recent Form to Training Data
    training_df = pd.merge(training_df, driver_form, on='Driver', how='left')
    training_df['Driver_Recent_Form'] = training_df['Driver_Recent_Form'].fillna(15.0)
    
    training_df = pd.merge(training_df, team_form, on='Team', how='left')
    training_df['Team_Recent_Form'] = training_df['Team_Recent_Form'].fillna(15.0)
    
    # Train the model live!
    features = ['Pace_Rank', 'Driver_Recent_Form', 'Team_Recent_Form', 'GridPosition', 'Track_Type', 'Tire_Deg_Rate']
    X = training_df[features]
    y = training_df['Race_Position']
    
    model = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=5, random_state=42)
    model.fit(X, y)
    
    return model, driver_form, team_form
