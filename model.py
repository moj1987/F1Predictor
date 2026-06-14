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

def get_recent_events(target_year, target_event_name, num_races=15):
    """Finds the N races immediately preceding the target race with an Era Boundary."""
    import fastf1
    from data_pipeline import get_track_downforce
    
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
            # ERA BOUNDARY: Don't pull 2025 (or older) data for a 2026 race!
            if target_year >= 2026 and curr_year < 2026:
                break
                
            curr_schedule = fastf1.get_event_schedule(curr_year)
            curr_round = curr_schedule[curr_schedule['RoundNumber'] > 0]['RoundNumber'].max()
            
        event_row = curr_schedule[curr_schedule['RoundNumber'] == curr_round].iloc[0]
        if event_row['EventFormat'] != 'testing':
            event_name = event_row['EventName']
            past_events.append({
                'year': curr_year, 
                'event': event_name,
                'track_type': get_track_downforce(event_name)
            })
        curr_round -= 1
        
    return past_events

def get_track_history(target_year, target_event_name, num_years=3):
    """Calculates a driver's average finish at this exact track over the last N years."""
    import fastf1
    from data_pipeline import get_race_results
    import pandas as pd
    
    historical_results = []
    
    try:
        # 1. Get the target event's Country/Location to avoid Sponsor Name Changes!
        target_schedule = fastf1.get_event_schedule(target_year)
        target_event = target_schedule[target_schedule['EventName'] == target_event_name]
        if target_event.empty: return pd.DataFrame(columns=['Driver', 'Driver_Track_History'])
        
        target_country = target_event.iloc[0]['Country']
        target_location = target_event.iloc[0]['Location']
    except Exception:
        return pd.DataFrame(columns=['Driver', 'Driver_Track_History'])
    
    for y in range(target_year - 1, target_year - 1 - num_years, -1):
        try:
            schedule = fastf1.get_event_schedule(y)
            # Match safely by Country and Location!
            event = schedule[(schedule['Country'] == target_country) & (schedule['Location'] == target_location)]
            if not event.empty:
                historical_event_name = event.iloc[0]['EventName']
                actual = get_race_results(y, historical_event_name)
                if actual is not None and not actual.empty:
                    actual['Race_Position'] = pd.to_numeric(actual['Race_Position'], errors='coerce').fillna(20.0)
                    historical_results.append(actual[['Driver', 'Race_Position']])
        except Exception:
            pass 
            
    if historical_results:
        all_hist = pd.concat(historical_results, ignore_index=True)
        track_affinity = all_hist.groupby('Driver')['Race_Position'].mean().reset_index()
        track_affinity.rename(columns={'Race_Position': 'Driver_Track_History'}, inplace=True)
        return track_affinity
    else:
        return pd.DataFrame(columns=['Driver', 'Driver_Track_History'])

def build_dynamic_model(target_year, target_event_name):
    from data_pipeline import get_session_laps, clean_laps, extract_long_runs, calculate_avg_pace, get_race_results, get_qualifying_results, get_track_downforce
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    
    # 1. Fetch up to 15 races for the Brain
    past_events = get_recent_events(target_year, target_event_name, num_races=15)
    target_track_type = get_track_downforce(target_event_name)
    
    # Inject historical data for THIS EXACT TRACK!
    import fastf1
    try:
        target_schedule = fastf1.get_event_schedule(target_year)
        target_event = target_schedule[target_schedule['EventName'] == target_event_name]
        if not target_event.empty:
            target_country = target_event.iloc[0]['Country']
            target_location = target_event.iloc[0]['Location']
            
            for y in range(target_year - 1, target_year - 4, -1):
                try:
                    schedule = fastf1.get_event_schedule(y)
                    # Match safely by Country and Location!
                    event_row = schedule[(schedule['Country'] == target_country) & (schedule['Location'] == target_location)]
                    if not event_row.empty:
                        historical_event_name = event_row.iloc[0]['EventName']
                        if not any(pe['year'] == y and pe['event'] == historical_event_name for pe in past_events):
                            past_events.append({
                                'year': y,
                                'event': historical_event_name,
                                'track_type': target_track_type
                            })
                except Exception:
                    pass
    except Exception:
        pass

    if not past_events:
        return None, None, None

    # --- 2. Track-Matched Momentum (The 7-Race Decay Window) ---
    recent_7_events = past_events[:7]
    matched_events = [pe for pe in recent_7_events if pe['track_type'] == target_track_type]
    
    # Take up to 3 exact Track Type matches
    momentum_events = matched_events[:3]
    
    # If we found fewer than 3, pad with the most chronologically recent races from the 7-race window
    if len(momentum_events) < 3:
        for pe in recent_7_events:
            if pe not in momentum_events:
                momentum_events.append(pe)
            if len(momentum_events) == 3:
                break

    # Calculate Recent Form from our perfectly curated Momentum Events!
    recent_driver_results = []
    recent_team_results = []
    
    for pe in momentum_events:
        actual = get_race_results(pe['year'], pe['event'])
        if actual is None or actual.empty: continue
        
        # Treat DNFs as 20th place!
        actual['Race_Position'] = pd.to_numeric(actual['Race_Position'], errors='coerce').fillna(20.0)
        recent_driver_results.append(actual[['Driver', 'Race_Position']])
        recent_team_results.append(actual[['TeamName', 'Race_Position']])
        
    if recent_driver_results:
        all_driver_res = pd.concat(recent_driver_results, ignore_index=True)
        driver_form = all_driver_res.groupby('Driver')['Race_Position'].mean().reset_index()
        driver_form.rename(columns={'Race_Position': 'Driver_Recent_Form'}, inplace=True)
        
        all_team_res = pd.concat(recent_team_results, ignore_index=True)
        team_form = all_team_res.groupby('TeamName')['Race_Position'].mean().reset_index()
        team_form.rename(columns={'Race_Position': 'Team_Recent_Form', 'TeamName': 'Team'}, inplace=True)
    else:
        # Extreme edge case (race 1 of a new era)
        driver_form = pd.DataFrame(columns=['Driver', 'Driver_Recent_Form'])
        team_form = pd.DataFrame(columns=['Team', 'Team_Recent_Form'])

    # --- 3. Build Training Dataset (The 15-Race Brain) ---
    all_training_data = []
    
    for pe in past_events:
        year, event = pe['year'], pe['event']
        
        actual = get_race_results(year, event)
        if actual is None or actual.empty: continue
            
        laps = get_session_laps(year, event, 'FP2')
        if laps is None or laps.empty: continue
        clean = clean_laps(laps)
        long_runs = extract_long_runs(clean)
        pace_df = calculate_avg_pace(long_runs)
        if pace_df.empty: continue
        pace_df = pace_df.groupby('Driver').first().reset_index()
        
        qualy = get_qualifying_results(year, event)
        
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
        return None, None, None, None
        
    training_df = pd.concat(all_training_data, ignore_index=True)
    
    # Calculate Track Affinity (3-year history at this exact track)
    track_affinity = get_track_history(target_year, target_event_name)
    
    # Map Recent Form to Training Data
    training_df = pd.merge(training_df, driver_form, on='Driver', how='left')
    training_df = pd.merge(training_df, team_form, on='Team', how='left')
    
    # Rookies inherit team form, otherwise 15.0
    training_df['Driver_Recent_Form'] = training_df['Driver_Recent_Form'].fillna(training_df['Team_Recent_Form']).fillna(15.0)
    training_df['Team_Recent_Form'] = training_df['Team_Recent_Form'].fillna(15.0)
    
    # Map Track Affinity to Training Data
    training_df = pd.merge(training_df, track_affinity, on='Driver', how='left')
    # If no track history, fallback to their overall recent form!
    training_df['Driver_Track_History'] = training_df['Driver_Track_History'].fillna(training_df['Driver_Recent_Form'])
    
    # --- 4. Train the model live with strict rules to prevent overfitting! ---
    features = ['Pace_Rank', 'Driver_Recent_Form', 'Team_Recent_Form', 'Driver_Track_History', 'GridPosition', 'Track_Type', 'Tire_Deg_Rate']
    X = training_df[features]
    
    # Map Race Position to F1 Points so the model focuses entirely on the Top 10!
    def map_points(pos):
        points = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
        return points.get(pos, 0.0)
        
    training_df['Points'] = training_df['Race_Position'].apply(map_points)
    y = training_df['Points']

    
    # max_depth=4 stops the trees from overthinking. 
    # min_samples_leaf=5 forces it to base predictions on a consensus of at least 5 similar drivers!
    model = RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_leaf=2, random_state=42)
    model.fit(X, y)
    
    # Return track_affinity so the Streamlit App can display it!
    return model, driver_form, team_form, track_affinity

