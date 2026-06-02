import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def build_training_dataset(fp2_pace_df, actual_race_results_df, historical_team_results_df, qualy_results_df):
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
    
    return df

def train_rf_model(training_df):
    # Use Pace_Rank instead of raw seconds!
    features = ['Pace_Rank', 'Driver_Hist_Pos', 'Team_Hist_Pos', 'GridPosition']
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
