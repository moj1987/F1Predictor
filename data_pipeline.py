import fastf1
import pandas as pd
import os

CACHE_DIR = 'f1_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def get_session_laps(year, event, session_type):
    print(f"Loading data for {year} {event} - {session_type}...")
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load(telemetry=False, weather=False)
        return session.laps
    except Exception as e:
        print(f"Failed to load session: {e}")
        return None

def clean_laps(laps_df):
    clean_df = laps_df[laps_df['IsAccurate'] == True].copy()
    clean_df = clean_df.dropna(subset=['LapTime', 'Compound', 'TyreLife'])
    clean_df['LapTime_s'] = clean_df['LapTime'].dt.total_seconds()
    clean_df['LapTime_Formatted'] = clean_df['LapTime_s'].apply(format_time)
    return clean_df

def extract_long_runs(laps_df, min_laps=4):
    # Group by driver and stint
    stints = laps_df.groupby(['Driver', 'Stint'])
    
    # Calculate the fastest lap for each stint
    fastest_per_stint = stints['LapTime_s'].transform('min')
    
    # The 107% Rule: Keep only laps within 107% of the stint's best lap
    # This throws out cool-down and recharge laps!
    filtered_df = laps_df[laps_df['LapTime_s'] <= (fastest_per_stint * 1.07)].copy()
    
    # Now, recalculate stint lengths AFTER throwing out the junk laps
    stint_lengths = filtered_df.groupby(['Driver', 'Stint']).size()
    long_run_stints = stint_lengths[stint_lengths >= min_laps].reset_index()
    
    # Merge back to get the final, clean long runs
    long_runs_df = pd.merge(
        filtered_df, 
        long_run_stints[['Driver', 'Stint']], 
        on=['Driver', 'Stint'], 
        how='inner'
    )
    
    return long_runs_df

# Helper function to format seconds into "M:SS:mmm"
def format_time(seconds):
    mins = int(seconds // 60)
    
    # 06.3f gives us a string like "36.078"
    secs_str = f"{seconds % 60:06.3f}"
    
    # Replace the decimal point with a colon to make it "36:078"
    secs_str = secs_str.replace('.', ':')
    
    return f"{mins}:{secs_str}"

def calculate_avg_pace(long_runs_df):
    """
    Calculates the average FP2 long-run pace AND lap count for each driver.
    """
    # Use .agg() to calculate both the mean (average time) AND the count (number of laps)
    avg_pace = long_runs_df.groupby(['Driver', 'Team', 'Compound'])['LapTime_s'].agg(['mean', 'count']).reset_index()
    
    # Rename the new columns so they are easy to read
    avg_pace.rename(columns={'mean': 'FP2_Avg_Pace_s', 'count': 'Laps_Count'}, inplace=True)
    
    # Sort from fastest (lowest time) to slowest
    avg_pace = avg_pace.sort_values(by='FP2_Avg_Pace_s').reset_index(drop=True)
    
    # Format the time for display
    avg_pace['FP2_Avg_Pace_Formatted'] = avg_pace['FP2_Avg_Pace_s'].apply(format_time)
    
    return avg_pace

def get_race_results(year, event):
    """
    Fetches the final classification (finishing order) of a past race.
    """
    print(f"Fetching race results for {year} {event}...")
    try:
        session = fastf1.get_session(year, event, 'R')
        session.load(telemetry=False, weather=False)
        
        # session.results contains a DataFrame with the final finishing order
        results = session.results[['Abbreviation', 'Position', 'TeamName', 'Status']]
        
        # 'Abbreviation' is the 3-letter driver code (e.g., VER)
        results = results.rename(columns={'Abbreviation': 'Driver', 'Position': 'Race_Position'})
        return results
    except Exception as e:
        print(f"Failed to load race results: {e}")
        return None
