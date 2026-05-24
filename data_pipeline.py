import fastf1
import pandas as pd
import os

# Enable cache to speed up subsequent runs
CACHE_DIR = 'f1_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def get_session_laps(year, event, session_type):
    """
    Fetches the laps for a specific session (e.g., 'R' for Race, 'FP2' for Practice 2).
    """
    print(f"Loading data for {year} {event} - {session_type}...")
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load(telemetry=False, weather=False) # We just need lap times and tire data
        return session.laps
    except Exception as e:
        print(f"Failed to load session: {e}")
        return None

def clean_laps(laps_df):
    """
    Cleans lap data by removing in/out laps, anomalous times, and safety car periods.
    """
    # fastf1 has an 'IsAccurate' flag that filters out in/out laps and VSC/SC laps automatically!
    clean_df = laps_df[laps_df['IsAccurate'] == True].copy()
    
    # Drop rows missing crucial data
    clean_df = clean_df.dropna(subset=['LapTime', 'Compound', 'TyreLife'])
    
    # Convert LapTime (timedelta) to numeric seconds for easier ML modeling
    clean_df['LapTime_s'] = clean_df['LapTime'].dt.total_seconds()
    
    return clean_df

def extract_long_runs(laps_df, min_laps=4):
    """
    Extracts 'long runs' from practice sessions.
    A long run is defined as a consecutive stint on the same tire compound 
    lasting for at least `min_laps` laps.
    """
    # Group by driver and stint to find long runs
    stint_lengths = laps_df.groupby(['Driver', 'Stint']).size()
    
    # Keep only stints that are longer than min_laps
    long_run_stints = stint_lengths[stint_lengths >= min_laps].reset_index()
    
    # Merge back to get the actual lap data for those long runs
    long_runs_df = pd.merge(
        laps_df, 
        long_run_stints[['Driver', 'Stint']], 
        on=['Driver', 'Stint'], 
        how='inner'
    )
    
    return long_runs_df
