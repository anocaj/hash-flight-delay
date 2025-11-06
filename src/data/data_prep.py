"""Data preparation module for flight delay prediction.

This module handles loading and preprocessing of flight data, including:
- Loading flight and airport data
- Cleaning and standardizing data
- Merging weather data
- Feature engineering foundations
"""

import os
import pandas as pd
import numpy as np
from meteostat import Point, Hourly
from tqdm import tqdm
from pathlib import Path
import pytz
from timezonefinder import TimezoneFinder

def get_airport_timezone(lat: float, lon: float) -> str:
    """Get timezone string for airport coordinates."""
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    return tz_name if tz_name else 'UTC'

def build_airport_timezone_map(airport_coords: dict, cache_path: Path = None) -> dict:
    """Build a mapping of airports to their timezones with caching."""

    # Try to load from cache first
    if cache_path and cache_path.exists():
        print("✅ Loading timezone map from cache...")
        import json
        with open(cache_path, 'r') as f:
            return json.load(f)

    print("🕐 Building airport timezone map (this will be cached for future runs)...")

    timezone_map = {}
    for airport, coords in tqdm(airport_coords.items(), desc="Loading timezones"):
        try:
            tz_name = get_airport_timezone(coords['lat'], coords['lon'])
            timezone_map[airport] = tz_name if tz_name else 'UTC'
        except Exception as e:
            print(f"⚠️ Timezone lookup failed for {airport}: {e}")
            timezone_map[airport] = 'UTC'

    # Cache the timezone map
    if cache_path:
        import json
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(timezone_map, f, indent=2)
        print(f"✅ Timezone map cached to {cache_path}")

    return timezone_map

def fetch_weather_for_airport_with_timezone(airport: str, coords: dict, timezone_map: dict, 
                                          start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Fetch weather data for a single airport, converting to local time during fetch."""
    try:
        point = Point(coords['lat'], coords['lon'])
        
        # Fetch weather data in UTC
        weather_data = Hourly(point, start_date, end_date).fetch()
        
        if weather_data is None or weather_data.empty:
            return pd.DataFrame()
        
        # Convert to local timezone immediately
        tz_name = timezone_map.get(airport, 'UTC')
        if tz_name != 'UTC':
            airport_tz = pytz.timezone(tz_name)
            # Convert UTC index to local time
            local_index = weather_data.index.tz_localize('UTC').tz_convert(airport_tz)
            weather_data.index = local_index.tz_localize(None)  # Remove timezone info for consistency
        
        # Prepare the dataframe
        weather_df = weather_data[['temp', 'wspd', 'prcp', 'pres']].reset_index()
        weather_df['airport'] = airport
        weather_df['weather_date'] = weather_df['time'].dt.date
        weather_df['weather_hour'] = weather_df['time'].dt.hour
        
        return weather_df
        
    except Exception as e:
        print(f"⚠️ Weather fetch failed for {airport}: {e}")
        return pd.DataFrame()

def load_flight_data(jan_path: str, feb_path: str) -> pd.DataFrame:
    """Load and combine January and February flight data."""
    print("📁 Loading flight data...")
    df_jan = pd.read_csv(jan_path)
    df_feb = pd.read_csv(feb_path)
    df = pd.concat([df_jan, df_feb], ignore_index=True)
    return df

def load_airport_data(airports_path: str) -> pd.DataFrame:
    """Load and preprocess airport data."""
    airports = pd.read_csv(airports_path, dtype=str, keep_default_na=True)
    airports['latitude_deg'] = pd.to_numeric(airports['latitude_deg'], errors='coerce')
    airports['longitude_deg'] = pd.to_numeric(airports['longitude_deg'], errors='coerce')

    # Filter for US airports with valid IATA codes
    us_airports = airports[
        (airports['iso_country'] == 'US') &
        (airports['iata_code'].notna()) &
        (airports['iata_code'].str.strip() != '')
    ]
    return us_airports

def clean_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize flight data."""
    # Remove rows with missing target
    df = df.dropna(subset=["ARR_DEL15"])
    
    # Convert date and standardize airport codes
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    df['ORIGIN'] = df['ORIGIN'].str.strip().str.upper()
    df['DEST'] = df['DEST'].str.strip().str.upper()
    
    return df

def merge_airport_info(df: pd.DataFrame, airports_df: pd.DataFrame) -> pd.DataFrame:
    """Merge airport metadata (coordinates, type) with flight data."""
    print("🗺️ Adding airport metadata...")
    # Prepare airport info
    airport_info = airports_df[['iata_code', 'latitude_deg', 'longitude_deg', 'type']]
    
    # Merge origin airport info
    origin_info = airport_info.rename(columns={
        'iata_code': 'ORIGIN',
        'latitude_deg': 'ORIGIN_LAT',
        'longitude_deg': 'ORIGIN_LON',
        'type': 'ORIGIN_TYPE'
    })
    df = df.drop(columns=['ORIGIN_LAT', 'ORIGIN_LON', 'ORIGIN_TYPE'], errors='ignore')
    df = df.merge(origin_info, on='ORIGIN', how='left')
    
    # Merge destination airport info
    dest_info = airport_info.rename(columns={
        'iata_code': 'DEST',
        'latitude_deg': 'DEST_LAT',
        'longitude_deg': 'DEST_LON',
        'type': 'DEST_TYPE'
    })
    df = df.drop(columns=['DEST_LAT', 'DEST_LON', 'DEST_TYPE'], errors='ignore')
    df = df.merge(dest_info, on='DEST', how='left')
    
    return df

def fetch_weather_data(df: pd.DataFrame, cache_path: str) -> pd.DataFrame:
    """Fetch or load cached weather data for airports with timezone correction applied during fetch."""
    print("🌦️ Processing weather data with timezone correction...")
    
    # Prepare date range and hours
    df['DEP_HOUR'] = (df['CRS_DEP_TIME'] // 100).astype(int)
    df['ARR_HOUR'] = (df['CRS_ARR_TIME'] // 100).astype(int)
    start = df['FL_DATE'].min()
    end = df['FL_DATE'].max()
    
    # Get unique airport coordinates
    airport_coords_df = pd.concat([
        df[['ORIGIN', 'ORIGIN_LAT', 'ORIGIN_LON']],
        df[['DEST', 'DEST_LAT', 'DEST_LON']].rename(
            columns={'DEST':'ORIGIN', 'DEST_LAT':'ORIGIN_LAT', 'DEST_LON':'ORIGIN_LON'}
        )
    ], ignore_index=True).drop_duplicates(subset=['ORIGIN']).dropna(subset=['ORIGIN_LAT','ORIGIN_LON'])
    
    # Create airport coordinates dictionary
    airport_coords = {}
    for _, row in airport_coords_df.iterrows():
        airport_coords[row['ORIGIN']] = {
            'lat': row['ORIGIN_LAT'], 
            'lon': row['ORIGIN_LON']
        }
    
    # Build timezone map BEFORE fetching weather (with caching)
    timezone_cache = Path(cache_path).parent / "airport_timezone_cache.json"
    timezone_map = build_airport_timezone_map(airport_coords, timezone_cache)
    
    if os.path.exists(cache_path):
        print("✅ Loading cached weather data...")
        weather_df = pd.read_csv(cache_path, parse_dates=['time'])
        
        # Check if this is timezone-corrected data (new format)
        if len(weather_df) > 0:
            # Simple heuristic: if we have reasonable local hours, assume it's corrected
            sample_hours = weather_df['weather_hour'].value_counts()
            if len(sample_hours) > 20:  # Should have good hour distribution if timezone-corrected
                print("✅ Weather cache appears to have timezone correction")
                return weather_df
        
        print("🕐 Regenerating weather data with proper timezone correction...")
        # Remove old cache to force regeneration
        os.remove(cache_path)
    
    print("🌦️ Fetching weather data with timezone correction (this may take a few minutes)...")
    
    # Fetch weather for each airport with timezone correction applied during fetch
    weather_data = []
    for airport, coords in tqdm(airport_coords.items(), desc="Fetching weather"):
        airport_weather = fetch_weather_for_airport_with_timezone(
            airport, coords, timezone_map, start, end
        )
        if not airport_weather.empty:
            weather_data.append(airport_weather)
    
    if not weather_data:
        print("❌ No weather data could be fetched")
        return pd.DataFrame()
    
    weather_df = pd.concat(weather_data, ignore_index=True)
    
    # Cache the timezone-corrected data
    weather_df.to_csv(cache_path, index=False)
    print(f"✅ Weather cached to {cache_path} with timezone correction applied during fetch")
    
    return weather_df

def merge_weather_data(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge origin and destination weather data with flights using timezone-corrected local time."""
    print("🔗 Merging weather data using timezone-corrected local airport time...")

    # Ensure weather_date is in the same format for merging
    df['weather_date'] = pd.to_datetime(df['FL_DATE']).dt.date
    weather_df['weather_date'] = pd.to_datetime(weather_df['weather_date']).dt.date

    print("✅ Using timezone-corrected weather hours for matching")
    
    # Merge origin weather
    origin_weather = weather_df.rename(columns={
        'airport': 'ORIGIN',
        'temp': 'ORIGIN_TEMP',
        'wspd': 'ORIGIN_WSPD',
        'prcp': 'ORIGIN_PRCP',
        'pres': 'ORIGIN_PRES'
    })
    
    df = df.merge(
        origin_weather[['ORIGIN', 'weather_date', 'weather_hour', 'ORIGIN_TEMP',
                       'ORIGIN_WSPD', 'ORIGIN_PRCP', 'ORIGIN_PRES']],
        left_on=['ORIGIN', 'weather_date', 'DEP_HOUR'],
        right_on=['ORIGIN', 'weather_date', 'weather_hour'],
        how='left'
    ).drop(columns=['weather_hour'], errors='ignore')
    
    # Merge destination weather
    dest_weather = weather_df.rename(columns={
        'airport': 'DEST',
        'temp': 'DEST_TEMP',
        'wspd': 'DEST_WSPD',
        'prcp': 'DEST_PRCP',
        'pres': 'DEST_PRES'
    })
    
    df = df.merge(
        dest_weather[['DEST', 'weather_date', 'weather_hour', 'DEST_TEMP',
                     'DEST_WSPD', 'DEST_PRCP', 'DEST_PRES']],
        left_on=['DEST', 'weather_date', 'ARR_HOUR'],
        right_on=['DEST', 'weather_date', 'weather_hour'],
        how='left'
    ).drop(columns=['weather_hour', 'weather_date'], errors='ignore')
    
    return df

def process_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Process weather-related features."""
    # Identify weather columns
    weather_cols = ['ORIGIN_TEMP', 'ORIGIN_WSPD', 'ORIGIN_PRCP', 'ORIGIN_PRES',
                   'DEST_TEMP', 'DEST_WSPD', 'DEST_PRCP', 'DEST_PRES']
    
    # Add missing weather indicator
    df['WEATHER_MISSING'] = df[weather_cols].isna().any(axis=1).astype(int)
    
    # Fill missing values
    for col in weather_cols:
        df[col] = df[col].fillna(0)
    
    # Create weather risk indicators
    df['WEATHER_RISK_ORIGIN'] = (
        (df['ORIGIN_PRCP'] > 0.5).astype(int) +
        (df['ORIGIN_WSPD'] > 20).astype(int)
    )
    df['WEATHER_RISK_DEST'] = (
        (df['DEST_PRCP'] > 0.5).astype(int) +
        (df['DEST_WSPD'] > 20).astype(int)
    )
    
    return df

def load_and_prepare_data(base_path: str | Path, cache_weather: bool = True) -> pd.DataFrame:
    """Main function to load and prepare the complete dataset."""
    base_path = Path(base_path)
    
    # Load flight data
    df = load_flight_data(
        base_path / "flights_jan_2025.csv",
        base_path / "flights_feb_2025.csv"
    )
    
    # Load and merge airport data
    airports = load_airport_data(base_path / "airports.csv")
    df = clean_flight_data(df)
    df = merge_airport_info(df, airports)
    
    # Add weather data
    if cache_weather:
        weather_cache = base_path / "airport_weather_cache.csv"
        weather_df = fetch_weather_data(df, weather_cache)
        df = merge_weather_data(df, weather_df)
        df = process_weather_features(df)
    
    print("✅ Data preparation complete!")
    return df

def save_prepared_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save the prepared dataset to CSV."""
    df.to_csv(output_path, index=False)
    print(f"✅ Prepared dataset saved to: {output_path}")