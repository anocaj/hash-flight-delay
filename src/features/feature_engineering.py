"""Feature engineering module for flight delay prediction.

This module contains functions for creating various types of features from flight data:
- Time-based features (hour, day, week, month patterns)
- Route-based features (origin-destination pairs, distance categories)
- Aircraft-specific features (tail number patterns, carrier features)
- Advanced temporal features (rolling statistics, delay propagation)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from flight date and scheduled times.
    
    Args:
        df: DataFrame with FL_DATE, CRS_DEP_TIME, CRS_ARR_TIME columns
        
    Returns:
        DataFrame with additional time-based features
    """
    df = df.copy()
    
    # Ensure FL_DATE is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['FL_DATE']):
        df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    
    # Extract basic time components
    df['MONTH'] = df['FL_DATE'].dt.month
    df['DAY_OF_MONTH'] = df['FL_DATE'].dt.day
    df['WEEKDAY'] = df['FL_DATE'].dt.day_name()
    df['IS_WEEKEND'] = df['FL_DATE'].dt.weekday.isin([5, 6]).astype(int)
    
    # Create departure and arrival hour features if not already present
    if 'DEP_HOUR' not in df.columns:
        df['DEP_HOUR'] = (df['CRS_DEP_TIME'] // 100).astype(int)
    if 'ARR_HOUR' not in df.columns:
        df['ARR_HOUR'] = (df['CRS_ARR_TIME'] // 100).astype(int)
    
    # Create time period categories
    def categorize_time_period(hour):
        """Categorize hours into meaningful periods."""
        if 5 <= hour < 12:
            return 'MORNING'
        elif 12 <= hour < 17:
            return 'AFTERNOON'
        elif 17 <= hour < 21:
            return 'EVENING'
        else:
            return 'NIGHT'
    
    df['DEP_PERIOD'] = df['DEP_HOUR'].apply(categorize_time_period)
    df['ARR_PERIOD'] = df['ARR_HOUR'].apply(categorize_time_period)
    
    # Create time buckets for analysis (6-hour periods)
    df['TIME_BUCKET'] = df['DEP_HOUR'] // 6
    df['TIME_BUCKET_LABEL'] = df['TIME_BUCKET'].map({
        0: '00-06',
        1: '06-12', 
        2: '12-18',
        3: '18-24'
    })
    
    # Create rush hour indicators
    df['IS_RUSH_HOUR'] = ((df['DEP_HOUR'].between(7, 9)) | 
                          (df['DEP_HOUR'].between(17, 19))).astype(int)
    
    # Create holiday/special day indicators (basic implementation)
    # Note: This could be expanded with actual holiday data
    df['IS_HOLIDAY_SEASON'] = df['MONTH'].isin([11, 12]).astype(int)
    df['IS_SUMMER'] = df['MONTH'].isin([6, 7, 8]).astype(int)
    
    return df


def create_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create route-based features from origin and destination data.
    
    Args:
        df: DataFrame with ORIGIN, DEST, DISTANCE columns
        
    Returns:
        DataFrame with additional route-based features
    """
    df = df.copy()
    
    # Create route identifier
    if 'ROUTE' not in df.columns:
        df['ROUTE'] = df['ORIGIN'].astype(str) + '-' + df['DEST'].astype(str)
    
    # Create reverse route for bidirectional analysis
    df['REVERSE_ROUTE'] = df['DEST'].astype(str) + '-' + df['ORIGIN'].astype(str)
    
    # Distance-based categories
    if 'DISTANCE' in df.columns:
        df['DISTANCE_CATEGORY'] = pd.cut(
            df['DISTANCE'], 
            bins=[0, 500, 1000, 2000, float('inf')],
            labels=['SHORT', 'MEDIUM', 'LONG', 'VERY_LONG'],
            include_lowest=True
        )
        
        # Flight duration estimate (rough approximation)
        df['ESTIMATED_FLIGHT_TIME'] = df['DISTANCE'] / 500  # Rough estimate in hours
    
    # Hub airport indicators (major US hubs)
    major_hubs = ['ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'CLT', 'LAS', 'PHX', 'MIA', 'SEA', 'MSP', 'DTW', 'BOS', 'PHL', 'LGA', 'JFK', 'EWR', 'SFO', 'BWI', 'DCA', 'IAD', 'MDW', 'HOU', 'IAH']
    
    df['ORIGIN_IS_HUB'] = df['ORIGIN'].isin(major_hubs).astype(int)
    df['DEST_IS_HUB'] = df['DEST'].isin(major_hubs).astype(int)
    df['HUB_TO_HUB'] = (df['ORIGIN_IS_HUB'] & df['DEST_IS_HUB']).astype(int)
    
    # State-based features
    if 'ORIGIN_STATE_ABR' in df.columns and 'DEST_STATE_ABR' in df.columns:
        df['SAME_STATE'] = (df['ORIGIN_STATE_ABR'] == df['DEST_STATE_ABR']).astype(int)
        
        # Coast-to-coast flights
        west_coast_states = ['CA', 'OR', 'WA', 'NV', 'AZ']
        east_coast_states = ['NY', 'NJ', 'CT', 'MA', 'ME', 'NH', 'VT', 'RI', 'PA', 'DE', 'MD', 'VA', 'NC', 'SC', 'GA', 'FL']
        
        df['ORIGIN_WEST_COAST'] = df['ORIGIN_STATE_ABR'].isin(west_coast_states).astype(int)
        df['ORIGIN_EAST_COAST'] = df['ORIGIN_STATE_ABR'].isin(east_coast_states).astype(int)
        df['DEST_WEST_COAST'] = df['DEST_STATE_ABR'].isin(west_coast_states).astype(int)
        df['DEST_EAST_COAST'] = df['DEST_STATE_ABR'].isin(east_coast_states).astype(int)
        
        df['COAST_TO_COAST'] = ((df['ORIGIN_WEST_COAST'] & df['DEST_EAST_COAST']) |
                                (df['ORIGIN_EAST_COAST'] & df['DEST_WEST_COAST'])).astype(int)
    
    return df


def create_carrier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create carrier-specific features.
    
    Args:
        df: DataFrame with OP_CARRIER column
        
    Returns:
        DataFrame with additional carrier-based features
    """
    df = df.copy()
    
    # Carrier type categories (based on common US carriers)
    major_carriers = ['AA', 'DL', 'UA', 'WN']  # American, Delta, United, Southwest
    low_cost_carriers = ['WN', 'B6', 'NK', 'F9', 'G4']  # Southwest, JetBlue, Spirit, Frontier, Allegiant
    regional_carriers = ['OO', 'YX', 'MQ', 'OH', 'QX', '9E']  # SkyWest, Republic, Envoy, etc.
    
    df['IS_MAJOR_CARRIER'] = df['OP_CARRIER'].isin(major_carriers).astype(int)
    df['IS_LOW_COST_CARRIER'] = df['OP_CARRIER'].isin(low_cost_carriers).astype(int)
    df['IS_REGIONAL_CARRIER'] = df['OP_CARRIER'].isin(regional_carriers).astype(int)
    
    # Calculate carrier-specific statistics (historical performance)
    carrier_stats = df.groupby('OP_CARRIER').agg({
        'ARR_DEL15': ['mean', 'count'],
        'DISTANCE': 'mean'
    }).round(4)
    
    carrier_stats.columns = ['CARRIER_DELAY_RATE', 'CARRIER_FLIGHT_COUNT', 'CARRIER_AVG_DISTANCE']
    carrier_stats = carrier_stats.reset_index()
    
    # Merge carrier statistics back to main dataframe
    df = df.merge(carrier_stats, on='OP_CARRIER', how='left')
    
    # Carrier size categories based on flight count
    df['CARRIER_SIZE'] = pd.cut(
        df['CARRIER_FLIGHT_COUNT'],
        bins=[0, 1000, 10000, 50000, float('inf')],
        labels=['SMALL', 'MEDIUM', 'LARGE', 'MAJOR'],
        include_lowest=True
    )
    
    return df


def create_aircraft_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create aircraft-specific features from tail number and other aircraft data.
    
    Args:
        df: DataFrame with TAIL_NUM column and optionally aircraft registry data
        
    Returns:
        DataFrame with additional aircraft-based features
    """
    df = df.copy()
    
    if 'TAIL_NUM' not in df.columns:
        return df
    
    # Clean tail numbers
    df['TAIL_NUM_CLEAN'] = df['TAIL_NUM'].astype(str).str.strip().str.upper()
    
    # Extract aircraft age patterns from tail number (US registration patterns)
    # Note: This is a simplified approach - actual aircraft age would require registry data
    df['TAIL_NUM_PREFIX'] = df['TAIL_NUM_CLEAN'].str[:2]
    df['TAIL_NUM_NUMERIC'] = df['TAIL_NUM_CLEAN'].str.extract(r'(\d+)').astype(float)
    
    # Aircraft utilization (flights per tail number)
    tail_stats = df.groupby('TAIL_NUM_CLEAN').agg({
        'ARR_DEL15': ['mean', 'count'],
        'DISTANCE': 'mean'
    }).round(4)
    
    tail_stats.columns = ['AIRCRAFT_DELAY_RATE', 'AIRCRAFT_FLIGHT_COUNT', 'AIRCRAFT_AVG_DISTANCE']
    tail_stats = tail_stats.reset_index()
    tail_stats = tail_stats.rename(columns={'TAIL_NUM_CLEAN': 'TAIL_NUM_CLEAN'})
    
    # Merge aircraft statistics
    df = df.merge(tail_stats, on='TAIL_NUM_CLEAN', how='left')
    
    # Aircraft utilization categories
    df['AIRCRAFT_UTILIZATION'] = pd.cut(
        df['AIRCRAFT_FLIGHT_COUNT'],
        bins=[0, 10, 50, 200, float('inf')],
        labels=['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'],
        include_lowest=True
    )
    
    # Load aircraft registry data if available
    registry_path = Path('data/raw/faa_registry/MASTER.txt')
    if registry_path.exists():
        try:
            # Load FAA registry data for aircraft manufacturing year
            registry_df = pd.read_csv(
                registry_path,
                sep=',',
                usecols=['N-NUMBER', 'MFR MDL CODE', 'ENG MFR MDL', 'YEAR MFR'],
                dtype=str
            )
            registry_df['TAIL_NUM_CLEAN'] = 'N' + registry_df['N-NUMBER'].str.strip()
            registry_df['MFR_YEAR'] = pd.to_numeric(registry_df['YEAR MFR'], errors='coerce')
            
            # Merge manufacturing year
            df = df.merge(
                registry_df[['TAIL_NUM_CLEAN', 'MFR_YEAR']],
                on='TAIL_NUM_CLEAN',
                how='left'
            )
            
            # Calculate aircraft age
            current_year = df['FL_DATE'].dt.year.max()
            df['AIRCRAFT_AGE'] = current_year - df['MFR_YEAR']
            
            # Aircraft age categories
            df['AIRCRAFT_AGE_CATEGORY'] = pd.cut(
                df['AIRCRAFT_AGE'],
                bins=[0, 5, 15, 25, float('inf')],
                labels=['NEW', 'MODERN', 'MATURE', 'OLD'],
                include_lowest=True
            )
            
        except Exception as e:
            print(f"Warning: Could not load aircraft registry data: {e}")
    
    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional weather-based features beyond basic weather data.
    
    Args:
        df: DataFrame with weather columns (ORIGIN_TEMP, ORIGIN_WSPD, etc.)
        
    Returns:
        DataFrame with additional weather-based features
    """
    df = df.copy()
    
    # Weather columns
    origin_weather_cols = ['ORIGIN_TEMP', 'ORIGIN_WSPD', 'ORIGIN_PRCP', 'ORIGIN_PRES']
    dest_weather_cols = ['DEST_TEMP', 'DEST_WSPD', 'DEST_PRCP', 'DEST_PRES']
    
    # Check if weather columns exist
    if not all(col in df.columns for col in origin_weather_cols + dest_weather_cols):
        print("Warning: Weather columns not found. Skipping weather feature engineering.")
        return df
    
    # Temperature difference between origin and destination
    df['TEMP_DIFF'] = df['DEST_TEMP'] - df['ORIGIN_TEMP']
    df['TEMP_DIFF_ABS'] = abs(df['TEMP_DIFF'])
    
    # Pressure difference (weather system changes)
    df['PRESSURE_DIFF'] = df['DEST_PRES'] - df['ORIGIN_PRES']
    df['PRESSURE_DIFF_ABS'] = abs(df['PRESSURE_DIFF'])
    
    # Combined weather risk score
    df['WEATHER_RISK_COMBINED'] = df['WEATHER_RISK_ORIGIN'] + df['WEATHER_RISK_DEST']
    
    # Extreme weather indicators
    df['EXTREME_COLD_ORIGIN'] = (df['ORIGIN_TEMP'] < -10).astype(int)  # Below -10°C
    df['EXTREME_HEAT_ORIGIN'] = (df['ORIGIN_TEMP'] > 35).astype(int)   # Above 35°C
    df['EXTREME_COLD_DEST'] = (df['DEST_TEMP'] < -10).astype(int)
    df['EXTREME_HEAT_DEST'] = (df['DEST_TEMP'] > 35).astype(int)
    
    # High wind indicators
    df['HIGH_WIND_ORIGIN'] = (df['ORIGIN_WSPD'] > 25).astype(int)  # > 25 km/h
    df['HIGH_WIND_DEST'] = (df['DEST_WSPD'] > 25).astype(int)
    
    # Heavy precipitation indicators
    df['HEAVY_RAIN_ORIGIN'] = (df['ORIGIN_PRCP'] > 2.0).astype(int)  # > 2mm
    df['HEAVY_RAIN_DEST'] = (df['DEST_PRCP'] > 2.0).astype(int)
    
    # Weather severity score (0-10 scale)
    df['WEATHER_SEVERITY_ORIGIN'] = (
        df['EXTREME_COLD_ORIGIN'] * 2 +
        df['EXTREME_HEAT_ORIGIN'] * 1 +
        df['HIGH_WIND_ORIGIN'] * 2 +
        df['HEAVY_RAIN_ORIGIN'] * 3 +
        (df['ORIGIN_PRCP'] > 0.1).astype(int) * 1  # Any precipitation
    ).clip(0, 10)
    
    df['WEATHER_SEVERITY_DEST'] = (
        df['EXTREME_COLD_DEST'] * 2 +
        df['EXTREME_HEAT_DEST'] * 1 +
        df['HIGH_WIND_DEST'] * 2 +
        df['HEAVY_RAIN_DEST'] * 3 +
        (df['DEST_PRCP'] > 0.1).astype(int) * 1
    ).clip(0, 10)
    
    df['WEATHER_SEVERITY_MAX'] = df[['WEATHER_SEVERITY_ORIGIN', 'WEATHER_SEVERITY_DEST']].max(axis=1)
    
    return df


def create_advanced_temporal_features(df: pd.DataFrame, window_hours: int = 6) -> pd.DataFrame:
    """Create advanced temporal features including rolling statistics and delay propagation.
    
    Args:
        df: DataFrame with FL_DATE, DEP_HOUR, ORIGIN columns
        window_hours: Hours to look back for rolling statistics
        
    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()
    
    # Ensure we have datetime column
    if 'FL_DATETIME' not in df.columns:
        if 'CRS_DEP_TIME' in df.columns:
            # Create full datetime from date and time
            df['CRS_DEP_TIME_STR'] = df['CRS_DEP_TIME'].astype(str).str.zfill(4)
            hours = df['CRS_DEP_TIME_STR'].str[:2].astype(int)
            minutes = df['CRS_DEP_TIME_STR'].str[2:].astype(int)
            df['FL_DATETIME'] = (pd.to_datetime(df['FL_DATE']) + 
                                pd.to_timedelta(hours, unit='h') + 
                                pd.to_timedelta(minutes, unit='m'))
        else:
            print("Warning: Cannot create FL_DATETIME. Skipping advanced temporal features.")
            return df
    
    # Sort by datetime for rolling calculations
    df = df.sort_values('FL_DATETIME')
    
    # Calculate rolling statistics by airport (simplified version for performance)
    # Note: Full implementation would require more sophisticated time-aware grouping
    
    # Airport delay patterns by hour
    hourly_airport_stats = df.groupby(['ORIGIN', 'DEP_HOUR']).agg({
        'ARR_DEL15': ['mean', 'count']
    }).round(4)
    
    hourly_airport_stats.columns = ['AIRPORT_HOUR_DELAY_RATE', 'AIRPORT_HOUR_FLIGHT_COUNT']
    hourly_airport_stats = hourly_airport_stats.reset_index()
    
    df = df.merge(hourly_airport_stats, on=['ORIGIN', 'DEP_HOUR'], how='left')
    
    # Day of week patterns by airport
    df['WEEKDAY_NUM'] = df['FL_DATE'].dt.weekday
    weekday_airport_stats = df.groupby(['ORIGIN', 'WEEKDAY_NUM']).agg({
        'ARR_DEL15': ['mean', 'count']
    }).round(4)
    
    weekday_airport_stats.columns = ['AIRPORT_WEEKDAY_DELAY_RATE', 'AIRPORT_WEEKDAY_FLIGHT_COUNT']
    weekday_airport_stats = weekday_airport_stats.reset_index()
    
    df = df.merge(weekday_airport_stats, on=['ORIGIN', 'WEEKDAY_NUM'], how='left')
    
    # Route-specific patterns
    route_stats = df.groupby('ROUTE').agg({
        'ARR_DEL15': ['mean', 'count'],
        'DISTANCE': 'mean'
    }).round(4)
    
    route_stats.columns = ['ROUTE_DELAY_RATE', 'ROUTE_FLIGHT_COUNT', 'ROUTE_AVG_DISTANCE']
    route_stats = route_stats.reset_index()
    
    df = df.merge(route_stats, on='ROUTE', how='left')
    
    return df


def prepare_features_for_modeling(df: pd.DataFrame, target_col: str = 'ARR_DEL15') -> Tuple[pd.DataFrame, List[str]]:
    """Prepare the final feature set for modeling by selecting relevant columns and handling encoding.
    
    Args:
        df: DataFrame with all engineered features
        target_col: Name of target column
        
    Returns:
        Tuple of (features_df, feature_names_list)
    """
    df = df.copy()
    
    # Define columns to drop (leakage, IDs, etc.)
    drop_cols = [
        target_col,           # Target variable
        'FL_DATE',           # Date string (we have engineered time features)
        'FL_DATETIME',       # Datetime (we have engineered time features)
        'DEP_DELAY',         # Direct leakage
        'ARR_DELAY',         # Direct leakage
        'OP_CARRIER_FL_NUM', # ID column
        'TAIL_NUM',          # ID column (we have engineered features from it)
        'TAIL_NUM_CLEAN',    # Processed ID column
        'CRS_DEP_TIME',      # Raw time (we have engineered features)
        'CRS_ARR_TIME',      # Raw time (we have engineered features)
        'CRS_DEP_TIME_STR',  # Processed time string
        'CANCELLED',         # Post-flight information
        'DIVERTED',          # Post-flight information
        'DEP_DEL15',         # Related to departure delay (potential leakage)
        'WEEKDAY',           # String version (we have numeric versions)
        'REVERSE_ROUTE',     # Helper column
        'TAIL_NUM_PREFIX',   # Helper column
        'TAIL_NUM_NUMERIC',  # Helper column
        'WEEKDAY_NUM',       # Helper column (we have DAY_OF_WEEK)
    ]
    
    # Remove columns that exist in the dataframe
    drop_cols = [col for col in drop_cols if col in df.columns]
    features_df = df.drop(columns=drop_cols, errors='ignore')
    
    # Get categorical columns that need encoding
    categorical_cols = []
    for col in features_df.columns:
        if features_df[col].dtype == 'object' or features_df[col].dtype.name == 'category':
            categorical_cols.append(col)
    
    # One-hot encode categorical variables with reasonable cardinality
    for col in categorical_cols:
        unique_count = features_df[col].nunique()
        if unique_count <= 50:  # Only encode if reasonable number of categories
            # Create dummy variables
            dummies = pd.get_dummies(features_df[col], prefix=col, drop_first=True)
            features_df = pd.concat([features_df, dummies], axis=1)
            features_df = features_df.drop(columns=[col])
        else:
            # For high cardinality categorical variables, use label encoding or drop
            print(f"Warning: Dropping high cardinality categorical column: {col} ({unique_count} unique values)")
            features_df = features_df.drop(columns=[col])
    
    # Fill any remaining missing values
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
    
    # Get final feature names
    feature_names = features_df.columns.tolist()
    
    print(f"Final feature set: {len(feature_names)} features")
    print(f"Feature types: {features_df.dtypes.value_counts().to_dict()}")
    
    return features_df, feature_names

def prepare_features_with_custom_drops(df: pd.DataFrame, additional_drops: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare features for modeling with additional custom columns to drop.
    
    Args:
        df: DataFrame with all engineered features
        additional_drops: Additional columns to drop beyond the default ones
        
    Returns:
        Tuple of (features_df, feature_names_list)
    """
    if additional_drops is None:
        additional_drops = []
    
    # Get the base feature preparation
    features_df, feature_names = prepare_features_for_modeling(df)
    
    # Drop your additional columns
    features_df = features_df.drop(columns=additional_drops, errors='ignore')
    feature_names = [col for col in feature_names if col not in additional_drops]
    
    return features_df, feature_names

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering functions in the correct order.
    
    Args:
        df: Raw flight data DataFrame
        
    Returns:
        DataFrame with all engineered features
    """
    print("🔧 Starting feature engineering...")
    
    # Apply feature engineering in order
    df = create_time_features(df)
    print("✅ Time features created")
    
    df = create_route_features(df)
    print("✅ Route features created")
    
    df = create_carrier_features(df)
    print("✅ Carrier features created")
    
    df = create_aircraft_features(df)
    print("✅ Aircraft features created")
    
    df = create_weather_features(df)
    print("✅ Weather features created")
    
    df = create_advanced_temporal_features(df)
    print("✅ Advanced temporal features created")
    
    print(f"🔧 Feature engineering complete! Dataset shape: {df.shape}")
    
    return df