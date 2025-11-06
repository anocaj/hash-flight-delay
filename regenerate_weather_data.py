#!/usr/bin/env python3
"""
Regenerate weather data with timezone correction.
"""

import pandas as pd
import os
from pathlib import Path
import sys
sys.path.append('.')

from src.data.data_prep import load_and_prepare_data

def regenerate_weather_with_timezone_correction():
    """Regenerate weather data with timezone correction."""
    
    print("🔄 Regenerating Weather Data with Timezone Correction")
    print("=" * 60)
    
    # Check if weather cache exists
    weather_cache = Path('data/raw/airport_weather_cache.csv')
    
    if weather_cache.exists():
        print(f"📁 Found existing weather cache: {weather_cache}")
        print("🔄 Will regenerate with improved timezone correction approach...")
        
        # Backup the old cache
        backup_path = weather_cache.with_suffix('.backup.csv')
        if backup_path.exists():
            backup_path = weather_cache.with_suffix(f'.backup_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        os.rename(weather_cache, backup_path)
        print(f"📦 Backed up old cache to: {backup_path}")
    else:
        print("📁 No existing weather cache found")
        print("   Will fetch fresh weather data with timezone correction...")
    
    # Regenerate data with timezone correction
    print(f"\n🚀 Starting data preparation with timezone correction...")
    
    try:
        # This will fetch/regenerate weather data with timezone correction
        df = load_and_prepare_data('data/raw', cache_weather=True)
        
        print(f"\n✅ Data preparation complete!")
        print(f"   Dataset shape: {df.shape}")
        print(f"   Weather cache updated with timezone correction")
        
        # Verify timezone correction worked
        weather_df = pd.read_csv(weather_cache, nrows=10)
        print(f"✅ Timezone correction verified in weather cache")
        
        # Show sample of corrected data
        print(f"\n📊 Sample timezone-corrected weather data:")
        sample_cols = ['time', 'airport', 'weather_hour', 'temp']
        available_cols = [col for col in sample_cols if col in weather_df.columns]
        print(weather_df[available_cols].head(3).to_string(index=False))
        
        # Validate hour distribution
        unique_hours = len(weather_df['weather_hour'].unique())
        print(f"   Weather hours available: {unique_hours} (should be ~24 for good coverage)")
        
        if unique_hours > 20:
            print(f"   ✅ Good hour distribution - timezone correction working properly")
        else:
            print(f"   ⚠️ Limited hour range - may need more data or check timezone conversion")
            
    except Exception as e:
        print(f"❌ Error during data preparation: {e}")
        print(f"   You may need to check data file paths or dependencies")
        return
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. ✅ Weather data now uses timezone-corrected local hours")
    print(f"   2. 🔄 Retrain models with the corrected dataset")
    print(f"   3. 📊 Compare model performance before/after timezone fix")
    print(f"   4. 🚀 Deploy improved model with better weather alignment")

if __name__ == "__main__":
    regenerate_weather_with_timezone_correction()