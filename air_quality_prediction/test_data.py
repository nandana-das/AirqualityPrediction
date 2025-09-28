#!/usr/bin/env python3
"""
Simple test script to check data loading
"""

import pandas as pd
import sys

print("Testing data loading...")
print("Python version:", sys.version)

try:
    # Load the dataset
    df = pd.read_csv('data/raw/city_day.csv')
    print("✅ Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    # Check for target cities
    target_cities = ['Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Visakhapatnam']
    if 'City' in df.columns:
        available_cities = df['City'].unique()
        print(f"\nTotal cities: {len(available_cities)}")
        print(f"Target cities found: {[city for city in target_cities if city in available_cities]}")
    
    # Check AQI
    if 'AQI' in df.columns:
        print(f"\nAQI range: {df['AQI'].min()} to {df['AQI'].max()}")
        print(f"Missing AQI values: {df['AQI'].isnull().sum()}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed!")
