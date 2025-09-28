#!/usr/bin/env python3
"""
Air Quality Data Preprocessing Script

This script runs the data preprocessing pipeline directly from Python
without needing Jupyter Notebook.

Author: MTech AI & Data Science Program
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings
import os
import sys

# Configure plotting and warnings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

def main():
    print("🌬️ AIR QUALITY DATA PREPROCESSING")
    print("=" * 60)
    
    # Step 1: Load raw data
    print("\n📁 LOADING RAW DATA")
    print("=" * 50)
    
    data_path = "data/raw/"
    
    # Check available files and load the dataset
    print("🔍 Checking available dataset files...")
    if os.path.exists(data_path):
        files = os.listdir(data_path)
        print(f"Available files: {files}")
        
        # Load the main dataset - city_day (daily city-level data)
        try:
            # Try loading city_day file (main dataset for our project)
            if 'city_day' in files:
                # Check if it's CSV or Excel format
                if 'city_day.csv' in files:
                    df_raw = pd.read_csv(data_path + 'city_day.csv')
                    print("✅ Loaded city_day.csv successfully!")
                else:
                    # Try as Excel file
                    df_raw = pd.read_excel(data_path + 'city_day')
                    print("✅ Loaded city_day Excel file successfully!")
                
                print(f"📊 Raw dataset shape: {df_raw.shape}")
                print(f"📋 Columns: {list(df_raw.columns)}")
                
            else:
                print("⚠️ city_day file not found. Using first available file...")
                available_files = [f for f in files if f.endswith(('.csv', '.xlsx', '.xls')) or '.' not in f]
                if available_files:
                    first_file = available_files[0]
                    if first_file.endswith('.csv'):
                        df_raw = pd.read_csv(data_path + first_file)
                    else:
                        df_raw = pd.read_excel(data_path + first_file)
                    print(f"✅ Loaded {first_file} as main dataset")
                else:
                    print("❌ No suitable data files found")
                    df_raw = pd.DataFrame()
                    
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            df_raw = pd.DataFrame()
            
    else:
        print(f"❌ Data directory not found: {data_path}")
        df_raw = pd.DataFrame()
    
    if df_raw.empty:
        print("❌ No data loaded. Exiting...")
        return
    
    # Step 2: Data Exploration
    print("\n📊 RAW DATASET ANALYSIS")
    print("=" * 60)
    
    # Basic information
    print(f"Dataset Shape: {df_raw.shape}")
    print(f"Memory Usage: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types and missing values
    print("\n📝 Data Types and Missing Values:")
    missing_info = pd.DataFrame({
        'Data Type': df_raw.dtypes,
        'Missing Values': df_raw.isnull().sum(),
        'Missing %': (df_raw.isnull().sum() / len(df_raw) * 100).round(2)
    })
    print(missing_info)
    
    # Check for target cities
    target_cities = ['Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Visakhapatnam']
    
    if 'City' in df_raw.columns:
        available_cities = df_raw['City'].unique()
        print(f"\n🏙️ Total Cities in Dataset: {len(available_cities)}")
        print(f"First 10 cities: {available_cities[:10]}")
        
        # Check target cities
        target_cities_found = [city for city in target_cities if city in available_cities]
        print(f"\n🎯 Target Cities Found ({len(target_cities_found)}/{len(target_cities)}):")
        for city in target_cities:
            if city in available_cities:
                count = len(df_raw[df_raw['City'] == city])
                print(f"  ✅ {city}: {count:,} records")
            else:
                print(f"  ❌ {city}: Not found")
        
        # Check for similar city names
        if len(target_cities_found) < len(target_cities):
            print(f"\n🔍 Checking for similar city names...")
            for target in target_cities:
                if target not in available_cities:
                    similar = [city for city in available_cities if target.lower() in city.lower() or city.lower() in target.lower()]
                    if similar:
                        print(f"  💡 {target} → Similar: {similar}")
    else:
        print(f"\n⚠️ 'City' column not found. Available columns: {list(df_raw.columns)}")
    
    # Check AQI column
    if 'AQI' in df_raw.columns:
        print(f"\n🌬️ AQI Analysis:")
        print(f"  Range: {df_raw['AQI'].min():.2f} to {df_raw['AQI'].max():.2f}")
        print(f"  Mean: {df_raw['AQI'].mean():.2f}")
        print(f"  Median: {df_raw['AQI'].median():.2f}")
        print(f"  Missing AQI values: {df_raw['AQI'].isnull().sum():,} ({df_raw['AQI'].isnull().sum()/len(df_raw)*100:.2f}%)")
    else:
        print(f"\n⚠️ 'AQI' column not found. Available columns: {list(df_raw.columns)}")
    
    # Step 3: Filter data for target cities
    print("\n🏙️ FILTERING FOR TARGET CITIES")
    print("=" * 50)
    
    if 'City' in df_raw.columns:
        # Find exact matches and similar names
        city_mapping = {}
        for target in target_cities:
            if target in available_cities:
                city_mapping[target] = target
            else:
                # Look for similar names (case-insensitive)
                similar = [city for city in available_cities 
                          if target.lower() in city.lower() or city.lower() in target.lower()]
                if similar:
                    city_mapping[target] = similar[0]  # Take first match
                    print(f"  💡 Mapping: {target} → {similar[0]}")
        
        # Filter data for target cities
        target_city_data = []
        for target, actual_city in city_mapping.items():
            city_data = df_raw[df_raw['City'] == actual_city].copy()
            city_data['City'] = target  # Standardize city name
            target_city_data.append(city_data)
            print(f"  ✅ {target}: {len(city_data):,} records")
        
        if target_city_data:
            df_filtered = pd.concat(target_city_data, ignore_index=True)
            print(f"\n📊 Filtered dataset shape: {df_filtered.shape}")
        else:
            print("❌ No target cities found in dataset")
            df_filtered = df_raw.copy()
    else:
        print("⚠️ Cannot filter cities - City column not found")
        df_filtered = df_raw.copy()
    
    # Step 4: Handle missing values - Remove rows with null AQI
    print("\n🧹 HANDLING MISSING VALUES")
    print("=" * 50)
    
    if 'AQI' in df_filtered.columns:
        initial_shape = df_filtered.shape
        print(f"Initial dataset shape: {initial_shape}")
        
        # Remove rows with null AQI (target variable)
        df_no_null_aqi = df_filtered.dropna(subset=['AQI'])
        null_aqi_removed = initial_shape[0] - df_no_null_aqi.shape[0]
        
        print(f"✅ Removed {null_aqi_removed:,} rows with null AQI")
        print(f"   Remaining rows: {df_no_null_aqi.shape[0]:,}")
        
        df_cleaned = df_no_null_aqi.copy()
    else:
        print("⚠️ Cannot clean missing values - AQI column not found")
        df_cleaned = df_filtered.copy()
    
    # Step 5: Outlier Detection and Treatment
    print("\n🔍 OUTLIER DETECTION AND TREATMENT")
    print("=" * 50)
    
    # Select numeric columns for outlier analysis
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if 'Date' not in col and col not in ['year', 'month', 'day']]
    
    print(f"Analyzing outliers in {len(numeric_cols)} numeric columns")
    
    # Detect outliers using IQR method and apply capping
    df_treated = df_cleaned.copy()
    outliers_treated = 0
    
    for col in numeric_cols:
        if col in df_cleaned.columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                # Cap outliers instead of removing them
                df_treated[col] = np.where(df_treated[col] < lower_bound, lower_bound, df_treated[col])
                df_treated[col] = np.where(df_treated[col] > upper_bound, upper_bound, df_treated[col])
                outliers_treated += outlier_count
                print(f"  {col}: {outlier_count:,} outliers treated")
    
    if outliers_treated > 0:
        print(f"\n✅ Treated {outliers_treated} outlier values using capping method")
    else:
        print(f"\n✅ No outliers found in the dataset")
    
    df_final = df_treated.copy()
    
    # Step 6: Save processed data
    print("\n💾 SAVING PROCESSED DATA")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    output_dir = "data/processed/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed dataset
    output_file = output_dir + "cleaned_data.csv"
    df_final.to_csv(output_file, index=False)
    
    print(f"✅ Processed data saved to: {output_file}")
    print(f"   Shape: {df_final.shape}")
    print(f"   File size: {os.path.getsize(output_file) / 1024**2:.2f} MB")
    
    # Display final summary
    print(f"\n📊 PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"Original dataset: {df_raw.shape}")
    print(f"After city filtering: {df_filtered.shape}")
    print(f"After missing value removal: {df_cleaned.shape}")
    print(f"Final processed dataset: {df_final.shape}")
    
    # City-wise summary
    if 'City' in df_final.columns:
        print(f"\n🏙️ City-wise record counts:")
        city_counts = df_final['City'].value_counts()
        for city, count in city_counts.items():
            print(f"  {city}: {count:,} records")
    
    # AQI summary
    if 'AQI' in df_final.columns:
        print(f"\n🌬️ Final AQI Statistics:")
        print(f"  Range: {df_final['AQI'].min():.2f} to {df_final['AQI'].max():.2f}")
        print(f"  Mean: {df_final['AQI'].mean():.2f}")
        print(f"  Std: {df_final['AQI'].std():.2f}")
    
    print(f"\n🎉 Preprocessing completed successfully!")
    print(f"Next step: Run feature engineering (03_feature_engineering.ipynb)")

if __name__ == "__main__":
    main()
