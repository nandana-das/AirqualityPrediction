#!/usr/bin/env python3
"""
Air Quality Feature Engineering Script

This script runs the feature engineering pipeline directly from Python
without needing Jupyter Notebook.

Author: MTech AI & Data Science Program
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
import os
import sys

# Configure plotting and warnings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

def main():
    print("🔧 AIR QUALITY FEATURE ENGINEERING")
    print("=" * 60)
    
    # Step 1: Load processed data
    print("\n📁 LOADING PROCESSED DATA")
    print("=" * 50)
    
    data_path = "data/processed/"
    
    try:
        df_processed = pd.read_csv(data_path + "cleaned_data.csv")
        print(f"✅ Loaded processed data successfully!")
        print(f"📊 Dataset shape: {df_processed.shape}")
        print(f"📋 Columns: {list(df_processed.columns)}")
        
        # Display basic info
        print(f"\n📈 Basic Information:")
        print(f"  Memory usage: {df_processed.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"  Date range: {df_processed['Date'].min()} to {df_processed['Date'].max()}")
        
        # Check cities
        if 'City' in df_processed.columns:
            city_counts = df_processed['City'].value_counts()
            print(f"  Cities: {dict(city_counts)}")
        
        # Check AQI
        if 'AQI' in df_processed.columns:
            print(f"  AQI range: {df_processed['AQI'].min():.2f} to {df_processed['AQI'].max():.2f}")
            print(f"  AQI mean: {df_processed['AQI'].mean():.2f}")
        
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")
        return
    
    # Step 2: Create temporal features
    print("\n📅 CREATING TEMPORAL FEATURES")
    print("=" * 50)
    
    if 'Date' in df_processed.columns:
        # Make a copy for feature engineering
        df_features = df_processed.copy()
        
        # Convert Date to datetime if not already
        df_features['Date'] = pd.to_datetime(df_features['Date'])
        
        # Extract basic temporal features
        df_features['year'] = df_features['Date'].dt.year
        df_features['month'] = df_features['Date'].dt.month
        df_features['day'] = df_features['Date'].dt.day
        df_features['day_of_week'] = df_features['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df_features['day_of_year'] = df_features['Date'].dt.dayofyear
        df_features['week_of_year'] = df_features['Date'].dt.isocalendar().week
        
        # Create season feature
        season_mapping = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        df_features['season'] = df_features['month'].map(season_mapping)
        
        # Create cyclical encoding for temporal features
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
        df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
        
        # Create weekend indicator
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        
        # Create month-end indicator
        df_features['is_month_end'] = (df_features['day'] >= 29).astype(int)
        
        temporal_features = [
            'year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year',
            'season', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            'day_of_year_sin', 'day_of_year_cos', 'is_weekend', 'is_month_end'
        ]
        
        print(f"✅ Created {len(temporal_features)} temporal features")
        
    else:
        print("⚠️ Cannot create temporal features - Date column not found")
        df_features = df_processed.copy()
        temporal_features = []
    
    # Step 3: Create lag features
    print("\n⏰ CREATING LAG FEATURES")
    print("=" * 50)
    
    if 'City' in df_features.columns:
        # Sort data by city and date for proper lag calculation
        df_features = df_features.sort_values(['City', 'Date']).reset_index(drop=True)
        
        # Define pollutant columns
        pollutant_columns = [col for col in df_features.columns 
                            if col in ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']]
        
        # Define lag days to create
        lag_days = [1, 2, 3, 7]  # Previous 1, 2, 3, and 7 days
        
        print(f"Creating lag features for {len(pollutant_columns)} pollutants with {lag_days} lag days")
        
        lag_features = []
        
        # Create lag features for pollutants
        for pollutant in pollutant_columns:
            if pollutant in df_features.columns:
                for lag in lag_days:
                    lag_col_name = f'prev_{lag}d_{pollutant}'
                    df_features[lag_col_name] = df_features.groupby('City')[pollutant].shift(lag)
                    lag_features.append(lag_col_name)
        
        # Create lag features for AQI
        if 'AQI' in df_features.columns:
            for lag in lag_days:
                lag_col_name = f'prev_{lag}d_AQI'
                df_features[lag_col_name] = df_features.groupby('City')['AQI'].shift(lag)
                lag_features.append(lag_col_name)
        
        print(f"✅ Created {len(lag_features)} lag features")
        
    else:
        print("⚠️ Cannot create lag features - City column not found")
        lag_features = []
    
    # Step 4: Create rolling features
    print("\n📈 CREATING ROLLING FEATURES")
    print("=" * 50)
    
    if 'City' in df_features.columns:
        # Define rolling windows
        rolling_windows = [3, 7, 14]  # 3-day, 7-day, and 14-day rolling windows
        
        rolling_features = []
        
        # Create rolling features for pollutants
        for pollutant in pollutant_columns:
            if pollutant in df_features.columns:
                for window in rolling_windows:
                    # Rolling mean
                    mean_col = f'{window}d_avg_{pollutant}'
                    df_features[mean_col] = df_features.groupby('City')[pollutant].rolling(
                        window=window, min_periods=1
                    ).mean().reset_index(0, drop=True)
                    rolling_features.append(mean_col)
                    
                    # Rolling standard deviation
                    std_col = f'{window}d_std_{pollutant}'
                    df_features[std_col] = df_features.groupby('City')[pollutant].rolling(
                        window=window, min_periods=1
                    ).std().reset_index(0, drop=True)
                    rolling_features.append(std_col)
                    
                    # Rolling maximum
                    max_col = f'{window}d_max_{pollutant}'
                    df_features[max_col] = df_features.groupby('City')[pollutant].rolling(
                        window=window, min_periods=1
                    ).max().reset_index(0, drop=True)
                    rolling_features.append(max_col)
        
        # Create rolling features for AQI
        if 'AQI' in df_features.columns:
            for window in rolling_windows:
                aqi_mean_col = f'{window}d_avg_AQI'
                df_features[aqi_mean_col] = df_features.groupby('City')['AQI'].rolling(
                    window=window, min_periods=1
                ).mean().reset_index(0, drop=True)
                rolling_features.append(aqi_mean_col)
        
        print(f"✅ Created {len(rolling_features)} rolling features")
        
    else:
        print("⚠️ Cannot create rolling features - City column not found")
        rolling_features = []
    
    # Step 5: Create ratio features
    print("\n🔗 CREATING RATIO FEATURES")
    print("=" * 50)
    
    ratio_features = []
    
    # PM2.5/PM10 ratio
    if 'PM2.5' in df_features.columns and 'PM10' in df_features.columns:
        df_features['PM25_PM10_ratio'] = np.where(
            df_features['PM10'] != 0, 
            df_features['PM2.5'] / df_features['PM10'], 
            0
        )
        ratio_features.append('PM25_PM10_ratio')
    
    # NO2/NO ratio
    if 'NO2' in df_features.columns and 'NO' in df_features.columns:
        df_features['NO2_NO_ratio'] = np.where(
            df_features['NO'] != 0, 
            df_features['NO2'] / df_features['NO'], 
            0
        )
        ratio_features.append('NO2_NO_ratio')
    
    # SO2/NO2 ratio
    if 'SO2' in df_features.columns and 'NO2' in df_features.columns:
        df_features['SO2_NO2_ratio'] = np.where(
            df_features['NO2'] != 0, 
            df_features['SO2'] / df_features['NO2'], 
            0
        )
        ratio_features.append('SO2_NO2_ratio')
    
    # CO/PM2.5 ratio
    if 'CO' in df_features.columns and 'PM2.5' in df_features.columns:
        df_features['CO_PM25_ratio'] = np.where(
            df_features['PM2.5'] != 0, 
            df_features['CO'] / df_features['PM2.5'], 
            0
        )
        ratio_features.append('CO_PM25_ratio')
    
    # O3/NO2 ratio
    if 'O3' in df_features.columns and 'NO2' in df_features.columns:
        df_features['O3_NO2_ratio'] = np.where(
            df_features['NO2'] != 0, 
            df_features['O3'] / df_features['NO2'], 
            0
        )
        ratio_features.append('O3_NO2_ratio')
    
    # Benzene/Toluene ratio
    if 'Benzene' in df_features.columns and 'Toluene' in df_features.columns:
        df_features['Benzene_Toluene_ratio'] = np.where(
            df_features['Toluene'] != 0, 
            df_features['Benzene'] / df_features['Toluene'], 
            0
        )
        ratio_features.append('Benzene_Toluene_ratio')
    
    # NH3/NOx ratio
    if 'NH3' in df_features.columns and 'NOx' in df_features.columns:
        df_features['NH3_NOx_ratio'] = np.where(
            df_features['NOx'] != 0, 
            df_features['NH3'] / df_features['NOx'], 
            0
        )
        ratio_features.append('NH3_NOx_ratio')
    
    print(f"✅ Created {len(ratio_features)} ratio features")
    
    # Step 6: Save engineered dataset
    print("\n💾 SAVING ENGINEERED DATASET")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    output_dir = "data/features/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the engineered dataset
    output_file = output_dir + "engineered_features.csv"
    df_features.to_csv(output_file, index=False)
    
    print(f"✅ Engineered dataset saved to: {output_file}")
    print(f"   Shape: {df_features.shape}")
    print(f"   File size: {os.path.getsize(output_file) / 1024**2:.2f} MB")
    
    # Calculate total engineered features
    all_engineered_features = temporal_features + lag_features + rolling_features + ratio_features
    total_engineered = len(all_engineered_features)
    
    # Final summary
    print(f"\n📊 FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"📈 Feature Engineering Results:")
    print(f"  Original features: {len(df_processed.columns)}")
    print(f"  Engineered features: {total_engineered}")
    print(f"  Total features: {len(df_features.columns)}")
    print(f"  Feature increase: {len(df_features.columns) - len(df_processed.columns)}")
    
    print(f"\n🔧 Feature Categories:")
    print(f"  📅 Temporal features: {len(temporal_features)}")
    print(f"  ⏰ Lag features: {len(lag_features)}")
    print(f"  📈 Rolling features: {len(rolling_features)}")
    print(f"  🔗 Ratio features: {len(ratio_features)}")
    
    print(f"\n📊 Dataset Information:")
    print(f"  Shape: {df_features.shape}")
    print(f"  Memory usage: {df_features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Date range: {df_features['Date'].min()} to {df_features['Date'].max()}")
    
    # Save feature metadata
    metadata_file = output_dir + "feature_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("FEATURE ENGINEERING METADATA\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Engineered Features: {total_engineered}\n")
        f.write(f"Dataset Shape: {df_features.shape}\n")
        f.write(f"Date Range: {df_features['Date'].min()} to {df_features['Date'].max()}\n\n")
        
        f.write("TEMPORAL FEATURES:\n")
        for feature in temporal_features:
            f.write(f"  - {feature}\n")
        
        f.write(f"\nLAG FEATURES ({len(lag_features)}):\n")
        for feature in lag_features:
            f.write(f"  - {feature}\n")
        
        f.write(f"\nROLLING FEATURES ({len(rolling_features)}):\n")
        for feature in rolling_features:
            f.write(f"  - {feature}\n")
        
        f.write(f"\nRATIO FEATURES ({len(ratio_features)}):\n")
        for feature in ratio_features:
            f.write(f"  - {feature}\n")
    
    print(f"✅ Feature metadata saved to: {metadata_file}")
    
    print(f"\n🎉 FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print(f"📈 Dataset transformed from {len(df_processed.columns)} to {len(df_features.columns)} features")
    print(f"🚀 Ready for model training! Next step: 04_model_training.ipynb")

if __name__ == "__main__":
    main()
