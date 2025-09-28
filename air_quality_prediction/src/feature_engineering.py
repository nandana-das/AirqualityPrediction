"""
Feature Engineering Module for Air Quality Prediction

This module handles feature creation, selection, and engineering
for the air quality prediction project.

Author: MTech AI & Data Science Program
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import os
import logging
from datetime import datetime, timedelta
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AirQualityFeatureEngineer:
    """
    Comprehensive feature engineering class for air quality data.
    
    Creates temporal features, lag features, rolling averages, and interaction terms.
    """
    
    def __init__(self, data_path: str, output_dir: str = "data/features"):
        """
        Initialize the feature engineer.
        
        Args:
            data_path (str): Path to the processed data file
            output_dir (str): Directory to save engineered features
        """
        self.data_path = data_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Feature lists for different categories
        self.pollutant_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
        self.temporal_features = []
        self.lag_features = []
        self.rolling_features = []
        self.interaction_features = []
        self.engineered_features = []
        
        logger.info(f"Initialized AirQualityFeatureEngineer with data path: {data_path}")
    
    def load_processed_data(self) -> pd.DataFrame:
        """
        Load the processed air quality dataset.
        
        Returns:
            pd.DataFrame: Processed dataset
        """
        try:
            # TODO: Implement data loading logic
            # Load processed data from preprocessing step
            # Validate data format and required columns
            
            logger.info("Loading processed data...")
            # Placeholder - replace with actual loading logic
            data = pd.DataFrame()
            
            logger.info(f"Processed data loaded successfully. Shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise
    
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from date column.
        
        Args:
            data (pd.DataFrame): Input dataset with date column
            
        Returns:
            pd.DataFrame: Dataset with temporal features added
        """
        try:
            # TODO: Implement temporal feature creation
            # Extract year, month, day, day_of_week, season
            # Apply cyclical encoding for temporal features
            # Handle different date formats and timezones
            
            logger.info("Creating temporal features...")
            enhanced_data = data.copy()
            
            # Convert date column to datetime if not already
            if 'Date' in enhanced_data.columns:
                enhanced_data['Date'] = pd.to_datetime(enhanced_data['Date'])
                
                # Extract basic temporal features
                enhanced_data['year'] = enhanced_data['Date'].dt.year
                enhanced_data['month'] = enhanced_data['Date'].dt.month
                enhanced_data['day'] = enhanced_data['Date'].dt.day
                enhanced_data['day_of_week'] = enhanced_data['Date'].dt.dayofweek
                enhanced_data['day_of_year'] = enhanced_data['Date'].dt.dayofyear
                
                # Create season feature
                enhanced_data['season'] = enhanced_data['month'].map({
                    12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Fall', 10: 'Fall', 11: 'Fall'
                })
                
                # TODO: Apply cyclical encoding for month and day_of_week
                # enhanced_data['month_sin'] = np.sin(2 * np.pi * enhanced_data['month'] / 12)
                # enhanced_data['month_cos'] = np.cos(2 * np.pi * enhanced_data['month'] / 12)
                
                self.temporal_features = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'season']
                
                logger.info(f"Created {len(self.temporal_features)} temporal features")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error creating temporal features: {str(e)}")
            raise
    
    def create_lag_features(self, data: pd.DataFrame, lag_days: List[int] = [1, 2, 3, 7]) -> pd.DataFrame:
        """
        Create lag features for pollutants and AQI.
        
        Args:
            data (pd.DataFrame): Input dataset
            lag_days (List[int]): List of lag days to create
            
        Returns:
            pd.DataFrame: Dataset with lag features added
        """
        try:
            # TODO: Implement lag feature creation
            # Create lag features for all pollutants and AQI
            # Handle city-wise grouping for lag creation
            # Handle missing values in lag features
            
            logger.info(f"Creating lag features for days: {lag_days}")
            lagged_data = data.copy()
            
            # Sort data by city and date
            if 'City' in lagged_data.columns and 'Date' in lagged_data.columns:
                lagged_data = lagged_data.sort_values(['City', 'Date'])
                
                # Create lag features for each pollutant and AQI
                features_to_lag = self.pollutant_features + ['AQI']
                
                for feature in features_to_lag:
                    if feature in lagged_data.columns:
                        for lag in lag_days:
                            lag_col_name = f'prev_{lag}d_{feature}'
                            lagged_data[lag_col_name] = lagged_data.groupby('City')[feature].shift(lag)
                            self.lag_features.append(lag_col_name)
                
                logger.info(f"Created {len(self.lag_features)} lag features")
            
            return lagged_data
            
        except Exception as e:
            logger.error(f"Error creating lag features: {str(e)}")
            raise
    
    def create_rolling_features(self, data: pd.DataFrame, windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
        """
        Create rolling average and standard deviation features.
        
        Args:
            data (pd.DataFrame): Input dataset
            windows (List[int]): List of window sizes for rolling features
            
        Returns:
            pd.DataFrame: Dataset with rolling features added
        """
        try:
            # TODO: Implement rolling feature creation
            # Create 3-day and 7-day rolling averages and std dev
            # Handle city-wise grouping for rolling calculations
            # Handle edge cases (insufficient data for rolling windows)
            
            logger.info(f"Creating rolling features for windows: {windows}")
            rolling_data = data.copy()
            
            # Sort data by city and date
            if 'City' in rolling_data.columns and 'Date' in rolling_data.columns:
                rolling_data = rolling_data.sort_values(['City', 'Date'])
                
                # Create rolling features for each pollutant
                for feature in self.pollutant_features:
                    if feature in rolling_data.columns:
                        for window in windows:
                            # Rolling mean
                            mean_col = f'{window}d_avg_{feature}'
                            rolling_data[mean_col] = rolling_data.groupby('City')[feature].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
                            
                            # Rolling standard deviation
                            std_col = f'{window}d_std_{feature}'
                            rolling_data[std_col] = rolling_data.groupby('City')[feature].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
                            
                            self.rolling_features.extend([mean_col, std_col])
                
                logger.info(f"Created {len(self.rolling_features)} rolling features")
            
            return rolling_data
            
        except Exception as e:
            logger.error(f"Error creating rolling features: {str(e)}")
            raise
    
    def create_ratio_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio features between different pollutants.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with ratio features added
        """
        try:
            # TODO: Implement ratio feature creation
            # Create PM2.5/PM10 ratio
            # Create other meaningful pollutant ratios
            # Handle division by zero cases
            
            logger.info("Creating ratio features...")
            ratio_data = data.copy()
            
            # PM2.5/PM10 ratio
            if 'PM2.5' in ratio_data.columns and 'PM10' in ratio_data.columns:
                ratio_data['PM25_PM10_ratio'] = np.where(
                    ratio_data['PM10'] != 0, 
                    ratio_data['PM2.5'] / ratio_data['PM10'], 
                    0
                )
                self.engineered_features.append('PM25_PM10_ratio')
            
            # TODO: Create other meaningful ratios
            # NO2/NO ratio, SO2/NO2 ratio, etc.
            
            logger.info(f"Created {len([f for f in self.engineered_features if 'ratio' in f])} ratio features")
            return ratio_data
            
        except Exception as e:
            logger.error(f"Error creating ratio features: {str(e)}")
            raise
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key pollutants and weather conditions.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with interaction features added
        """
        try:
            # TODO: Implement interaction feature creation
            # Create pollutant interaction terms
            # Create temporal-pollutant interactions
            # Select most relevant interactions based on domain knowledge
            
            logger.info("Creating interaction features...")
            interaction_data = data.copy()
            
            # TODO: Create meaningful interaction features
            # PM2.5 * temperature, PM10 * humidity, etc.
            # Consider polynomial features for key pollutants
            
            logger.info(f"Created {len([f for f in self.engineered_features if 'interaction' in f])} interaction features")
            return interaction_data
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {str(e)}")
            raise
    
    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features (min, max, range, etc.) for pollutants.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with statistical features added
        """
        try:
            # TODO: Implement statistical feature creation
            # Create min, max, range, skewness, kurtosis for pollutants
            # Consider city-wise and temporal aggregations
            
            logger.info("Creating statistical features...")
            statistical_data = data.copy()
            
            # TODO: Implement statistical feature creation logic
            
            logger.info(f"Created {len([f for f in self.engineered_features if 'stat' in f])} statistical features")
            return statistical_data
            
        except Exception as e:
            logger.error(f"Error creating statistical features: {str(e)}")
            raise
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'mutual_info', k: int = 50) -> List[str]:
        """
        Select top-k features using specified method.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Feature selection method ('mutual_info', 'f_regression', 'correlation')
            k (int): Number of features to select
            
        Returns:
            List[str]: List of selected feature names
        """
        try:
            # TODO: Implement feature selection
            # Support multiple feature selection methods
            # Handle categorical features appropriately
            # Log feature importance scores
            
            logger.info(f"Selecting top {k} features using {method} method...")
            
            if method == 'mutual_info':
                # TODO: Implement mutual information-based selection
                selector = SelectKBest(score_func=mutual_info_regression, k=k)
                # X_selected = selector.fit_transform(X, y)
                # selected_features = X.columns[selector.get_support()].tolist()
                selected_features = X.columns[:k].tolist()
            
            elif method == 'f_regression':
                # TODO: Implement F-regression-based selection
                selector = SelectKBest(score_func=f_regression, k=k)
                # X_selected = selector.fit_transform(X, y)
                # selected_features = X.columns[selector.get_support()].tolist()
                selected_features = X.columns[:k].tolist()
            
            elif method == 'correlation':
                # TODO: Implement correlation-based selection
                # Calculate correlation with target and select top features
                selected_features = X.columns[:k].tolist()
            
            logger.info(f"Selected {len(selected_features)} features")
            return selected_features
            
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            raise
    
    def save_engineered_features(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save engineered features to CSV file.
        
        Args:
            data (pd.DataFrame): Dataset with engineered features
            filename (str): Output filename
        """
        try:
            # TODO: Implement feature saving
            # Save to CSV with appropriate formatting
            # Save feature metadata (feature lists, importance scores)
            
            output_path = os.path.join(self.output_dir, filename)
            # data.to_csv(output_path, index=False)
            logger.info(f"Engineered features saved to: {output_path}")
            
            # Save feature metadata
            metadata = {
                'temporal_features': self.temporal_features,
                'lag_features': self.lag_features,
                'rolling_features': self.rolling_features,
                'engineered_features': self.engineered_features,
                'total_features': len(data.columns)
            }
            
            # TODO: Save metadata to JSON or pickle file
            
        except Exception as e:
            logger.error(f"Error saving engineered features: {str(e)}")
            raise
    
    def run_full_feature_engineering(self) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Returns:
            pd.DataFrame: Dataset with all engineered features
        """
        try:
            logger.info("Starting full feature engineering pipeline...")
            
            # TODO: Implement complete feature engineering pipeline
            # 1. Load processed data
            # 2. Create temporal features
            # 3. Create lag features
            # 4. Create rolling features
            # 5. Create ratio features
            # 6. Create interaction features
            # 7. Create statistical features
            # 8. Select best features
            # 9. Save engineered features
            
            # Placeholder implementation
            engineered_data = pd.DataFrame()
            
            logger.info("Full feature engineering pipeline completed successfully")
            return engineered_data
            
        except Exception as e:
            logger.error(f"Error in full feature engineering pipeline: {str(e)}")
            raise


def main():
    """
    Main function to run feature engineering.
    """
    try:
        # TODO: Implement main execution logic
        # Set up paths and parameters
        # Run feature engineering pipeline
        # Generate feature summary statistics
        
        logger.info("Starting feature engineering...")
        
        # Initialize feature engineer
        # engineer = AirQualityFeatureEngineer("data/processed/cleaned_data.csv")
        # engineered_data = engineer.run_full_feature_engineering()
        
        logger.info("Feature engineering completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main feature engineering: {str(e)}")
        raise


if __name__ == "__main__":
    main()
