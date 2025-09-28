"""
Visualization Module for Air Quality Prediction

This module provides comprehensive visualization utilities for
air quality prediction analysis and results.

Author: MTech AI & Data Science Program
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib and seaborn
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AirQualityVisualizer:
    """
    Comprehensive visualization class for air quality prediction analysis.
    
    Creates plots for EDA, model results, and feature analysis.
    """
    
    def __init__(self, data_path: str, output_dir: str = "results/plots"):
        """
        Initialize the visualizer.
        
        Args:
            data_path (str): Path to the dataset
            output_dir (str): Directory to save plots
        """
        self.data_path = data_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Target cities for visualization
        self.target_cities = [
            'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Visakhapatnam'
        ]
        
        # Color palette for cities
        self.city_colors = {
            'Delhi': '#FF6B6B',
            'Bangalore': '#4ECDC4',
            'Kolkata': '#45B7D1',
            'Hyderabad': '#96CEB4',
            'Chennai': '#FECA57',
            'Visakhapatnam': '#FF9FF3'
        }
        
        # Plot settings
        self.figsize = (12, 8)
        self.dpi = 300
        
        logger.info(f"Initialized AirQualityVisualizer with data path: {data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load dataset for visualization.
        
        Returns:
            pd.DataFrame: Dataset for visualization
        """
        try:
            # TODO: Implement data loading logic
            # Load appropriate dataset based on visualization needs
            # Handle different data formats
            
            logger.info("Loading data for visualization...")
            # Placeholder - replace with actual loading logic
            data = pd.DataFrame()
            
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def plot_data_overview(self, data: pd.DataFrame) -> None:
        """
        Create data overview visualizations.
        
        Args:
            data (pd.DataFrame): Input dataset
        """
        try:
            # TODO: Implement data overview plots
            # Dataset shape, null counts, data types
            # Basic statistics summary
            # Data distribution overview
            
            logger.info("Creating data overview visualizations...")
            
            # Create subplots for overview
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Air Quality Dataset Overview', fontsize=16, fontweight='bold')
            
            # 1. Dataset shape and basic info
            axes[0, 0].text(0.1, 0.8, f"Dataset Shape: {data.shape}", fontsize=12, 
                           transform=axes[0, 0].transAxes)
            axes[0, 0].text(0.1, 0.6, f"Columns: {len(data.columns)}", fontsize=12,
                           transform=axes[0, 0].transAxes)
            axes[0, 0].text(0.1, 0.4, f"Rows: {len(data)}", fontsize=12,
                           transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Dataset Information')
            axes[0, 0].axis('off')
            
            # 2. Missing values heatmap
            if not data.empty:
                missing_data = data.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                if len(missing_data) > 0:
                    missing_data.plot(kind='bar', ax=axes[0, 1], color='coral')
                    axes[0, 1].set_title('Missing Values by Column')
                    axes[0, 1].set_xlabel('Columns')
                    axes[0, 1].set_ylabel('Missing Count')
                    axes[0, 1].tick_params(axis='x', rotation=45)
                else:
                    axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center',
                                   transform=axes[0, 1].transAxes, fontsize=14)
                    axes[0, 1].set_title('Missing Values')
            
            # 3. Data types distribution
            if not data.empty:
                dtype_counts = data.dtypes.value_counts()
                axes[1, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
                axes[1, 0].set_title('Data Types Distribution')
            
            # 4. Sample data preview
            if not data.empty:
                sample_data = data.head().to_string()
                axes[1, 1].text(0.05, 0.95, sample_data, transform=axes[1, 1].transAxes,
                               fontsize=8, verticalalignment='top', fontfamily='monospace')
                axes[1, 1].set_title('Sample Data Preview')
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'data_overview.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info("Data overview plots saved successfully")
            
        except Exception as e:
            logger.error(f"Error creating data overview plots: {str(e)}")
            raise
    
    def plot_city_wise_analysis(self, data: pd.DataFrame) -> None:
        """
        Create city-wise analysis visualizations.
        
        Args:
            data (pd.DataFrame): Input dataset with city information
        """
        try:
            # TODO: Implement city-wise analysis plots
            # AQI distribution by city
            # Temporal trends by city
            # Pollutant comparisons by city
            # City-wise statistics summary
            
            logger.info("Creating city-wise analysis visualizations...")
            
            if 'City' not in data.columns or 'AQI' not in data.columns:
                logger.warning("City or AQI column not found in data")
                return
            
            # Create subplots for city analysis
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle('City-wise Air Quality Analysis', fontsize=16, fontweight='bold')
            
            # 1. AQI distribution by city (box plot)
            city_data = data[data['City'].isin(self.target_cities)]
            if not city_data.empty:
                sns.boxplot(data=city_data, x='City', y='AQI', ax=axes[0, 0])
                axes[0, 0].set_title('AQI Distribution by City')
                axes[0, 0].set_xlabel('City')
                axes[0, 0].set_ylabel('AQI')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Average AQI by city (bar plot)
            if not city_data.empty:
                avg_aqi = city_data.groupby('City')['AQI'].mean().sort_values(ascending=False)
                avg_aqi.plot(kind='bar', ax=axes[0, 1], color='skyblue')
                axes[0, 1].set_title('Average AQI by City')
                axes[0, 1].set_xlabel('City')
                axes[0, 1].set_ylabel('Average AQI')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. AQI trends over time by city (if date column exists)
            if 'Date' in city_data.columns and not city_data.empty:
                city_data['Date'] = pd.to_datetime(city_data['Date'])
                for city in self.target_cities:
                    city_subset = city_data[city_data['City'] == city]
                    if not city_subset.empty:
                        monthly_avg = city_subset.groupby(city_subset['Date'].dt.to_period('M'))['AQI'].mean()
                        axes[1, 0].plot(monthly_avg.index.astype(str), monthly_avg.values, 
                                       label=city, marker='o')
                axes[1, 0].set_title('AQI Trends Over Time by City')
                axes[1, 0].set_xlabel('Month')
                axes[1, 0].set_ylabel('Average AQI')
                axes[1, 0].legend()
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. City-wise data availability
            if not city_data.empty:
                city_counts = city_data['City'].value_counts()
                city_counts.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%')
                axes[1, 1].set_title('Data Availability by City')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'city_wise_analysis.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info("City-wise analysis plots saved successfully")
            
        except Exception as e:
            logger.error(f"Error creating city-wise analysis plots: {str(e)}")
            raise
    
    def plot_feature_correlations(self, data: pd.DataFrame) -> None:
        """
        Create feature correlation visualizations.
        
        Args:
            data (pd.DataFrame): Input dataset with features
        """
        try:
            # TODO: Implement feature correlation plots
            # Correlation heatmap
            # Feature correlation with target (AQI)
            # Multi-collinearity analysis
            
            logger.info("Creating feature correlation visualizations...")
            
            # Select numeric columns for correlation analysis
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                logger.warning("Insufficient numeric columns for correlation analysis")
                return
            
            # Create correlation matrix
            corr_matrix = data[numeric_cols].corr()
            
            # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # 1. Full correlation heatmap
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                       square=True, ax=axes[0])
            axes[0].set_title('Feature Correlation Heatmap')
            
            # 2. Correlation with AQI (if available)
            if 'AQI' in corr_matrix.columns:
                aqi_corr = corr_matrix['AQI'].drop('AQI').sort_values(key=abs, ascending=False)
                aqi_corr.plot(kind='barh', ax=axes[1], color='steelblue')
                axes[1].set_title('Feature Correlation with AQI')
                axes[1].set_xlabel('Correlation Coefficient')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_correlations.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info("Feature correlation plots saved successfully")
            
        except Exception as e:
            logger.error(f"Error creating feature correlation plots: {str(e)}")
            raise
    
    def plot_model_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          model_name: str, cities: Optional[np.ndarray] = None) -> None:
        """
        Create model results visualizations.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Name of the model
            cities (Optional[np.ndarray]): City information for test data
        """
        try:
            # TODO: Implement model results plots
            # Prediction vs actual scatter plot
            # Residuals plot
            # Error distribution
            # City-wise prediction accuracy
            
            logger.info(f"Creating model results visualizations for {model_name}...")
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{model_name} - Model Results', fontsize=16, fontweight='bold')
            
            # 1. Prediction vs Actual scatter plot
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='steelblue')
            
            # Add perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            axes[0, 0].set_xlabel('Actual AQI')
            axes[0, 0].set_ylabel('Predicted AQI')
            axes[0, 0].set_title('Prediction vs Actual')
            
            # Calculate R² for the plot
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 2. Residuals plot
            residuals = y_true - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='coral')
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted AQI')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals Plot')
            
            # 3. Error distribution
            axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Error Distribution')
            
            # Add mean and std to the plot
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)
            axes[1, 0].axvline(mean_res, color='r', linestyle='--', label=f'Mean: {mean_res:.2f}')
            axes[1, 0].legend()
            
            # 4. City-wise performance (if city information available)
            if cities is not None:
                city_errors = []
                city_names = []
                
                for city in self.target_cities:
                    city_mask = cities == city
                    if city_mask.sum() > 0:
                        city_mae = np.mean(np.abs(residuals[city_mask]))
                        city_errors.append(city_mae)
                        city_names.append(city)
                
                if city_errors:
                    axes[1, 1].bar(city_names, city_errors, color='orange')
                    axes[1, 1].set_xlabel('City')
                    axes[1, 1].set_ylabel('Mean Absolute Error')
                    axes[1, 1].set_title('City-wise MAE')
                    axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, 'City information\nnot available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('City-wise Performance')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{model_name.lower()}_results.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Model results plots saved for {model_name}")
            
        except Exception as e:
            logger.error(f"Error creating model results plots: {str(e)}")
            raise
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                               model_name: str, top_n: int = 20) -> None:
        """
        Create feature importance visualizations.
        
        Args:
            feature_importance (pd.DataFrame): Feature importance data
            model_name (str): Name of the model
            top_n (int): Number of top features to display
        """
        try:
            # TODO: Implement feature importance plots
            # Horizontal bar plot of top features
            # Feature importance by category
            # Comparison between models (if multiple)
            
            logger.info(f"Creating feature importance visualizations for {model_name}...")
            
            if feature_importance.empty:
                logger.warning("No feature importance data provided")
                return
            
            # Sort by importance and select top N
            top_features = feature_importance.head(top_n)
            
            # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            fig.suptitle(f'{model_name} - Feature Importance', fontsize=16, fontweight='bold')
            
            # 1. Top N features horizontal bar plot
            axes[0].barh(range(len(top_features)), top_features['importance'], 
                        color='steelblue')
            axes[0].set_yticks(range(len(top_features)))
            axes[0].set_yticklabels(top_features['feature'], fontsize=10)
            axes[0].set_xlabel('Importance Score')
            axes[0].set_title(f'Top {top_n} Most Important Features')
            axes[0].invert_yaxis()
            
            # 2. Feature importance by category (if feature names indicate categories)
            feature_categories = {}
            for feature in top_features['feature']:
                if 'lag' in feature.lower():
                    category = 'Lag Features'
                elif 'rolling' in feature.lower() or 'avg' in feature.lower():
                    category = 'Rolling Features'
                elif 'ratio' in feature.lower():
                    category = 'Ratio Features'
                elif feature.lower() in ['year', 'month', 'day', 'season']:
                    category = 'Temporal Features'
                else:
                    category = 'Other Features'
                
                if category not in feature_categories:
                    feature_categories[category] = []
                feature_categories[category].append(top_features[top_features['feature'] == feature]['importance'].iloc[0])
            
            # Create pie chart for categories
            category_importance = {cat: sum(imps) for cat, imps in feature_categories.items()}
            if category_importance:
                axes[1].pie(category_importance.values(), labels=category_importance.keys(), 
                           autopct='%1.1f%%', startangle=90)
                axes[1].set_title('Feature Importance by Category')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{model_name.lower()}_feature_importance.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plots saved for {model_name}")
            
        except Exception as e:
            logger.error(f"Error creating feature importance plots: {str(e)}")
            raise
    
    def create_interactive_plots(self, data: pd.DataFrame) -> None:
        """
        Create interactive plots using Plotly.
        
        Args:
            data (pd.DataFrame): Input dataset
        """
        try:
            # TODO: Implement interactive plots
            # Time series plots with city selection
            # Interactive correlation heatmap
            # 3D scatter plots for multi-dimensional analysis
            
            logger.info("Creating interactive visualizations...")
            
            if data.empty or 'City' not in data.columns or 'AQI' not in data.columns:
                logger.warning("Required columns not found for interactive plots")
                return
            
            # 1. Interactive time series plot
            fig = px.line(data, x='Date', y='AQI', color='City', 
                         title='Interactive AQI Time Series by City',
                         color_discrete_map=self.city_colors)
            
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='AQI',
                hovermode='x unified',
                height=600
            )
            
            # Save as HTML
            fig.write_html(os.path.join(self.output_dir, 'interactive_timeseries.html'))
            
            # 2. Interactive scatter plot matrix
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 3:
                fig = px.scatter_matrix(data[numeric_cols[:5]], 
                                       color=data['City'] if 'City' in data.columns else None,
                                       title='Interactive Scatter Plot Matrix')
                fig.write_html(os.path.join(self.output_dir, 'interactive_scatter_matrix.html'))
            
            logger.info("Interactive plots saved successfully")
            
        except Exception as e:
            logger.error(f"Error creating interactive plots: {str(e)}")
            raise
    
    def generate_visualization_summary(self) -> str:
        """
        Generate a summary of all created visualizations.
        
        Returns:
            str: Visualization summary
        """
        try:
            # TODO: Implement visualization summary
            # List all created plots
            # Provide descriptions and insights
            # Create a visualization catalog
            
            logger.info("Generating visualization summary...")
            
            summary = f"""
# Air Quality Prediction - Visualization Summary

## Generated Visualizations

### 1. Data Overview (`data_overview.png`)
- Dataset shape and basic information
- Missing values analysis
- Data types distribution
- Sample data preview

### 2. City-wise Analysis (`city_wise_analysis.png`)
- AQI distribution by city (box plots)
- Average AQI by city (bar chart)
- AQI trends over time by city (line plot)
- Data availability by city (pie chart)

### 3. Feature Correlations (`feature_correlations.png`)
- Complete correlation heatmap
- Feature correlation with AQI
- Multi-collinearity analysis

### 4. Model Results
- **Baseline Model** (`baseline_results.png`)
- **Primary Model** (`primary_results.png`)
- Prediction vs actual scatter plots
- Residuals analysis
- Error distributions
- City-wise performance

### 5. Feature Importance
- **Baseline Model** (`baseline_feature_importance.png`)
- **Primary Model** (`primary_feature_importance.png`)
- Top features horizontal bar plots
- Feature importance by category

### 6. Interactive Plots
- Interactive time series (`interactive_timeseries.html`)
- Interactive scatter plot matrix (`interactive_scatter_matrix.html`)

## Insights and Recommendations

1. **Data Quality**: [Analysis of data completeness and quality]
2. **City Patterns**: [Key observations about city-wise variations]
3. **Feature Relationships**: [Important correlations and relationships]
4. **Model Performance**: [Visual assessment of model accuracy]
5. **Feature Importance**: [Most influential features for predictions]

---
*Visualization summary generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating visualization summary: {str(e)}")
            raise
    
    def run_full_visualization_pipeline(self) -> None:
        """
        Run the complete visualization pipeline.
        """
        try:
            logger.info("Starting full visualization pipeline...")
            
            # TODO: Implement complete visualization pipeline
            # 1. Load data
            # 2. Create data overview plots
            # 3. Create city-wise analysis plots
            # 4. Create feature correlation plots
            # 5. Create model results plots (if available)
            # 6. Create feature importance plots (if available)
            # 7. Create interactive plots
            # 8. Generate visualization summary
            
            # Placeholder implementation
            logger.info("Full visualization pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in full visualization pipeline: {str(e)}")
            raise


def main():
    """
    Main function to run visualization pipeline.
    """
    try:
        # TODO: Implement main execution logic
        # Set up paths and parameters
        # Run visualization pipeline
        # Generate and save visualization summary
        
        logger.info("Starting visualization pipeline...")
        
        # Initialize visualizer
        # visualizer = AirQualityVisualizer("data/features/engineered_features.csv")
        # visualizer.run_full_visualization_pipeline()
        
        logger.info("Visualization pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main visualization: {str(e)}")
        raise


if __name__ == "__main__":
    main()
