# Air Quality Prediction Project

An end-to-end machine learning project for predicting next-day Air Quality Index (AQI) for major Indian cities: Delhi, Bangalore, Kolkata, Hyderabad, Chennai, and Visakhapatnam.

## Project Overview

This project aims to outperform the baseline paper "Optimized machine learning model for air quality index prediction in major cities in India" (2024, Scientific Reports, Nature; 94.25% accuracy) using advanced feature engineering and model optimization techniques.

## Dataset

- **Source**: Kaggle "Air Quality Data in India (2015–2020)"
- **Features**: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, AQI, Date, City
- **Target**: Next-day AQI prediction

## Project Structure

```
air_quality_prediction/
├── data/raw/                    # Raw dataset files
├── data/processed/              # Cleaned and processed data
├── data/features/               # Feature-engineered datasets
├── src/                         # Source code modules
│   ├── data_preprocessing.py    # Data cleaning and preprocessing
│   ├── feature_engineering.py  # Feature creation and selection
│   ├── models.py               # ML model implementations
│   ├── evaluation.py           # Model evaluation utilities
│   └── visualization.py        # Plotting and visualization
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_results_analysis.ipynb
├── results/                     # Output files
│   ├── models/                 # Trained model files
│   ├── plots/                  # Generated visualizations
│   └── reports/                # Analysis reports
├── requirements.txt             # Python dependencies
├── environment.yml             # Conda environment specification
└── README.md                   # This file
```

## Setup Instructions

### Option 1: Using Conda/Mamba (Recommended)

1. **Create and activate the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate air_quality_prediction
   ```

2. **Download the dataset**:
   - Go to [Kaggle Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
   - Download the dataset and extract it
   - Place the CSV file(s) in the `data/raw/` directory

3. **Verify the setup**:
   ```bash
   python -c "import pandas as pd; import lightgbm; print('Setup successful!')"
   ```

### Option 2: Using pip

1. **Create a virtual environment**:
   ```bash
   python -m venv air_quality_env
   source air_quality_env/bin/activate  # On Windows: air_quality_env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and place the dataset** as described in Option 1, step 2.

### Option 3: Automatic Installation Script

1. **Run the installation script**:
   ```bash
   python install_dependencies.py
   ```

2. **This script will**:
   - Check for existing installations
   - Install missing packages automatically
   - Verify all installations
   - Provide detailed feedback

### Option 4: Quick Start (Recommended for New Users)

1. **Run the quick start script**:
   ```bash
   python quick_start.py
   ```

2. **This script will automatically**:
   - Check Python version compatibility
   - Create all necessary directories
   - Install all dependencies
   - Verify installations
   - Check for dataset availability
   - Provide next steps guidance

### Option 5: Notebook-based Installation

Each Jupyter notebook includes automatic package installation in the first cell. Simply run the notebooks and packages will be installed as needed.

**Note**: All notebooks include pip install commands for their required packages, so you can run them independently without prior setup.

## Usage

### Running the Complete Pipeline

1. **Data Exploration**:
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

2. **Data Preprocessing**:
   ```bash
   python src/data_preprocessing.py
   jupyter notebook notebooks/02_preprocessing.ipynb
   ```

3. **Feature Engineering**:
   ```bash
   python src/feature_engineering.py
   jupyter notebook notebooks/03_feature_engineering.ipynb
   ```

4. **Model Training**:
   ```bash
   python src/models.py
   jupyter notebook notebooks/04_model_training.ipynb
   ```

5. **Results Analysis**:
   ```bash
   python src/evaluation.py
   python src/visualization.py
   jupyter notebook notebooks/05_results_analysis.ipynb
   ```

### Running Individual Scripts

Each script in the `src/` directory can be run independently:

```bash
# Data preprocessing
python src/data_preprocessing.py

# Feature engineering
python src/feature_engineering.py

# Model training and evaluation
python src/models.py

# Generate visualizations
python src/visualization.py
```

## Key Features

- **Advanced Feature Engineering**: Temporal features, lag features, rolling averages, and pollutant ratios
- **Model Optimization**: LightGBM with hyperparameter tuning using Optuna
- **Time-based Validation**: Proper time-series cross-validation
- **Comprehensive Evaluation**: Multiple metrics including custom accuracy measure
- **City-wise Analysis**: Individual performance analysis for each city
- **Reproducible Results**: All intermediate data saved as CSV files

## Performance Metrics

- **Primary Metric**: Custom accuracy = [1 - MAE/mean(actual)] × 100
- **Secondary Metrics**: RMSE, MAE, R²
- **Target**: Exceed 94.25% accuracy from baseline paper

## Model Architecture

1. **Baseline**: Random Forest (n_estimators=100, max_depth=10)
2. **Primary**: LightGBM with hyperparameter optimization
3. **Feature Engineering**: 50+ engineered features including temporal, lag, and interaction features
4. **Data Balancing**: SMOTE for handling imbalanced AQI categories

## Output Files

- **Models**: Saved in `results/models/`
- **Plots**: Generated in `results/plots/`
- **Reports**: Technical analysis in `results/reports/`
- **Processed Data**: All intermediate datasets in `data/processed/` and `data/features/`

## Contributing

1. Follow PEP-8 coding standards
2. Add docstrings to all functions and classes
3. Include type hints for function arguments
4. Test all changes before committing

## License

This project is for academic purposes as part of MTech AI & Data Science program.

## Contact

For questions or issues, please refer to the project documentation or contact the development team.
