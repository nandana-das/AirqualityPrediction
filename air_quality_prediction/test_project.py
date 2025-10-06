#!/usr/bin/env python3
"""
Test Script for Air Quality Prediction Project
This script tests the ACO+DT implementation and makes sample predictions
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def test_data_loading():
    """Test if data can be loaded correctly"""
    print("=" * 50)
    print("TESTING DATA LOADING")
    print("=" * 50)
    
    try:
        # Load original data
        df_original = pd.read_csv('data/raw/aqi.csv')
        print(f"✅ Original data loaded: {df_original.shape}")
        
        # Load transformed data
        df_transformed = pd.read_csv('data/processed/aqi_transformed_for_aco.csv')
        print(f"✅ Transformed data loaded: {df_transformed.shape}")
        
        # Check feature columns
        feature_columns = [col for col in df_transformed.columns 
                          if col not in ['date', 'area', 'aqi_value', 'state', 
                                       'number_of_monitoring_stations', 'prominent_pollutants', 
                                       'air_quality_status', 'unit', 'note']]
        print(f"✅ Features available: {len(feature_columns)}")
        
        return df_transformed, feature_columns
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None

def test_model_loading():
    """Test if trained models can be loaded"""
    print("\n" + "=" * 50)
    print("TESTING MODEL LOADING")
    print("=" * 50)
    
    models = {}
    try:
        # Load ACO-selected features
        with open('results/models/aco_selected_features.pkl', 'rb') as f:
            models['selected_features'] = joblib.load(f)
        print(f"✅ ACO selected features loaded: {len(models['selected_features'])} features")
        
        # Load ACO+DT models
        models['dt_aco_single'] = joblib.load('results/models/dt_aco_single_model.pkl')
        models['dt_aco_ensemble'] = joblib.load('results/models/dt_aco_ensemble_model.pkl')
        print("✅ ACO+DT models loaded")
        
        # Load Random Forest baseline
        models['rf_baseline'] = joblib.load('results/models/rf_baseline_model.pkl')
        print("✅ Random Forest baseline loaded")
        
        return models
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_predictions(models, df_transformed, feature_columns):
    """Test model predictions on sample data"""
    print("\n" + "=" * 50)
    print("TESTING PREDICTIONS")
    print("=" * 50)
    
    try:
        # Prepare data
        X = df_transformed[feature_columns]
        y = df_transformed['aqi_value']
        
        # Take a small sample for testing
        sample_size = min(100, len(X))
        X_sample = X.iloc[:sample_size]
        y_sample = y.iloc[:sample_size]
        
        # Make predictions
        selected_features = models['selected_features']
        X_selected = X_sample[selected_features]
        
        # ACO+DT predictions
        y_pred_single = models['dt_aco_single'].predict(X_selected)
        y_pred_ensemble = models['dt_aco_ensemble'].predict(X_selected)
        
        # Random Forest predictions
        y_pred_rf = models['rf_baseline'].predict(X_sample)
        
        # Calculate metrics
        def calculate_metrics(y_true, y_pred, model_name):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            custom_acc = (1 - mae / np.mean(y_true)) * 100
            
            print(f"\n{model_name} Performance (Sample):")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE:  {mae:.2f}")
            print(f"  R²:   {r2:.4f}")
            print(f"  Custom Accuracy: {custom_acc:.2f}%")
            
            return rmse, mae, r2, custom_acc
        
        # Evaluate models
        rmse_single, _, _, acc_single = calculate_metrics(y_sample, y_pred_single, "ACO+DT (Single)")
        rmse_ensemble, _, _, acc_ensemble = calculate_metrics(y_sample, y_pred_ensemble, "ACO+DT (Ensemble)")
        rmse_rf, _, _, acc_rf = calculate_metrics(y_sample, y_pred_rf, "Random Forest")
        
        print(f"\n✅ Predictions completed successfully!")
        print(f"✅ Best ACO+DT model: {'Ensemble' if rmse_ensemble < rmse_single else 'Single'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction testing failed: {e}")
        return False

def test_results_validation():
    """Validate saved results"""
    print("\n" + "=" * 50)
    print("TESTING RESULTS VALIDATION")
    print("=" * 50)
    
    try:
        # Load results summary
        with open('results/reports/aco_results_summary.json', 'r') as f:
            results = json.load(f)
        
        # Validate key metrics
        ensemble_accuracy = results['model_comparison']['aco_dt_ensemble']['custom_accuracy']
        feature_reduction = results['feature_selection']['feature_reduction_percent']
        policy_features = results['feature_selection']['policy_controllable_selected']
        
        print(f"✅ Results summary loaded")
        print(f"✅ ACO+DT Ensemble Accuracy: {ensemble_accuracy:.2f}%")
        print(f"✅ Feature Reduction: {feature_reduction:.1f}%")
        print(f"✅ Policy-controllable Features: {policy_features}")
        
        # Check if target achieved
        target_accuracy = 94.25
        if ensemble_accuracy >= target_accuracy:
            print(f"✅ Target accuracy achieved: {ensemble_accuracy:.2f}% >= {target_accuracy}%")
        else:
            print(f"⚠️  Target accuracy not achieved: {ensemble_accuracy:.2f}% < {target_accuracy}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Results validation failed: {e}")
        return False

def make_sample_prediction(models, df_transformed, feature_columns):
    """Make a prediction on sample data"""
    print("\n" + "=" * 50)
    print("MAKING SAMPLE PREDICTION")
    print("=" * 50)
    
    try:
        # Create sample input data (simulating Delhi conditions)
        sample_data = {
            'PM10_concentration': 120,
            'PM2.5_concentration': 85,
            'NO2_concentration': 45,
            'SO2_concentration': 25,
            'CO_concentration': 15,
            'O3_concentration': 35,
            'PM10_avg3': 115,
            'PM2.5_lag1': 80,
            'PM2.5_avg3': 82
        }
        
        # Create DataFrame with all features (fill others with defaults)
        sample_df = pd.DataFrame([sample_data])
        
        # Fill missing features with default values
        for feature in feature_columns:
            if feature not in sample_df.columns:
                sample_df[feature] = 0.0
        
        # Make prediction using best model (ensemble)
        selected_features = models['selected_features']
        X_sample = sample_df[selected_features]
        
        prediction = models['dt_aco_ensemble'].predict(X_sample)[0]
        
        print(f"✅ Sample prediction made successfully!")
        print(f"✅ Predicted AQI: {prediction:.1f}")
        
        # Interpret AQI level
        if prediction <= 50:
            level = "Good"
        elif prediction <= 100:
            level = "Satisfactory"
        elif prediction <= 200:
            level = "Moderate"
        elif prediction <= 300:
            level = "Poor"
        elif prediction <= 400:
            level = "Very Poor"
        else:
            level = "Severe"
        
        print(f"✅ Air Quality Level: {level}")
        
        return prediction
        
    except Exception as e:
        print(f"❌ Sample prediction failed: {e}")
        return None

def main():
    """Main testing function"""
    print("🧪 AIR QUALITY PREDICTION PROJECT TESTING")
    print("=" * 60)
    
    # Test 1: Data Loading
    df_transformed, feature_columns = test_data_loading()
    if df_transformed is None:
        print("❌ Cannot proceed without data")
        return
    
    # Test 2: Model Loading
    models = test_model_loading()
    if models is None:
        print("❌ Cannot proceed without models")
        return
    
    # Test 3: Predictions
    prediction_success = test_predictions(models, df_transformed, feature_columns)
    
    # Test 4: Results Validation
    results_success = test_results_validation()
    
    # Test 5: Sample Prediction
    sample_prediction = make_sample_prediction(models, df_transformed, feature_columns)
    
    # Final Summary
    print("\n" + "=" * 60)
    print("TESTING SUMMARY")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 5
    
    if df_transformed is not None:
        tests_passed += 1
        print("✅ Data Loading: PASSED")
    else:
        print("❌ Data Loading: FAILED")
    
    if models is not None:
        tests_passed += 1
        print("✅ Model Loading: PASSED")
    else:
        print("❌ Model Loading: FAILED")
    
    if prediction_success:
        tests_passed += 1
        print("✅ Predictions: PASSED")
    else:
        print("❌ Predictions: FAILED")
    
    if results_success:
        tests_passed += 1
        print("✅ Results Validation: PASSED")
    else:
        print("❌ Results Validation: FAILED")
    
    if sample_prediction is not None:
        tests_passed += 1
        print("✅ Sample Prediction: PASSED")
    else:
        print("❌ Sample Prediction: FAILED")
    
    print(f"\nOverall: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Your project is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
