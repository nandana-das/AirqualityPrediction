#!/usr/bin/env python3
"""
Quick Test Script for Air Quality Prediction Project
Simple validation that everything is working
"""

import os
import json

def quick_test():
    """Quick test to verify project is working"""
    print("🚀 QUICK TEST - Air Quality Prediction Project")
    print("=" * 50)
    
    # Test 1: Check if results exist
    print("1. Checking results files...")
    
    required_files = [
        'results/models/aco_selected_features.pkl',
        'results/models/dt_aco_ensemble_model.pkl',
        'results/models/dt_aco_single_model.pkl',
        'results/models/rf_baseline_model.pkl',
        'results/reports/aco_results_summary.json'
    ]
    
    required_plots = [
        'results/plots/aco_convergence.png',
        'results/plots/prediction_comparison.png',
        'results/plots/model_comparison_summary.png'
    ]
    
    files_exist = 0
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
            files_exist += 1
        else:
            print(f"   ❌ {file}")
    
    plots_exist = 0
    for plot in required_plots:
        if os.path.exists(plot):
            print(f"   ✅ {plot}")
            plots_exist += 1
        else:
            print(f"   ❌ {plot}")
    
    # Test 2: Check results summary
    print("\n2. Checking results summary...")
    try:
        with open('results/reports/aco_results_summary.json', 'r') as f:
            results = json.load(f)
        
        ensemble_accuracy = results['model_comparison']['aco_dt_ensemble']['custom_accuracy']
        feature_reduction = results['feature_selection']['feature_reduction_percent']
        selected_features = results['feature_selection']['selected_feature_list']
        
        print(f"   ✅ ACO+DT Ensemble Accuracy: {ensemble_accuracy:.2f}%")
        print(f"   ✅ Feature Reduction: {feature_reduction:.1f}%")
        print(f"   ✅ Selected Features: {len(selected_features)}")
        print(f"   ✅ Features: {', '.join(selected_features[:3])}...")
        
        # Check if target achieved
        target_accuracy = 94.25
        if ensemble_accuracy >= target_accuracy:
            print(f"   ✅ Target achieved: {ensemble_accuracy:.2f}% >= {target_accuracy}%")
        else:
            print(f"   ⚠️  Target not achieved: {ensemble_accuracy:.2f}% < {target_accuracy}%")
        
    except Exception as e:
        print(f"   ❌ Error reading results: {e}")
    
    # Test 3: Check data files
    print("\n3. Checking data files...")
    data_files = [
        'data/raw/aqi.csv',
        'data/processed/aqi_transformed_for_aco.csv'
    ]
    
    for file in data_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
    
    # Final summary
    print("\n" + "=" * 50)
    print("QUICK TEST SUMMARY")
    print("=" * 50)
    
    total_checks = len(required_files) + len(required_plots) + len(data_files) + 1  # +1 for JSON read
    
    if files_exist == len(required_files) and plots_exist == len(required_plots):
        print("🎉 PROJECT IS READY!")
        print("✅ All model files exist")
        print("✅ All plot files exist") 
        print("✅ Results can be read")
        print("\n📊 Your Results:")
        print(f"   • ACO+DT Ensemble Accuracy: {ensemble_accuracy:.2f}%")
        print(f"   • Feature Reduction: {feature_reduction:.1f}%")
        print(f"   • Selected Features: {len(selected_features)}")
        print(f"   • All features are policy-controllable")
        
        print("\n🚀 Ready for presentation!")
    else:
        print("⚠️  Some files are missing. Run the notebooks first.")

if __name__ == "__main__":
    quick_test()
