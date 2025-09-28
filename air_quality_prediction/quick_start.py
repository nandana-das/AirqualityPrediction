"""
Quick Start Script for Air Quality Prediction Project

This script provides a one-command setup for the entire project.

Author: MTech AI & Data Science Program
Date: 2024
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🌬️  Air Quality Prediction Project - Quick Start     ║
    ║                                                              ║
    ║              MTech AI & Data Science Program                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/features",
        "results/models",
        "results/plots",
        "results/reports"
    ]
    
    print("📁 Setting up project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("✅ All directories created successfully!")


def install_dependencies():
    """Install project dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        # Import and run the installation script
        from install_dependencies import install_core_dependencies, verify_installation
        
        # Install dependencies
        results = install_core_dependencies()
        
        # Verify installation
        if verify_installation():
            print("✅ All dependencies installed and verified!")
            return True
        else:
            print("⚠️  Some dependencies may need manual installation.")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def check_dataset():
    """Check if dataset is available."""
    dataset_path = Path("data/raw")
    
    if not dataset_path.exists():
        print("❌ Data directory not found!")
        return False
    
    # Look for common dataset file names
    possible_files = [
        "air_quality_data.csv",
        "city_day.csv", 
        "station_day.csv",
        "*.csv"
    ]
    
    found_files = []
    for pattern in possible_files:
        found_files.extend(list(dataset_path.glob(pattern)))
    
    if found_files:
        print(f"✅ Dataset found: {[f.name for f in found_files]}")
        return True
    else:
        print("⚠️  No dataset files found in data/raw/")
        print("   Please download the dataset from Kaggle and place it in data/raw/")
        return False


def display_next_steps():
    """Display next steps for the user."""
    print("\n" + "="*60)
    print("🎉 QUICK START COMPLETED!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. 📊 Download the dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india")
    print("   Place the CSV file(s) in the data/raw/ directory")
    
    print("\n2. 🚀 Run the analysis pipeline:")
    print("   Option A - Run notebooks in sequence:")
    print("   - notebooks/01_data_exploration.ipynb")
    print("   - notebooks/02_preprocessing.ipynb") 
    print("   - notebooks/03_feature_engineering.ipynb")
    print("   - notebooks/04_model_training.ipynb")
    print("   - notebooks/05_results_analysis.ipynb")
    
    print("\n   Option B - Run Python scripts:")
    print("   - python src/data_preprocessing.py")
    print("   - python src/feature_engineering.py")
    print("   - python src/models.py")
    print("   - python src/evaluation.py")
    print("   - python src/visualization.py")
    
    print("\n3. 📈 View results in:")
    print("   - results/plots/ (visualizations)")
    print("   - results/models/ (trained models)")
    print("   - results/reports/ (evaluation reports)")
    
    print("\n💡 Tips:")
    print("   - Each notebook includes automatic package installation")
    print("   - Check README.md for detailed instructions")
    print("   - Use jupyter notebook or jupyter lab to run notebooks")


def main():
    """Main quick start function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Continuing with partial setup...")
    
    # Check for dataset
    dataset_available = check_dataset()
    
    # Display next steps
    display_next_steps()
    
    if not dataset_available:
        print("\n⚠️  Remember to download and place the dataset before running analysis!")


if __name__ == "__main__":
    main()
