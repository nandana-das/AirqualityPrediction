"""
Dependency Installation Script for Air Quality Prediction Project

This script installs all required dependencies for the air quality prediction project.
Can be run independently or imported into notebooks.

Author: MTech AI & Data Science Program
Date: 2024
"""

import subprocess
import sys
import importlib
from typing import List, Dict


def install_package(package: str, pip_name: str = None) -> bool:
    """
    Install package using pip if not already installed.
    
    Args:
        package (str): Package name for import
        pip_name (str): Package name for pip install (if different from import name)
        
    Returns:
        bool: True if package was installed or already available
    """
    try:
        # Try to import the package
        importlib.import_module(package)
        print(f"✓ {package} is already installed")
        return True
    except ImportError:
        try:
            # Install the package
            install_name = pip_name if pip_name else package
            print(f"Installing {install_name}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", install_name
            ])
            print(f"✓ {install_name} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {install_name}: {e}")
            return False


def install_requirements_from_file(requirements_file: str = "requirements.txt") -> bool:
    """
    Install packages from requirements.txt file.
    
    Args:
        requirements_file (str): Path to requirements file
        
    Returns:
        bool: True if all packages installed successfully
    """
    try:
        print(f"Installing packages from {requirements_file}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        print(f"✓ All packages from {requirements_file} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install packages from {requirements_file}: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ Requirements file {requirements_file} not found")
        return False


def install_core_dependencies() -> Dict[str, bool]:
    """
    Install core dependencies for the air quality prediction project.
    
    Returns:
        Dict[str, bool]: Dictionary with installation status for each package
    """
    print("Installing core dependencies for Air Quality Prediction Project...")
    print("=" * 60)
    
    # Define packages with their pip install names (if different)
    packages = {
        # Core Data Science
        "numpy": "numpy>=1.21.0",
        "pandas": "pandas>=1.3.0",
        "scipy": "scipy>=1.7.0",
        
        # Machine Learning
        "sklearn": "scikit-learn>=1.0.0",
        "lightgbm": "lightgbm>=3.3.0",
        "xgboost": "xgboost>=1.5.0",
        "optuna": "optuna>=3.0.0",
        "imblearn": "imbalanced-learn>=0.8.0",
        
        # Visualization
        "matplotlib": "matplotlib>=3.5.0",
        "seaborn": "seaborn>=0.11.0",
        "plotly": "plotly>=5.0.0",
        
        # Jupyter Support
        "jupyter": "jupyter>=1.0.0",
        "IPython": "ipython>=7.0.0",
        "nbformat": "nbformat>=5.1.0",
        
        # Utilities
        "tqdm": "tqdm>=4.62.0",
        "joblib": "joblib>=1.1.0",
        "dateutil": "python-dateutil>=2.8.0",
        
        # Development
        "pytest": "pytest>=6.2.0",
        "black": "black>=21.0.0",
        "flake8": "flake8>=4.0.0"
    }
    
    installation_results = {}
    
    for package, pip_name in packages.items():
        installation_results[package] = install_package(package, pip_name)
    
    print("=" * 60)
    
    # Summary
    successful = sum(installation_results.values())
    total = len(installation_results)
    
    print(f"Installation Summary: {successful}/{total} packages installed successfully")
    
    if successful == total:
        print("✓ All dependencies installed successfully!")
    else:
        failed_packages = [pkg for pkg, status in installation_results.items() if not status]
        print(f"✗ Failed to install: {', '.join(failed_packages)}")
        print("Please install these packages manually or check for conflicts.")
    
    return installation_results


def verify_installation() -> bool:
    """
    Verify that all core packages can be imported successfully.
    
    Returns:
        bool: True if all packages can be imported
    """
    print("\nVerifying installation...")
    print("=" * 40)
    
    core_packages = [
        "numpy", "pandas", "matplotlib", "seaborn", "sklearn", 
        "lightgbm", "optuna", "plotly", "jupyter"
    ]
    
    verification_results = {}
    
    for package in core_packages:
        try:
            importlib.import_module(package)
            verification_results[package] = True
            print(f"✓ {package} - OK")
        except ImportError as e:
            verification_results[package] = False
            print(f"✗ {package} - FAILED: {e}")
    
    successful = sum(verification_results.values())
    total = len(verification_results)
    
    print("=" * 40)
    print(f"Verification Summary: {successful}/{total} packages verified")
    
    return successful == total


def main():
    """
    Main function to install all dependencies.
    """
    print("Air Quality Prediction Project - Dependency Installer")
    print("=" * 60)
    
    # Try to install from requirements.txt first
    print("Attempting to install from requirements.txt...")
    if install_requirements_from_file("requirements.txt"):
        print("Requirements.txt installation successful!")
    else:
        print("Falling back to individual package installation...")
        install_core_dependencies()
    
    # Verify installation
    if verify_installation():
        print("\n🎉 All dependencies are ready! You can now run the project.")
    else:
        print("\n⚠️  Some dependencies may need manual installation.")
    
    print("\nNext steps:")
    print("1. Download the dataset from Kaggle")
    print("2. Place it in the data/raw/ directory")
    print("3. Run the notebooks in sequence: 01 -> 02 -> 03 -> 04 -> 05")


if __name__ == "__main__":
    main()
