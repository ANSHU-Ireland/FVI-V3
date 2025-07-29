#!/usr/bin/env python3
"""
FVI Phase 1 Setup Script
Installs dependencies and tests the system setup.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create required directory structure"""
    directories = [
        "data/raw",
        "data/clean", 
        "data/templates",
        "artifacts/scores",
        "artifacts/facts",
        "artifacts/logs",
        "config",
        "docs",
        "src"
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✅ Directory structure created")

def check_data_sources():
    """Check if data-sources folder is accessible"""
    data_sources_path = Path("../data-sources")
    if data_sources_path.exists():
        print("✅ Data sources folder found")
        
        # List available datasets
        print("\nAvailable datasets:")
        for folder in data_sources_path.iterdir():
            if folder.is_dir():
                print(f"  📁 {folder.name}")
                # List some files in each folder
                files = list(folder.glob("*.csv")) + list(folder.glob("*.xlsx"))
                for file in files[:3]:  # Show first 3 files
                    print(f"    📄 {file.name}")
                if len(files) > 3:
                    print(f"    ... and {len(files) - 3} more files")
        return True
    else:
        print("❌ Data sources folder not found at ../data-sources")
        print("Please ensure the data-sources folder is in the parent directory")
        return False

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    try:
        from src.data_processor import DataProcessor
        from src.metrics_calculator import MetricsCalculator
        from src.weight_engine import WeightEngine
        from src.classifier import Classifier
        from src.facts_generator import FactsGenerator
        from src.validator import Validator
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def check_config():
    """Check configuration file"""
    config_path = Path("config/config.yaml")
    if config_path.exists():
        print("✅ Configuration file found")
        
        # Check if API key is present
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config.get('gemini_api_key'):
                print("✅ Gemini API key configured")
            else:
                print("⚠️  Gemini API key not found in config")
                
        except Exception as e:
            print(f"⚠️  Could not parse config file: {e}")
        
        return True
    else:
        print("❌ Configuration file not found")
        return False

def main():
    """Run setup and checks"""
    print("FVI Phase 1 Setup and Validation")
    print("=" * 40)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    all_good = True
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        all_good = False
    
    # Check data sources
    if not check_data_sources():
        all_good = False
    
    # Check config
    if not check_config():
        all_good = False
    
    # Test imports
    if not test_imports():
        all_good = False
    
    print("\n" + "=" * 40)
    if all_good:
        print("✅ Setup completed successfully!")
        print("You can now run: python main.py")
    else:
        print("❌ Setup completed with issues")
        print("Please resolve the issues above before running the system")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    exit(main())
