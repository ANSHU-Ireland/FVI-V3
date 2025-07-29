#!/usr/bin/env python3
"""
Quick test to validate the FVI system can initialize
"""

def test_system():
    try:
        # Test imports
        from src.data_processor import DataProcessor
        from src.metrics_calculator import MetricsCalculator
        from src.weight_engine import WeightEngine
        from src.classifier import Classifier
        from src.facts_generator import FactsGenerator
        from src.validator import Validator
        
        print("✅ All imports successful")
        
        # Test config loading
        import yaml
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded")
        
        # Test component initialization
        data_processor = DataProcessor(config)
        print("✅ DataProcessor initialized")
        
        metrics_calculator = MetricsCalculator(config)
        print("✅ MetricsCalculator initialized")
        
        weight_engine = WeightEngine(config)
        print("✅ WeightEngine initialized")
        
        classifier = Classifier(config)
        print("✅ Classifier initialized")
        
        facts_generator = FactsGenerator(config)
        print("✅ FactsGenerator initialized")
        
        validator = Validator(config)
        print("✅ Validator initialized")
        
        print("\n🎉 FVI Phase 1 system ready!")
        print("Run 'python main.py' to start an assessment")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_system()
