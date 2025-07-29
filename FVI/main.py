#!/usr/bin/env python3
"""
FVI Phase 1 Main Application
Handles user interaction, orchestrates the pipeline, and produces outputs.
"""

import yaml
import json
import os
from datetime import datetime
from pathlib import Path

# Import FVI modules
from src.data_processor import DataProcessor
from src.metrics_calculator import MetricsCalculator
from src.weight_engine import WeightEngine
from src.classifier import Classifier
from src.facts_generator import FactsGenerator
from src.validator import Validator

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_run_id():
    """Generate unique run identifier"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def collect_user_profile():
    """Collect user profile through interactive chat"""
    print("Welcome to the FVI Assessment System!")
    print("I'll ask you a few questions to personalize your analysis.\n")
    
    # Country validation
    valid_countries = ['IND', 'CHN', 'USA', 'JPN', 'ZAF']
    country_names = {
        'IND': 'India', 'CHN': 'China', 'USA': 'United States', 
        'JPN': 'Japan', 'ZAF': 'South Africa'
    }
    
    print("Available countries for analysis:")
    for code, name in country_names.items():
        print(f"  {code}: {name}")
    
    while True:
        user_iso3 = input("\nWhich country are you interested in? (IND/CHN/USA/JPN/ZAF): ").upper().strip()
        if user_iso3 in valid_countries:
            break
        print("Please select one of the available countries.")
    
    # Depth
    while True:
        depth = input("\nAnalysis depth (quick/standard/deep): ").lower().strip()
        if depth in ['quick', 'standard', 'deep']:
            break
        print("Please choose: quick, standard, or deep")
    
    # Persona
    while True:
        persona = input("\nYour role (investor/regulator/operator): ").lower().strip()
        if persona in ['investor', 'regulator', 'operator']:
            break
        print("Please choose: investor, regulator, or operator")
    
    # Time horizon
    while True:
        time_horizon = input("\nTime horizon (0-5y/5-10y/10-30y): ").lower().strip()
        if time_horizon in ['0-5y', '5-10y', '10-30y']:
            break
        print("Please choose: 0-5y, 5-10y, or 10-30y")
    
    # Risk tolerance
    while True:
        risk_tolerance = input("\nRisk tolerance (low/medium/high): ").lower().strip()
        if risk_tolerance in ['low', 'medium', 'high']:
            break
        print("Please choose: low, medium, or high")
    
    # Capex band
    while True:
        capex_band = input("\nCapital expenditure capacity (none/<50m/50-200m/>200m): ").lower().strip()
        if capex_band in ['none', '<50m', '50-200m', '>200m']:
            break
        print("Please choose: none, <50m, 50-200m, or >200m")
    
    # Policy exposure
    while True:
        policy_exposure = input("\nPolicy exposure level (low/medium/high/unknown): ").lower().strip()
        if policy_exposure in ['low', 'medium', 'high', 'unknown']:
            break
        print("Please choose: low, medium, high, or unknown")
    
    return {
        'user_iso3': user_iso3,
        'depth': depth,
        'persona': persona,
        'time_horizon': time_horizon,
        'risk_tolerance': risk_tolerance,
        'capex_band': capex_band,
        'policy_exposure': policy_exposure
    }

def main():
    """Main application flow"""
    try:
        # Load configuration
        config = load_config()
        run_id = create_run_id()
        
        print("FVI Phase 1 - Fossil Vulnerability Index Assessment")
        print("=" * 55)
        
        # Collect user profile
        user_profile = collect_user_profile()
        
        print(f"\nProcessing analysis for {user_profile['user_iso3']} ({user_profile['depth']} depth)...")
        
        # Initialize components
        data_processor = DataProcessor(config)
        metrics_calculator = MetricsCalculator(config)
        weight_engine = WeightEngine(config)
        classifier = Classifier(config)
        facts_generator = FactsGenerator(config)
        validator = Validator(config)
        
        # Step 1: Process and clean data
        print("Step 1: Processing and cleaning datasets...")
        cleaned_data = data_processor.process_all_datasets()
        reference_year = data_processor.determine_reference_year(cleaned_data)
        
        # Log run metadata
        run_meta = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'reference_year': reference_year,
            'user_profile': user_profile,
            'config_snapshot': config
        }
        
        os.makedirs('artifacts/logs', exist_ok=True)
        with open(f'artifacts/logs/run_meta_{run_id}.json', 'w') as f:
            json.dump(run_meta, f, indent=2)
        
        # Step 2: Calculate metrics for user's country
        print("Step 2: Calculating pillar metrics...")
        country_metrics = metrics_calculator.calculate_all_metrics(
            cleaned_data, user_profile['user_iso3'], reference_year
        )
        
        # Step 3: Apply dynamic weighting
        print("Step 3: Applying dynamic weight engine...")
        weights = weight_engine.calculate_weights(user_profile, country_metrics)
        
        # Step 4: Classification and aggregation
        print("Step 4: Computing classification...")
        classification_result = classifier.classify_country(
            country_metrics, weights, user_profile['user_iso3']
        )
        
        # Step 5: Generate facts and explanation
        print("Step 5: Generating explanation...")
        facts = facts_generator.generate_facts(
            user_profile['user_iso3'], 
            reference_year,
            country_metrics,
            weights,
            classification_result,
            user_profile,
            run_id
        )
        
        explanation = facts_generator.get_gemini_explanation(facts)
        
        # Step 6: Run validation and generate QA report
        print("Step 6: Running validation and QA checks...")
        qa_report = validator.generate_qa_report(
            run_id, reference_year, cleaned_data, country_metrics, weights, user_profile['user_iso3']
        )
        validator.save_qa_report(qa_report, run_id)
        
        # Step 7: Save outputs
        print("Step 7: Saving results...")
        
        # Save scores
        os.makedirs('artifacts/scores', exist_ok=True)
        scores_data = {
            'iso3': user_profile['user_iso3'],
            'reference_year': reference_year,
            'pillar_scores': classification_result['pillar_scores'],
            'composite_score': classification_result['composite_score'],
            'classification': classification_result['label'],
            'weights': weights,
            'run_id': run_id
        }
        
        with open(f'artifacts/scores/{user_profile["user_iso3"]}_scores_{run_id}.json', 'w') as f:
            json.dump(scores_data, f, indent=2)
        
        # Save facts
        os.makedirs('artifacts/facts', exist_ok=True)
        with open(f'artifacts/facts/{user_profile["user_iso3"]}_facts_{run_id}.json', 'w') as f:
            json.dump(facts, f, indent=2)
        
        # Display results
        print("\n" + "=" * 55)
        print("ANALYSIS RESULTS")
        print("=" * 55)
        print(f"Country: {user_profile['user_iso3']}")
        print(f"Reference Year: {reference_year}")
        print(f"Composite Score: {classification_result['composite_score']:.1f}")
        print(f"Classification: {classification_result['label']}")
        
        if classification_result.get('borderline_flag'):
            print("⚠️  BORDERLINE: This country is close to threshold boundaries")
        
        if classification_result.get('guardrails_triggered'):
            print("🚫 GUARDRAILS: Classification adjusted by policy rules")
        
        print(f"\nPillar Scores:")
        for pillar, score in classification_result['pillar_scores'].items():
            weight = weights['pillars'].get(pillar, 0)
            print(f"  {pillar}: {score:.1f} (weight: {weight:.2f})")
        
        print(f"\nExplanation:\n{explanation}")
        
        print(f"\nAll results saved to artifacts/ with run_id: {run_id}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
