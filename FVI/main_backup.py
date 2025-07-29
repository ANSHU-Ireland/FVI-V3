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
        
        # Step 6: Save outputs
        print("Step 6: Saving results...")
        
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
            
            # Validate target countries
            if set(config['countries']) != self.TARGET_COUNTRIES:
                logger.warning(f"Config countries {config['countries']} != target {self.TARGET_COUNTRIES}")
                
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _generate_run_id(self) -> str:
        """Generate unique run identifier"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_complete_pipeline(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete FVI pipeline for a user
        
        Args:
            user_profile: Dictionary containing user preferences and target country
            
        Returns:
            Dictionary containing results and artifacts paths
        """
        logger.info(f"Starting complete pipeline for user profile: {user_profile}")
        
        # Validate user profile
        self._validate_user_profile(user_profile)
        user_iso3 = user_profile['user_iso3']
        
        try:
            # Step 1: Process and clean data
            logger.info("Step 1: Processing and cleaning data")
            clean_data = self.data_processor.process_all_sources()
            
            # Step 2: Validate data quality
            logger.info("Step 2: Validating data quality")
            validation_results = self.validator.validate_all(clean_data)
            
            # Step 3: Calculate dynamic weights
            logger.info("Step 3: Calculating dynamic weights")
            weights = self.weight_engine.calculate_weights(user_profile, clean_data)
            
            # Step 4: Calculate pillar scores
            logger.info("Step 4: Calculating pillar scores")
            scores = self.score_calculator.calculate_all_scores(clean_data, weights)
            
            # Step 5: Classify country
            logger.info("Step 5: Classifying country")
            classification = self.classifier.classify_country(scores[user_iso3], user_iso3)
            
            # Step 6: Generate facts JSON
            logger.info("Step 6: Generating facts JSON")
            facts = self._generate_facts_json(user_profile, clean_data, weights, scores, classification)
            
            # Step 7: Generate Gemini explanation
            logger.info("Step 7: Generating explanation")
            explanation = self.gemini_explainer.generate_explanation(facts)
            
            # Step 8: Save artifacts
            logger.info("Step 8: Saving artifacts")
            artifact_paths = self._save_artifacts(user_iso3, scores, facts, weights, validation_results)
            
            # Compile results
            results = {
                'user_iso3': user_iso3,
                'run_id': self.run_id,
                'classification': classification,
                'explanation': explanation,
                'scores': scores[user_iso3],
                'weights': weights,
                'validation': validation_results,
                'artifacts': artifact_paths,
                'facts': facts
            }
            
            logger.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _validate_user_profile(self, user_profile: Dict[str, Any]) -> None:
        """Validate user profile contains required fields with valid values"""
        required_fields = {
            'user_iso3': self.TARGET_COUNTRIES,
            'depth': {'quick', 'standard', 'deep'},
            'persona': {'investor', 'regulator', 'operator'},
            'time_horizon': {'0-5y', '5-10y', '10-30y'},
            'risk_tolerance': {'low', 'medium', 'high'},
            'capex_band': {'none', '<50m', '50-200m', '>200m'},
            'policy_exposure': {'low', 'medium', 'high', 'unknown'}
        }
        
        for field, valid_values in required_fields.items():
            if field not in user_profile:
                raise ValueError(f"Missing required field: {field}")
            
            if user_profile[field] not in valid_values:
                raise ValueError(f"Invalid value for {field}: {user_profile[field]}. Valid: {valid_values}")
    
    def _generate_facts_json(self, user_profile: Dict[str, Any], clean_data: Dict[str, Any], 
                           weights: Dict[str, Any], scores: Dict[str, Any], 
                           classification: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the facts JSON object for LLM consumption"""
        user_iso3 = user_profile['user_iso3']
        country_scores = scores[user_iso3]
        
        # Get country name
        country_name = self._get_country_name(user_iso3)
        
        # Extract pillar scores and submetrics
        pillar_scores = {}
        submetrics = {}
        data_gaps = []
        
        for pillar in ['Emissions', 'Necessity', 'Ecological', 'ArtificialSupport']:
            if pillar in country_scores['pillars']:
                pillar_scores[pillar] = country_scores['pillars'][pillar]['score']
                submetrics[pillar] = country_scores['pillars'][pillar]['submetrics']
            else:
                data_gaps.append(f"Missing {pillar} pillar data")
        
        # Calculate top drivers
        composite_score = country_scores['composite']
        top_drivers = []
        for pillar, data in country_scores['pillars'].items():
            contribution = weights['pillars'][pillar] * data['score']
            top_drivers.append({
                'pillar': pillar,
                'contribution_pct': (contribution / composite_score) * 100
            })
        top_drivers = sorted(top_drivers, key=lambda x: x['contribution_pct'], reverse=True)[:3]
        
        # Quality summary
        quality_scores = []
        proxies_used = []
        for pillar_data in country_scores['pillars'].values():
            for submetric in pillar_data['submetrics']:
                quality_scores.append(submetric['data_quality'])
                if submetric.get('is_proxy', False):
                    proxies_used.append(submetric['name'])
        
        quality_summary = {
            'min_quality': min(quality_scores) if quality_scores else 0,
            'mean_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'proxies_used': proxies_used
        }
        
        # Build facts object
        facts = {
            'entity': {
                'iso3': user_iso3,
                'name': country_name
            },
            'reference_year': clean_data.get('reference_year'),
            'pillar_scores': pillar_scores,
            'submetrics': submetrics,
            'weights': weights,
            'composite': composite_score,
            'label': classification['label'],
            'borderline_flag': classification.get('borderline', False),
            'guardrails_triggered': classification.get('guardrails_triggered', []),
            'top_drivers': top_drivers,
            'data_gaps': data_gaps,
            'user_profile': user_profile,
            'assumptions': {
                'thresholds': self.config['pillar_thresholds'],
                'alpha': self.config['weight_engine']['alpha_by_depth'][user_profile['depth']],
                'floors': self.config['weight_engine']['floor'],
                'winsor_limits': self.config['winsor_limits_pct']
            },
            'quality_summary': quality_summary
        }
        
        return facts
    
    def _get_country_name(self, iso3: str) -> str:
        """Get full country name from ISO3 code"""
        names = {
            'IND': 'India',
            'CHN': 'China',
            'USA': 'United States',
            'JPN': 'Japan',
            'ZAF': 'South Africa'
        }
        return names.get(iso3, iso3)
    
    def _save_artifacts(self, user_iso3: str, scores: Dict[str, Any], facts: Dict[str, Any],
                       weights: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, str]:
        """Save all artifacts and return file paths"""
        artifacts_dir = Path("artifacts")
        
        # Ensure directories exist
        (artifacts_dir / "scores").mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "facts").mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Save scores (JSON and CSV)
        scores_json_path = artifacts_dir / "scores" / f"{user_iso3}_scores.json"
        with open(scores_json_path, 'w') as f:
            json.dump(scores[user_iso3], f, indent=2)
        paths['scores_json'] = str(scores_json_path)
        
        # Save facts JSON
        facts_json_path = artifacts_dir / "facts" / f"{user_iso3}_facts.json"
        with open(facts_json_path, 'w') as f:
            json.dump(facts, f, indent=2)
        paths['facts_json'] = str(facts_json_path)
        
        # Save run metadata
        run_meta_path = artifacts_dir / "logs" / "run_meta.json"
        run_meta = {
            'run_id': self.run_id,
            'reference_year': facts.get('reference_year'),
            'target_countries': list(self.TARGET_COUNTRIES),
            'timestamp': self.run_id
        }
        with open(run_meta_path, 'w') as f:
            json.dump(run_meta, f, indent=2)
        paths['run_meta'] = str(run_meta_path)
        
        # Save weights log
        weights_log_path = artifacts_dir / "logs" / f"weights_{user_iso3}_{self.run_id}.json"
        with open(weights_log_path, 'w') as f:
            json.dump(weights, f, indent=2)
        paths['weights_log'] = str(weights_log_path)
        
        # Save validation QA report
        qa_report_path = artifacts_dir / "logs" / f"qa_{self.run_id}.md"
        self._write_qa_report(qa_report_path, validation)
        paths['qa_report'] = str(qa_report_path)
        
        return paths
    
    def _write_qa_report(self, path: Path, validation: Dict[str, Any]) -> None:
        """Write validation QA report in markdown format"""
        with open(path, 'w') as f:
            f.write(f"# FVI Phase 1 QA Report - Run {self.run_id}\n\n")
            
            f.write("## Data Validation Summary\n\n")
            
            # Schema validation
            f.write("### Schema Validation\n")
            for check, result in validation.get('schema_checks', {}).items():
                status = "✅ PASS" if result['passed'] else "❌ FAIL"
                f.write(f"- {check}: {status}\n")
                if not result['passed']:
                    f.write(f"  - Issues: {result.get('issues', [])}\n")
            f.write("\n")
            
            # Key validation
            f.write("### Key Validation\n")
            for check, result in validation.get('key_checks', {}).items():
                status = "✅ PASS" if result['passed'] else "❌ FAIL"
                f.write(f"- {check}: {status}\n")
                if not result['passed']:
                    f.write(f"  - Issues: {result.get('issues', [])}\n")
            f.write("\n")
            
            # Cross-checks
            f.write("### Cross-Checks\n")
            for check, result in validation.get('cross_checks', {}).items():
                status = "✅ PASS" if result['passed'] else "⚠️ FLAG"
                f.write(f"- {check}: {status}\n")
                if not result['passed']:
                    f.write(f"  - Details: {result.get('details', '')}\n")
            f.write("\n")
            
            # Data quality
            f.write("### Data Quality Summary\n")
            quality = validation.get('quality_summary', {})
            f.write(f"- Minimum Quality Score: {quality.get('min_quality', 'N/A')}\n")
            f.write(f"- Average Quality Score: {quality.get('avg_quality', 'N/A')}\n")
            f.write(f"- Proxies Used: {len(quality.get('proxies', []))}\n")
            f.write(f"- Missing Data Points: {quality.get('missing_count', 'N/A')}\n")

def main():
    """Main entry point for interactive mode"""
    engine = FVIEngine()
    engine.chat_interface.start_interactive_session()

if __name__ == "__main__":
    main()
