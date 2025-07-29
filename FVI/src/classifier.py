"""
FVI Classification and Aggregation Module
Handles composite score calculation, classification, and guardrails.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

class FVIClassifier:
    def __init__(self, config: Dict):
        """Initialize the classifier."""
        self.config = config
        self.thresholds = config['pillar_thresholds']
        self.guardrails = config['guardrails']
        self.logger = logging.getLogger(__name__)
        
    def calculate_composite_scores(self, pillar_scores: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """
        Calculate composite scores using dynamic weights.
        
        Args:
            pillar_scores: DataFrame with pillar scores per country
            weights: Dictionary of pillar weights
            
        Returns:
            DataFrame with composite scores per country
        """
        
        self.logger.info("Calculating composite scores...")
        
        # Pivot pillar scores for easier calculation
        scores_pivot = pillar_scores.pivot(index='iso3', columns='pillar', values='pillar_score')
        
        composite_scores = []
        
        for iso3 in scores_pivot.index:
            country_scores = scores_pivot.loc[iso3]
            
            # Calculate weighted composite
            composite = 0.0
            total_weight = 0.0
            available_pillars = []
            
            for pillar, weight in weights.items():
                if pillar in country_scores and not pd.isna(country_scores[pillar]):
                    composite += weight * country_scores[pillar]
                    total_weight += weight
                    available_pillars.append(pillar)
            
            # Normalize by available weights
            if total_weight > 0:
                composite = composite / total_weight
            else:
                composite = np.nan
            
            # Get individual pillar scores for classification logic
            pillar_dict = {pillar: country_scores.get(pillar, np.nan) for pillar in weights.keys()}
            
            composite_scores.append({
                'iso3': iso3,
                'composite': composite,
                'available_pillars': available_pillars,
                'total_weight': total_weight,
                **pillar_dict
            })
        
        return pd.DataFrame(composite_scores)
    
    def classify_countries(self, composite_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify countries based on composite scores and guardrails.
        
        Classification rules:
        - Sustainable if composite ≤ 35 AND Emissions ≤ 30 AND (Necessity ≥ 50 OR Ecological ≤ median_of_five)
        - Critical Transition if 35 < composite ≤ 65 OR (Necessity ≥ 60 and Emissions ≥ 45)
        - Decommission if composite > 65
        
        Guardrails:
        - If ArtificialSupport ≥ 70 AND Emissions ≥ 60 → never Sustainable
        - If Ecological ≥ 60 AND Necessity ≤ 30 → bias toward Decommission
        """
        
        self.logger.info("Classifying countries...")
        
        result_df = composite_df.copy()
        
        # Calculate median ecological score for the five countries
        ecological_scores = result_df['Ecological'].dropna()
        eco_median = ecological_scores.median() if not ecological_scores.empty else 50
        
        classifications = []
        
        for _, row in result_df.iterrows():
            iso3 = row['iso3']
            composite = row['composite']
            
            # Get pillar scores (handle NaN)
            emissions = row.get('Emissions', np.nan)
            necessity = row.get('Necessity', np.nan)
            ecological = row.get('Ecological', np.nan)
            artificial_support = row.get('ArtificialSupport', np.nan)
            
            # Initialize classification variables
            classification = None
            borderline_flag = False
            guardrails_triggered = []
            
            # Check guardrails first
            guardrail_sustainable_block = False
            guardrail_decom_bias = False
            
            # Guardrail 1: AS ≥ 70 AND Emissions ≥ 60 → never Sustainable
            if (not pd.isna(artificial_support) and artificial_support >= self.guardrails['as_emissions_block']['as_min'] and
                not pd.isna(emissions) and emissions >= self.guardrails['as_emissions_block']['emissions_min']):
                guardrail_sustainable_block = True
                guardrails_triggered.append('as_emissions_block')
            
            # Guardrail 2: Ecological ≥ 60 AND Necessity ≤ 30 → bias toward Decommission
            if (not pd.isna(ecological) and ecological >= self.guardrails['eco_decom_bias']['eco_min'] and
                not pd.isna(necessity) and necessity <= self.guardrails['eco_decom_bias']['nec_max']):
                guardrail_decom_bias = True
                guardrails_triggered.append('eco_decom_bias')
            
            # Classification logic
            if pd.isna(composite):
                classification = 'Insufficient Data'
            else:
                # Check for borderline (within ±2 points of thresholds)
                if abs(composite - self.thresholds['sustainable_max']) <= 2 or abs(composite - self.thresholds['transition_max']) <= 2:
                    borderline_flag = True
                
                # Sustainable criteria
                sustainable_composite = composite <= self.thresholds['sustainable_max']
                sustainable_emissions = pd.isna(emissions) or emissions <= 30
                sustainable_necessity_eco = (
                    (not pd.isna(necessity) and necessity >= 50) or
                    (not pd.isna(ecological) and ecological <= eco_median)
                )
                
                if (sustainable_composite and sustainable_emissions and sustainable_necessity_eco and 
                    not guardrail_sustainable_block):
                    classification = 'Sustainable'
                
                # Critical Transition criteria
                elif (self.thresholds['sustainable_max'] < composite <= self.thresholds['transition_max'] or
                      (not pd.isna(necessity) and necessity >= 60 and not pd.isna(emissions) and emissions >= 45)):
                    classification = 'Critical Transition'
                
                # Decommission criteria
                elif composite > self.thresholds['transition_max']:
                    classification = 'Decommission'
                    
                    # Apply decommission bias if guardrail triggered
                    if guardrail_decom_bias:
                        classification = 'Decommission → Repurpose'
                
                # Default to Critical Transition if no clear classification
                else:
                    classification = 'Critical Transition'
            
            classifications.append({
                'iso3': iso3,
                'composite': composite,
                'classification': classification,
                'borderline_flag': borderline_flag,
                'guardrails_triggered': guardrails_triggered,
                'eco_median_reference': eco_median
            })
        
        classification_df = pd.DataFrame(classifications)
        
        # Merge back with original data
        result_df = result_df.merge(classification_df[['iso3', 'classification', 'borderline_flag', 'guardrails_triggered']], 
                                   on='iso3', how='left')
        
        self.logger.info(f"Classification completed: {len(result_df)} countries classified")
        return result_df
    
    def calculate_top_drivers(self, pillar_scores: pd.DataFrame, weights: Dict[str, float], 
                            iso3: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """Calculate top contributing pillars to composite score."""
        
        country_scores = pillar_scores[pillar_scores['iso3'] == iso3]
        if country_scores.empty:
            return []
        
        drivers = []
        
        for _, row in country_scores.iterrows():
            pillar = row['pillar']
            pillar_score = row['pillar_score']
            weight = weights.get(pillar, 0)
            
            contribution = weight * pillar_score
            contribution_pct = contribution  # This will be normalized later
            
            drivers.append({
                'pillar': pillar,
                'pillar_score': pillar_score,
                'weight': weight,
                'contribution': contribution,
                'contribution_pct': contribution_pct
            })
        
        # Sort by contribution and normalize percentages
        drivers.sort(key=lambda x: x['contribution'], reverse=True)
        total_contribution = sum(d['contribution'] for d in drivers)
        
        if total_contribution > 0:
            for driver in drivers:
                driver['contribution_pct'] = (driver['contribution'] / total_contribution) * 100
        
        return drivers[:top_n]
    
    def run_validation_checks(self, pillar_scores: pd.DataFrame, composite_df: pd.DataFrame) -> Dict[str, Any]:
        """Run validation and quality checks."""
        
        self.logger.info("Running validation checks...")
        
        validation_results = {
            'schema_gate': True,
            'key_gate': True,
            'type_range_gate': True,
            'cross_checks': [],
            'quality_summary': {},
            'issues': []
        }
        
        # Schema gate: Check required columns
        required_pillar_cols = ['iso3', 'pillar', 'pillar_score']
        required_composite_cols = ['iso3', 'composite', 'classification']
        
        if not all(col in pillar_scores.columns for col in required_pillar_cols):
            validation_results['schema_gate'] = False
            validation_results['issues'].append("Missing required columns in pillar_scores")
        
        if not all(col in composite_df.columns for col in required_composite_cols):
            validation_results['schema_gate'] = False
            validation_results['issues'].append("Missing required columns in composite_df")
        
        # Key gate: Check all rows restricted to target countries
        target_countries = set(self.config['countries'])
        pillar_countries = set(pillar_scores['iso3'].unique())
        composite_countries = set(composite_df['iso3'].unique())
        
        if not pillar_countries.issubset(target_countries):
            validation_results['key_gate'] = False
            extra_countries = pillar_countries - target_countries
            validation_results['issues'].append(f"Extra countries in pillar_scores: {extra_countries}")
        
        if not composite_countries.issubset(target_countries):
            validation_results['key_gate'] = False
            extra_countries = composite_countries - target_countries
            validation_results['issues'].append(f"Extra countries in composite_df: {extra_countries}")
        
        # Type/range gate: Check value ranges
        if 'pillar_score' in pillar_scores.columns:
            invalid_scores = pillar_scores[(pillar_scores['pillar_score'] < 0) | (pillar_scores['pillar_score'] > 100)]
            if not invalid_scores.empty:
                validation_results['type_range_gate'] = False
                validation_results['issues'].append(f"Invalid pillar scores (not 0-100): {len(invalid_scores)} records")
        
        if 'composite' in composite_df.columns:
            invalid_composite = composite_df[(composite_df['composite'] < 0) | (composite_df['composite'] > 100)]
            if not invalid_composite.empty:
                validation_results['type_range_gate'] = False
                validation_results['issues'].append(f"Invalid composite scores (not 0-100): {len(invalid_composite)} records")
        
        # Cross-checks
        self._run_cross_checks(pillar_scores, composite_df, validation_results)
        
        # Quality summary
        if 'mean_data_quality' in pillar_scores.columns:
            quality_stats = pillar_scores['mean_data_quality'].describe()
            validation_results['quality_summary'] = {
                'min_quality': quality_stats.get('min', 0),
                'mean_quality': quality_stats.get('mean', 0),
                'max_quality': quality_stats.get('max', 1)
            }
        
        return validation_results
    
    def _run_cross_checks(self, pillar_scores: pd.DataFrame, composite_df: pd.DataFrame, 
                         validation_results: Dict[str, Any]):
        """Run cross-validation checks."""
        
        cross_checks = []
        
        # Check 1: If NEC_CoalShareElec < 5% and NEC_ConsumptionLock top-2 among five → flag inconsistency
        necessity_data = pillar_scores[pillar_scores['pillar'] == 'Necessity']
        if not necessity_data.empty:
            # This would require access to sub-metric data for detailed checking
            # For now, just check if Necessity scores are consistent with expectations
            pass
        
        # Check 2: If exit_year ≤ current year but ES_Absolute high → policy mismatch
        emissions_data = pillar_scores[pillar_scores['pillar'] == 'Emissions']
        support_data = pillar_scores[pillar_scores['pillar'] == 'ArtificialSupport']
        
        if not emissions_data.empty and not support_data.empty:
            high_emissions_countries = emissions_data[emissions_data['pillar_score'] > 70]['iso3'].tolist()
            
            # This would require access to exit_year data for detailed checking
            if high_emissions_countries:
                cross_checks.append({
                    'check': 'policy_mismatch',
                    'description': 'High emissions countries identified for exit year validation',
                    'countries': high_emissions_countries
                })
        
        validation_results['cross_checks'] = cross_checks
    
    def generate_qa_summary(self, validation_results: Dict[str, Any], run_id: str) -> str:
        """Generate QA summary report."""
        
        qa_summary = f"# FVI Quality Assurance Report\n"
        qa_summary += f"Run ID: {run_id}\n"
        qa_summary += f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        qa_summary += "## Validation Gates\n"
        qa_summary += f"- Schema Gate: {'✓ PASS' if validation_results['schema_gate'] else '✗ FAIL'}\n"
        qa_summary += f"- Key Gate: {'✓ PASS' if validation_results['key_gate'] else '✗ FAIL'}\n"
        qa_summary += f"- Type/Range Gate: {'✓ PASS' if validation_results['type_range_gate'] else '✗ FAIL'}\n\n"
        
        if validation_results['issues']:
            qa_summary += "## Issues Found\n"
            for issue in validation_results['issues']:
                qa_summary += f"- {issue}\n"
            qa_summary += "\n"
        
        if validation_results['cross_checks']:
            qa_summary += "## Cross-Checks\n"
            for check in validation_results['cross_checks']:
                qa_summary += f"- {check['check']}: {check['description']}\n"
            qa_summary += "\n"
        
        if validation_results['quality_summary']:
            qa_summary += "## Quality Summary\n"
            quality = validation_results['quality_summary']
            qa_summary += f"- Minimum Quality: {quality.get('min_quality', 0):.3f}\n"
            qa_summary += f"- Mean Quality: {quality.get('mean_quality', 0):.3f}\n"
            qa_summary += f"- Maximum Quality: {quality.get('max_quality', 1):.3f}\n\n"
        
        qa_summary += "## Recommendations\n"
        if not validation_results['schema_gate']:
            qa_summary += "- Fix schema issues before proceeding\n"
        if not validation_results['key_gate']:
            qa_summary += "- Remove data for non-target countries\n"
        if not validation_results['type_range_gate']:
            qa_summary += "- Investigate out-of-range values\n"
        
        if all([validation_results['schema_gate'], validation_results['key_gate'], validation_results['type_range_gate']]):
            qa_summary += "- All validation gates passed ✓\n"
        
        return qa_summary
