"""
FVI Phase 1 Validation Module
Implements all validation gates and QA checks as specified.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass

class Validator:
    """Validation and QA system for FVI Phase 1"""
    
    TARGET_COUNTRIES = {'IND', 'CHN', 'USA', 'JPN', 'ZAF'}
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_log = []
        
    def validate_schema(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Schema gate: required columns present for each input"""
        required_schemas = {
            'fact_emissions': ['iso3', 'year', 'emissions_tco2e', 'data_quality', 'dataset_ids'],
            'fact_necessity': ['iso3', 'year', 'coal_share_electricity_pct', 'coal_consumption_twh', 'data_quality', 'dataset_ids'],
            'fact_ecological': ['iso3', 'mining_area_km2', 'mining_sites_count', 'data_quality', 'dataset_ids'],
            'fact_support': ['iso3', 'year', 'data_quality', 'dataset_ids']
        }
        
        for table_name, required_cols in required_schemas.items():
            if table_name in data:
                df = data[table_name]
                missing_cols = set(required_cols) - set(df.columns)
                if missing_cols:
                    self.validation_log.append(f"❌ Schema: {table_name} missing columns: {missing_cols}")
                    return False
                else:
                    self.validation_log.append(f"✅ Schema: {table_name} has all required columns")
        
        return True
    
    def validate_country_filter(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Key gate: all rows restricted to target countries"""
        for table_name, df in data.items():
            if 'iso3' in df.columns:
                invalid_countries = set(df['iso3'].unique()) - self.TARGET_COUNTRIES
                if invalid_countries:
                    self.validation_log.append(f"❌ Country Filter: {table_name} contains invalid countries: {invalid_countries}")
                    return False
                else:
                    self.validation_log.append(f"✅ Country Filter: {table_name} only contains target countries")
        
        return True
    
    def validate_reference_year(self, reference_year: int, data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Year gate: validate reference year selection and document fallbacks"""
        fallbacks = {}
        
        for table_name, df in data.items():
            if 'year' in df.columns:
                country_fallbacks = []
                for country in self.TARGET_COUNTRIES:
                    country_data = df[df['iso3'] == country]
                    if not country_data.empty:
                        available_years = country_data['year'].unique()
                        if reference_year not in available_years:
                            latest_year = max(available_years)
                            if latest_year <= reference_year:
                                country_fallbacks.append(f"{country}: using {latest_year} (quality penalty applied)")
                            else:
                                country_fallbacks.append(f"{country}: no valid data <= {reference_year}")
                
                if country_fallbacks:
                    fallbacks[table_name] = country_fallbacks
                    self.validation_log.append(f"⚠️ Year Fallbacks in {table_name}: {len(country_fallbacks)} countries")
                else:
                    self.validation_log.append(f"✅ Year Gate: {table_name} has data for reference year {reference_year}")
        
        return fallbacks
    
    def validate_data_types_and_ranges(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Type/range gate: validate data types and value ranges"""
        validation_rules = {
            'emissions_tco2e': {'type': 'numeric', 'min': 0},
            'coal_share_electricity_pct': {'type': 'numeric', 'min': 0, 'max': 100},
            'coal_consumption_twh': {'type': 'numeric', 'min': 0},
            'mining_area_km2': {'type': 'numeric', 'min': 0},
            'mining_sites_count': {'type': 'numeric', 'min': 0},
            'data_quality': {'type': 'numeric', 'min': 0, 'max': 1},
            'coverage_pct': {'type': 'numeric', 'min': 0, 'max': 100},
            'state_ownership_share_pct': {'type': 'numeric', 'min': 0, 'max': 100},
            'exit_year': {'type': 'numeric', 'min': 2020, 'max': 2100}
        }
        
        all_valid = True
        
        for table_name, df in data.items():
            for col_name, rules in validation_rules.items():
                if col_name in df.columns:
                    col_data = df[col_name].dropna()
                    
                    # Check numeric type
                    if rules['type'] == 'numeric':
                        if not pd.api.types.is_numeric_dtype(col_data):
                            self.validation_log.append(f"❌ Type: {table_name}.{col_name} is not numeric")
                            all_valid = False
                            continue
                        
                        # Check range constraints
                        if 'min' in rules:
                            violations = (col_data < rules['min']).sum()
                            if violations > 0:
                                self.validation_log.append(f"❌ Range: {table_name}.{col_name} has {violations} values < {rules['min']}")
                                all_valid = False
                        
                        if 'max' in rules:
                            violations = (col_data > rules['max']).sum()
                            if violations > 0:
                                self.validation_log.append(f"❌ Range: {table_name}.{col_name} has {violations} values > {rules['max']}")
                                all_valid = False
                        
                        if all_valid:
                            self.validation_log.append(f"✅ Type/Range: {table_name}.{col_name} valid")
        
        return all_valid
    
    def validate_normalization(self, metrics: Dict[str, Any]) -> bool:
        """Outlier gate: validate winsorization and normalization"""
        if 'pillar_scores' not in metrics:
            return True
        
        for pillar_name, pillar_data in metrics.get('submetrics', {}).items():
            for metric in pillar_data:
                # Check for NaNs after normalization
                if pd.isna(metric.get('norm_value')):
                    self.validation_log.append(f"❌ Normalization: {pillar_name} metric has NaN normalized value")
                    return False
                
                # Check winsor bounds are recorded
                if 'winsor_low' not in metric or 'winsor_high' not in metric:
                    self.validation_log.append(f"❌ Winsorization: {pillar_name} metric missing winsor bounds")
                    return False
                
                # Check normalized value is in valid range
                norm_val = metric['norm_value']
                if not (0 <= norm_val <= 100):
                    self.validation_log.append(f"❌ Normalization: {pillar_name} normalized value {norm_val} out of range [0,100]")
                    return False
        
        self.validation_log.append("✅ Normalization: All metrics properly normalized and winsorized")
        return True
    
    def run_cross_checks(self, metrics: Dict[str, Any], country: str) -> List[str]:
        """Run cross-validation checks between metrics"""
        inconsistencies = []
        
        try:
            # Get pillar scores
            pillar_scores = metrics.get('pillar_scores', {})
            necessity_score = pillar_scores.get('Necessity', 0)
            emissions_score = pillar_scores.get('Emissions', 0)
            
            # Get submetrics for detailed checks
            submetrics = metrics.get('submetrics', {})
            
            # Check 1: Low coal share but high consumption
            necessity_subs = submetrics.get('Necessity', [])
            coal_share = None
            coal_consumption = None
            
            for metric in necessity_subs:
                if 'coal_share' in metric.get('name', '').lower():
                    coal_share = metric.get('raw_value')
                elif 'consumption' in metric.get('name', '').lower():
                    coal_consumption = metric.get('raw_value')
            
            if coal_share is not None and coal_consumption is not None:
                if coal_share < 5 and coal_consumption > 500:  # Arbitrary threshold for "high"
                    inconsistencies.append(f"Coal share <5% but consumption >{coal_consumption} TWh - possible data inconsistency")
            
            # Check 2: Early exit year but high emissions
            support_subs = submetrics.get('ArtificialSupport', [])
            exit_year = None
            
            for metric in support_subs:
                if 'exit' in metric.get('name', '').lower():
                    exit_year = metric.get('raw_value')
            
            if exit_year is not None and exit_year <= 2030 and emissions_score >= 70:
                inconsistencies.append(f"Exit year {exit_year} but high emissions score {emissions_score} - policy mismatch")
            
        except Exception as e:
            inconsistencies.append(f"Error during cross-checks: {str(e)}")
        
        return inconsistencies
    
    def validate_provenance(self, metrics: Dict[str, Any]) -> bool:
        """Ensure every metric has dataset provenance"""
        submetrics = metrics.get('submetrics', {})
        
        for pillar_name, pillar_metrics in submetrics.items():
            for metric in pillar_metrics:
                if not metric.get('dataset_ids'):
                    self.validation_log.append(f"❌ Provenance: {pillar_name} metric missing dataset_ids")
                    return False
        
        self.validation_log.append("✅ Provenance: All metrics have dataset provenance")
        return True
    
    def validate_weights(self, weights: Dict[str, Any]) -> bool:
        """Audit weight calculations"""
        pillar_weights = weights.get('pillars', {})
        
        # Check weights sum to 1
        weight_sum = sum(pillar_weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            self.validation_log.append(f"❌ Weights: Pillar weights sum to {weight_sum}, not 1.0")
            return False
        
        # Check minimum floor applied
        min_weight = min(pillar_weights.values())
        floor = self.config.get('weight_engine', {}).get('floor', 0.05)
        if min_weight < floor - 0.001:
            self.validation_log.append(f"❌ Weights: Minimum weight {min_weight} below floor {floor}")
            return False
        
        self.validation_log.append("✅ Weights: All weight constraints satisfied")
        return True
    
    def generate_qa_report(self, run_id: str, reference_year: int, 
                          data: Dict[str, pd.DataFrame], metrics: Dict[str, Any], 
                          weights: Dict[str, Any], country: str) -> str:
        """Generate comprehensive QA report"""
        
        # Run all validations
        schema_valid = self.validate_schema(data)
        country_valid = self.validate_country_filter(data)
        year_fallbacks = self.validate_reference_year(reference_year, data)
        types_valid = self.validate_data_types_and_ranges(data)
        norm_valid = self.validate_normalization(metrics)
        prov_valid = self.validate_provenance(metrics)
        weights_valid = self.validate_weights(weights)
        
        cross_check_issues = self.run_cross_checks(metrics, country)
        
        # Generate report
        report_lines = [
            f"# FVI Phase 1 QA Report",
            f"**Run ID:** {run_id}",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Country:** {country}",
            f"**Reference Year:** {reference_year}",
            "",
            "## Validation Results",
            "",
            "### Gate Validations",
        ]
        
        report_lines.extend([f"- {log}" for log in self.validation_log])
        
        if year_fallbacks:
            report_lines.extend([
                "",
                "### Year Fallbacks Applied",
            ])
            for table, fallbacks in year_fallbacks.items():
                report_lines.append(f"**{table}:**")
                report_lines.extend([f"- {fb}" for fb in fallbacks])
        
        if cross_check_issues:
            report_lines.extend([
                "",
                "### Cross-Check Inconsistencies",
            ])
            report_lines.extend([f"- {issue}" for issue in cross_check_issues])
        
        # Summary
        all_gates_passed = all([schema_valid, country_valid, types_valid, norm_valid, prov_valid, weights_valid])
        
        report_lines.extend([
            "",
            "## Summary",
            f"**Overall Status:** {'✅ PASS' if all_gates_passed else '❌ FAIL'}",
            f"**Schema Validation:** {'✅' if schema_valid else '❌'}",
            f"**Country Filter:** {'✅' if country_valid else '❌'}",
            f"**Data Types/Ranges:** {'✅' if types_valid else '❌'}",
            f"**Normalization:** {'✅' if norm_valid else '❌'}",
            f"**Provenance:** {'✅' if prov_valid else '❌'}",
            f"**Weight Constraints:** {'✅' if weights_valid else '❌'}",
            f"**Year Fallbacks:** {len(year_fallbacks)} tables affected",
            f"**Cross-Check Issues:** {len(cross_check_issues)} found",
        ])
        
        return "\n".join(report_lines)
    
    def save_qa_report(self, report: str, run_id: str):
        """Save QA report to artifacts/logs"""
        Path("artifacts/logs").mkdir(parents=True, exist_ok=True)
        
        with open(f"artifacts/logs/qa_{run_id}.md", 'w') as f:
            f.write(report)
