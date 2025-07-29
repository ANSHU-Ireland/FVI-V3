"""
FVI Metrics Calculation Module
Computes pillar sub-metrics and scores according to Phase 1 specifications.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

class FVIMetricsCalculator:
    def __init__(self, config: Dict, fact_tables: Dict[str, pd.DataFrame], reference_year: int):
        """Initialize metrics calculator."""
        self.config = config
        self.fact_tables = fact_tables
        self.reference_year = reference_year
        self.target_countries = config['countries']
        self.logger = logging.getLogger(__name__)
        
        # Winsorization limits
        self.winsor_limits = config['winsor_limits_pct']
        
        # Normalization strategies
        self.norm_strategies = config['normalize_strategy_per_metric']
        
    def calculate_emissions_pillar(self) -> Dict[str, pd.DataFrame]:
        """Calculate emissions pillar sub-metrics."""
        self.logger.info("Calculating Emissions pillar metrics...")
        
        fact_emissions = self.fact_tables.get('fact_emissions', pd.DataFrame())
        if fact_emissions.empty:
            self.logger.warning("No emissions data available")
            return {'submetrics': pd.DataFrame(), 'pillar_scores': pd.DataFrame()}
        
        submetrics = []
        
        for _, row in fact_emissions.iterrows():
            iso3 = row['iso3']
            
            # ES_Absolute (log-scaled before normalization)
            es_absolute = {
                'iso3': iso3,
                'metric': 'ES_Absolute',
                'raw_value': row['emissions_tco2e'],
                'log_value': np.log(max(row['emissions_tco2e'], 1)),  # Avoid log(0)
                'data_quality': row['data_quality'],
                'dataset_ids': row['dataset_ids']
            }
            submetrics.append(es_absolute)
            
            # ES_Intensity (if available or can be proxied)
            intensity_value = None
            proxy_intensity = False
            
            if 'intensity_tco2_per_mwh' in row and pd.notna(row['intensity_tco2_per_mwh']):
                intensity_value = row['intensity_tco2_per_mwh']
            else:
                # Try to proxy from coal consumption
                fact_necessity = self.fact_tables.get('fact_necessity', pd.DataFrame())
                country_necessity = fact_necessity[fact_necessity['iso3'] == iso3]
                
                if not country_necessity.empty:
                    coal_consumption_twh = country_necessity.iloc[0]['coal_consumption_twh']
                    if pd.notna(coal_consumption_twh) and coal_consumption_twh > 0:
                        intensity_value = row['emissions_tco2e'] / (coal_consumption_twh * 1e6)  # TWh to MWh
                        proxy_intensity = True
            
            if intensity_value is not None:
                es_intensity = {
                    'iso3': iso3,
                    'metric': 'ES_Intensity',
                    'raw_value': intensity_value,
                    'data_quality': row['data_quality'] - (0.2 if proxy_intensity else 0),
                    'dataset_ids': row['dataset_ids'],
                    'proxy_intensity': proxy_intensity
                }
                submetrics.append(es_intensity)
            
            # ES_CoverageGap
            if 'coverage_pct' in row and pd.notna(row['coverage_pct']):
                coverage_gap = 100 - row['coverage_pct']
                es_coverage = {
                    'iso3': iso3,
                    'metric': 'ES_CoverageGap',
                    'raw_value': coverage_gap,
                    'data_quality': row['data_quality'],
                    'dataset_ids': row['dataset_ids']
                }
                submetrics.append(es_coverage)
        
        submetrics_df = pd.DataFrame(submetrics)
        
        # Apply normalization
        normalized_submetrics = self._normalize_submetrics(submetrics_df, 'Emissions')
        
        # Calculate pillar scores
        pillar_scores = self._calculate_pillar_scores(normalized_submetrics, 'Emissions')
        
        return {'submetrics': normalized_submetrics, 'pillar_scores': pillar_scores}
    
    def calculate_necessity_pillar(self) -> Dict[str, pd.DataFrame]:
        """Calculate necessity pillar sub-metrics."""
        self.logger.info("Calculating Necessity pillar metrics...")
        
        fact_necessity = self.fact_tables.get('fact_necessity', pd.DataFrame())
        if fact_necessity.empty:
            self.logger.warning("No necessity data available")
            return {'submetrics': pd.DataFrame(), 'pillar_scores': pd.DataFrame()}
        
        submetrics = []
        
        for _, row in fact_necessity.iterrows():
            iso3 = row['iso3']
            
            # NEC_CoalShareElec
            if pd.notna(row['coal_share_electricity_pct']):
                nec_coal_share = {
                    'iso3': iso3,
                    'metric': 'NEC_CoalShareElec',
                    'raw_value': row['coal_share_electricity_pct'],
                    'data_quality': row['data_quality'],
                    'dataset_ids': row['dataset_ids']
                }
                submetrics.append(nec_coal_share)
            
            # NEC_ConsumptionLock
            if pd.notna(row['coal_consumption_twh']):
                nec_consumption = {
                    'iso3': iso3,
                    'metric': 'NEC_ConsumptionLock',
                    'raw_value': row['coal_consumption_twh'],
                    'data_quality': row['data_quality'],
                    'dataset_ids': row['dataset_ids']
                }
                submetrics.append(nec_consumption)
            
            # NEC_PopAtRisk (placeholder - would need population data)
            # For Phase 1, mark as missing if not available
        
        submetrics_df = pd.DataFrame(submetrics)
        
        # Apply normalization
        normalized_submetrics = self._normalize_submetrics(submetrics_df, 'Necessity')
        
        # Calculate pillar scores
        pillar_scores = self._calculate_pillar_scores(normalized_submetrics, 'Necessity')
        
        return {'submetrics': normalized_submetrics, 'pillar_scores': pillar_scores}
    
    def calculate_ecological_pillar(self) -> Dict[str, pd.DataFrame]:
        """Calculate ecological pillar sub-metrics."""
        self.logger.info("Calculating Ecological pillar metrics...")
        
        fact_ecological = self.fact_tables.get('fact_ecological', pd.DataFrame())
        if fact_ecological.empty:
            self.logger.warning("No ecological data available")
            return {'submetrics': pd.DataFrame(), 'pillar_scores': pd.DataFrame()}
        
        submetrics = []
        
        for _, row in fact_ecological.iterrows():
            iso3 = row['iso3']
            
            # Get activity denominator
            denominator = 1.0  # Default
            if pd.notna(row.get('activity_denominator_value')):
                denominator = row['activity_denominator_value']
            
            # ECO_LandDisturbance
            if pd.notna(row['mining_area_km2']):
                land_disturbance = row['mining_area_km2'] / denominator
                eco_land = {
                    'iso3': iso3,
                    'metric': 'ECO_LandDisturbance',
                    'raw_value': land_disturbance,
                    'data_quality': row['data_quality'],
                    'dataset_ids': row['dataset_ids']
                }
                submetrics.append(eco_land)
            
            # ECO_SiteDensity
            if pd.notna(row['mining_sites_count']):
                site_density = row['mining_sites_count'] / denominator
                eco_sites = {
                    'iso3': iso3,
                    'metric': 'ECO_SiteDensity',
                    'raw_value': site_density,
                    'data_quality': row['data_quality'],
                    'dataset_ids': row['dataset_ids']
                }
                submetrics.append(eco_sites)
            
            # ECO_DeforestExposure
            if pd.notna(row.get('deforest_area_km2')):
                # Calculate as share of total deforestation across the five countries
                total_deforest = fact_ecological['deforest_area_km2'].sum()
                if total_deforest > 0:
                    deforest_share = (row['deforest_area_km2'] / total_deforest) * 100
                    eco_deforest = {
                        'iso3': iso3,
                        'metric': 'ECO_DeforestExposure',
                        'raw_value': deforest_share,
                        'data_quality': row['data_quality'],
                        'dataset_ids': row['dataset_ids']
                    }
                    submetrics.append(eco_deforest)
            
            # ECO_AshRisk (optional in Phase 1)
            # Would be calculated from ash sites data if available
        
        submetrics_df = pd.DataFrame(submetrics)
        
        # Apply normalization
        normalized_submetrics = self._normalize_submetrics(submetrics_df, 'Ecological')
        
        # Calculate pillar scores
        pillar_scores = self._calculate_pillar_scores(normalized_submetrics, 'Ecological')
        
        return {'submetrics': normalized_submetrics, 'pillar_scores': pillar_scores}
    
    def calculate_artificial_support_pillar(self) -> Dict[str, pd.DataFrame]:
        """Calculate artificial support pillar sub-metrics."""
        self.logger.info("Calculating Artificial Support pillar metrics...")
        
        fact_support = self.fact_tables.get('fact_support', pd.DataFrame())
        if fact_support.empty:
            self.logger.warning("No artificial support data available")
            return {'submetrics': pd.DataFrame(), 'pillar_scores': pd.DataFrame()}
        
        submetrics = []
        
        for _, row in fact_support.iterrows():
            iso3 = row['iso3']
            
            # AS_ExitPenalty
            if pd.notna(row.get('exit_year')):
                exit_penalty = 100 * (row['exit_year'] - self.reference_year) / (2100 - self.reference_year)
                exit_penalty = np.clip(exit_penalty, 0, 100)
                
                as_exit = {
                    'iso3': iso3,
                    'metric': 'AS_ExitPenalty',
                    'raw_value': exit_penalty,
                    'data_quality': row['data_quality'],
                    'dataset_ids': row['dataset_ids']
                }
                submetrics.append(as_exit)
            
            # AS_StateOwnership
            if pd.notna(row.get('state_ownership_share_pct')):
                as_ownership = {
                    'iso3': iso3,
                    'metric': 'AS_StateOwnership',
                    'raw_value': row['state_ownership_share_pct'],
                    'data_quality': row['data_quality'],
                    'dataset_ids': row['dataset_ids']
                }
                submetrics.append(as_ownership)
            
            # AS_CoverageGap (same as ES_CoverageGap)
            if 'coverage_pct' in row and pd.notna(row['coverage_pct']):
                coverage_gap = 100 - row['coverage_pct']
                as_coverage = {
                    'iso3': iso3,
                    'metric': 'AS_CoverageGap',
                    'raw_value': coverage_gap,
                    'data_quality': row['data_quality'],
                    'dataset_ids': row['dataset_ids']
                }
                submetrics.append(as_coverage)
        
        # AS_SubsidyProxy (percentile rank of emissions across five countries)
        fact_emissions = self.fact_tables.get('fact_emissions', pd.DataFrame())
        if not fact_emissions.empty:
            emissions_data = fact_emissions[['iso3', 'emissions_tco2e']].copy()
            emissions_data['subsidy_proxy'] = emissions_data['emissions_tco2e'].rank(pct=True) * 100
            
            for _, row in emissions_data.iterrows():
                as_subsidy = {
                    'iso3': row['iso3'],
                    'metric': 'AS_SubsidyProxy',
                    'raw_value': row['subsidy_proxy'],
                    'data_quality': 0.8,  # Proxy quality
                    'dataset_ids': ['annual-co2-emission.csv (proxy)']
                }
                submetrics.append(as_subsidy)
        
        submetrics_df = pd.DataFrame(submetrics)
        
        # Apply normalization
        normalized_submetrics = self._normalize_submetrics(submetrics_df, 'ArtificialSupport')
        
        # Calculate pillar scores
        pillar_scores = self._calculate_pillar_scores(normalized_submetrics, 'ArtificialSupport')
        
        return {'submetrics': normalized_submetrics, 'pillar_scores': pillar_scores}
    
    def _normalize_submetrics(self, submetrics_df: pd.DataFrame, pillar: str) -> pd.DataFrame:
        """Apply winsorization and normalization to sub-metrics."""
        if submetrics_df.empty:
            return submetrics_df
        
        normalized_df = submetrics_df.copy()
        
        # Group by metric for normalization
        for metric in submetrics_df['metric'].unique():
            metric_data = submetrics_df[submetrics_df['metric'] == metric].copy()
            
            if len(metric_data) < 2:  # Need at least 2 points for normalization
                metric_data['winsor_low'] = metric_data['raw_value']
                metric_data['winsor_high'] = metric_data['raw_value']
                metric_data['norm_value'] = 50.0  # Default middle value
            else:
                values = metric_data['raw_value'].values
                
                # Apply winsorization
                low_pct, high_pct = self.winsor_limits
                winsor_low = np.percentile(values, low_pct)
                winsor_high = np.percentile(values, high_pct)
                
                winsorized_values = np.clip(values, winsor_low, winsor_high)
                
                # Apply log transformation if specified
                if metric == 'ES_Absolute' and self.norm_strategies.get('emissions_abs') == 'log_minmax':
                    # Use log values that were already computed
                    if 'log_value' in metric_data.columns:
                        transform_values = metric_data['log_value'].values
                    else:
                        transform_values = np.log(np.maximum(winsorized_values, 1))
                else:
                    transform_values = winsorized_values
                
                # Min-max normalization to 0-100 (where 100 = worse)
                if len(np.unique(transform_values)) > 1:
                    min_val = transform_values.min()
                    max_val = transform_values.max()
                    norm_values = (transform_values - min_val) / (max_val - min_val) * 100
                else:
                    norm_values = np.full(len(transform_values), 50.0)
                
                # For Necessity pillar, keep as need index (don't invert)
                # For other pillars, higher normalized value = worse
                
                metric_data['winsor_low'] = winsor_low
                metric_data['winsor_high'] = winsor_high
                metric_data['norm_value'] = norm_values
            
            # Update normalized dataframe
            mask = normalized_df['metric'] == metric
            normalized_df.loc[mask, 'winsor_low'] = metric_data['winsor_low'].values
            normalized_df.loc[mask, 'winsor_high'] = metric_data['winsor_high'].values
            normalized_df.loc[mask, 'norm_value'] = metric_data['norm_value'].values
        
        return normalized_df
    
    def _calculate_pillar_scores(self, submetrics_df: pd.DataFrame, pillar: str) -> pd.DataFrame:
        """Calculate pillar scores by averaging available sub-metrics."""
        if submetrics_df.empty:
            return pd.DataFrame()
        
        pillar_scores = []
        
        for iso3 in self.target_countries:
            country_metrics = submetrics_df[submetrics_df['iso3'] == iso3]
            
            if not country_metrics.empty:
                # Equal weights for sub-metrics within pillar
                pillar_score = country_metrics['norm_value'].mean()
                min_quality = country_metrics['data_quality'].min()
                mean_quality = country_metrics['data_quality'].mean()
                
                pillar_scores.append({
                    'iso3': iso3,
                    'pillar': pillar,
                    'pillar_score': pillar_score,
                    'num_submetrics': len(country_metrics),
                    'min_data_quality': min_quality,
                    'mean_data_quality': mean_quality,
                    'available_metrics': country_metrics['metric'].tolist()
                })
        
        return pd.DataFrame(pillar_scores)
    
    def calculate_all_pillars(self) -> Dict[str, Any]:
        """Calculate all pillar metrics and scores."""
        self.logger.info("Calculating all pillar metrics...")
        
        results = {
            'Emissions': self.calculate_emissions_pillar(),
            'Necessity': self.calculate_necessity_pillar(),
            'Ecological': self.calculate_ecological_pillar(),
            'ArtificialSupport': self.calculate_artificial_support_pillar()
        }
        
        # Combine all submetrics
        all_submetrics = []
        all_pillar_scores = []
        
        for pillar, data in results.items():
            if not data['submetrics'].empty:
                pillar_submetrics = data['submetrics'].copy()
                pillar_submetrics['pillar'] = pillar
                all_submetrics.append(pillar_submetrics)
            
            if not data['pillar_scores'].empty:
                all_pillar_scores.append(data['pillar_scores'])
        
        combined_submetrics = pd.concat(all_submetrics, ignore_index=True) if all_submetrics else pd.DataFrame()
        combined_pillar_scores = pd.concat(all_pillar_scores, ignore_index=True) if all_pillar_scores else pd.DataFrame()
        
        # Data gap analysis
        data_gaps = self._analyze_data_gaps(results)
        
        return {
            'submetrics': combined_submetrics,
            'pillar_scores': combined_pillar_scores,
            'pillar_results': results,
            'data_gaps': data_gaps
        }
    
    def _analyze_data_gaps(self, results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze missing data for each country."""
        data_gaps = {iso3: [] for iso3 in self.target_countries}
        
        # Expected metrics per pillar
        expected_metrics = {
            'Emissions': ['ES_Absolute', 'ES_Intensity', 'ES_CoverageGap'],
            'Necessity': ['NEC_CoalShareElec', 'NEC_ConsumptionLock', 'NEC_PopAtRisk'],
            'Ecological': ['ECO_LandDisturbance', 'ECO_SiteDensity', 'ECO_DeforestExposure', 'ECO_AshRisk'],
            'ArtificialSupport': ['AS_ExitPenalty', 'AS_StateOwnership', 'AS_CoverageGap', 'AS_SubsidyProxy']
        }
        
        for pillar, pillar_data in results.items():
            if pillar_data['submetrics'].empty:
                # Entire pillar missing
                for iso3 in self.target_countries:
                    data_gaps[iso3].extend([f"{pillar}: {metric}" for metric in expected_metrics[pillar]])
            else:
                # Check for missing metrics per country
                for iso3 in self.target_countries:
                    country_metrics = pillar_data['submetrics'][pillar_data['submetrics']['iso3'] == iso3]['metric'].tolist()
                    missing_metrics = set(expected_metrics[pillar]) - set(country_metrics)
                    data_gaps[iso3].extend([f"{pillar}: {metric}" for metric in missing_metrics])
        
        return data_gaps
