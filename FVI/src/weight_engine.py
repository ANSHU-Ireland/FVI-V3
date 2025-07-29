"""
FVI Dynamic Weight Engine
Implements the depth-aware dynamic weighting system based on user profile.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

class FVIWeightEngine:
    def __init__(self, config: Dict):
        """Initialize the weight engine."""
        self.config = config
        self.weight_config = config['weight_engine']
        self.logger = logging.getLogger(__name__)
        
        # Alpha values by depth
        self.alpha_by_depth = self.weight_config['alpha_by_depth']
        self.floor = self.weight_config['floor']
        
        # Persona/time/risk nudge tables (exact values from specification)
        self.nudge_tables = self._initialize_nudge_tables()
        
    def _initialize_nudge_tables(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Initialize the exact nudge tables from specification."""
        
        return {
            'persona': {
                'investor': {'Emissions': 0.20, 'Necessity': 0.00, 'Ecological': 0.05, 'ArtificialSupport': 0.15},
                'regulator': {'Emissions': 0.25, 'Necessity': 0.10, 'Ecological': 0.20, 'ArtificialSupport': 0.25},
                'operator': {'Emissions': 0.15, 'Necessity': 0.20, 'Ecological': 0.05, 'ArtificialSupport': 0.10}
            },
            'time_horizon': {
                '0-5y': {'Emissions': 0.05, 'Necessity': 0.15, 'Ecological': 0.00, 'ArtificialSupport': 0.20},
                '5-10y': {'Emissions': 0.10, 'Necessity': 0.10, 'Ecological': 0.10, 'ArtificialSupport': 0.10},
                '10-30y': {'Emissions': 0.20, 'Necessity': 0.05, 'Ecological': 0.20, 'ArtificialSupport': 0.05}
            },
            'risk_tolerance': {
                'low': {'Emissions': 0.20, 'Necessity': 0.10, 'Ecological': 0.15, 'ArtificialSupport': 0.20},
                'medium': {'Emissions': 0.10, 'Necessity': 0.10, 'Ecological': 0.10, 'ArtificialSupport': 0.10},
                'high': {'Emissions': 0.00, 'Necessity': 0.15, 'Ecological': 0.05, 'ArtificialSupport': 0.00}
            },
            'capex_band': {
                'none': {'Emissions': 0.00, 'Necessity': 0.20, 'Ecological': 0.10, 'ArtificialSupport': 0.10},
                '<50m': {'Emissions': 0.05, 'Necessity': 0.15, 'Ecological': 0.10, 'ArtificialSupport': 0.10},
                '50-200m': {'Emissions': 0.10, 'Necessity': 0.10, 'Ecological': 0.10, 'ArtificialSupport': 0.10},
                '>200m': {'Emissions': 0.15, 'Necessity': 0.05, 'Ecological': 0.15, 'ArtificialSupport': 0.05}
            },
            'policy_exposure': {
                'low': {'Emissions': 0.10, 'Necessity': 0.00, 'Ecological': 0.05, 'ArtificialSupport': 0.25},
                'medium': {'Emissions': 0.10, 'Necessity': 0.05, 'Ecological': 0.10, 'ArtificialSupport': 0.15},
                'high': {'Emissions': 0.15, 'Necessity': 0.10, 'Ecological': 0.10, 'ArtificialSupport': 0.10},
                'unknown': {'Emissions': 0.10, 'Necessity': 0.05, 'Ecological': 0.10, 'ArtificialSupport': 0.15}  # Default to medium
            }
        }
    
    def collect_user_profile(self, user_iso3: str) -> Dict[str, str]:
        """
        Collect user profile through interactive prompts.
        In a real implementation, this would be a chat interface.
        For Phase 1, this returns a structure that can be filled.
        """
        
        # Validate user country
        if user_iso3 not in self.config['countries']:
            raise ValueError(f"User country {user_iso3} not in target countries {self.config['countries']}")
        
        # Profile structure (to be filled by chat interface)
        profile_template = {
            'user_iso3': user_iso3,
            'depth': None,  # 'quick', 'standard', 'deep'
            'persona': None,  # 'investor', 'regulator', 'operator'
            'time_horizon': None,  # '0-5y', '5-10y', '10-30y'
            'risk_tolerance': None,  # 'low', 'medium', 'high'
            'capex_band': None,  # 'none', '<50m', '50-200m', '>200m'
            'policy_exposure': None  # 'low', 'medium', 'high', 'unknown'
        }
        
        return profile_template
    
    def calculate_dynamic_weights(self, user_profile: Dict[str, str], available_pillars: List[str], 
                                run_id: str = None) -> Dict[str, Any]:
        """
        Calculate dynamic weights based on user profile.
        
        Args:
            user_profile: User profile dictionary
            available_pillars: List of available pillars for the user's country
            run_id: Optional run ID for logging
            
        Returns:
            Dictionary with weights and explainability information
        """
        
        self.logger.info(f"Calculating dynamic weights for {user_profile['user_iso3']}")
        
        # Validate inputs
        self._validate_user_profile(user_profile)
        
        # 1. Start with equal priors for available pillars
        w0 = self._get_prior_weights(available_pillars)
        
        # 2. Calculate deltas from nudge tables
        deltas = self._calculate_deltas(user_profile, available_pillars)
        
        # 3. Apply depth scaling (alpha)
        alpha = self.alpha_by_depth[user_profile['depth']]
        
        # 4. Compute logits and apply softmax
        w_logits = {}
        for pillar in available_pillars:
            w_logits[pillar] = w0[pillar] + alpha * deltas[pillar]
        
        # Convert to weights using softmax
        logit_values = np.array(list(w_logits.values()))
        softmax_weights = np.exp(logit_values - np.max(logit_values))  # Numerical stability
        softmax_weights = softmax_weights / np.sum(softmax_weights)
        
        # 5. Apply floor and renormalize
        final_weights = {}
        for i, pillar in enumerate(available_pillars):
            final_weights[pillar] = max(softmax_weights[i], self.floor)
        
        # Renormalize to sum to 1
        total_weight = sum(final_weights.values())
        for pillar in final_weights:
            final_weights[pillar] = final_weights[pillar] / total_weight
        
        # 6. Calculate within-pillar subweights (equal for Phase 1, with optional deep nudges)
        subweights = self._calculate_subweights(user_profile, available_pillars)
        
        # 7. Create explainability record
        explainability = {
            'user_profile': user_profile,
            'available_pillars': available_pillars,
            'priors': w0,
            'deltas': deltas,
            'alpha': alpha,
            'logits': w_logits,
            'pre_floor_weights': dict(zip(available_pillars, softmax_weights)),
            'final_weights': final_weights,
            'subweights': subweights,
            'timestamp': datetime.now().isoformat(),
            'run_id': run_id
        }
        
        # 8. Save weights to artifacts
        self._save_weights_log(explainability, user_profile['user_iso3'], run_id)
        
        return {
            'pillar_weights': final_weights,
            'subweights': subweights,
            'explainability': explainability
        }
    
    def _validate_user_profile(self, profile: Dict[str, str]):
        """Validate user profile has all required fields."""
        required_fields = ['user_iso3', 'depth', 'persona', 'time_horizon', 'risk_tolerance', 'capex_band', 'policy_exposure']
        
        for field in required_fields:
            if field not in profile or profile[field] is None:
                raise ValueError(f"Missing required profile field: {field}")
        
        # Validate field values
        valid_values = {
            'depth': ['quick', 'standard', 'deep'],
            'persona': ['investor', 'regulator', 'operator'],
            'time_horizon': ['0-5y', '5-10y', '10-30y'],
            'risk_tolerance': ['low', 'medium', 'high'],
            'capex_band': ['none', '<50m', '50-200m', '>200m'],
            'policy_exposure': ['low', 'medium', 'high', 'unknown']
        }
        
        for field, valid_list in valid_values.items():
            if profile[field] not in valid_list:
                raise ValueError(f"Invalid value for {field}: {profile[field]}. Must be one of {valid_list}")
    
    def _get_prior_weights(self, available_pillars: List[str]) -> Dict[str, float]:
        """Get equal prior weights for available pillars."""
        if not available_pillars:
            return {}
        
        equal_weight = 1.0 / len(available_pillars)
        return {pillar: equal_weight for pillar in available_pillars}
    
    def _calculate_deltas(self, user_profile: Dict[str, str], available_pillars: List[str]) -> Dict[str, float]:
        """Calculate additive deltas from all nudge factors."""
        
        deltas = {pillar: 0.0 for pillar in available_pillars}
        
        # Sum deltas from each factor
        factors = ['persona', 'time_horizon', 'risk_tolerance', 'capex_band', 'policy_exposure']
        
        for factor in factors:
            factor_value = user_profile[factor]
            if factor_value in self.nudge_tables[factor]:
                factor_nudges = self.nudge_tables[factor][factor_value]
                for pillar in available_pillars:
                    if pillar in factor_nudges:
                        deltas[pillar] += factor_nudges[pillar]
        
        return deltas
    
    def _calculate_subweights(self, user_profile: Dict[str, str], available_pillars: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate within-pillar sub-metric weights."""
        
        subweights = {}
        
        # For Phase 1, use equal weights within pillars
        # Optional: apply small persona nudges if depth = deep
        
        pillar_submetrics = {
            'Emissions': ['ES_Absolute', 'ES_Intensity', 'ES_CoverageGap'],
            'Necessity': ['NEC_CoalShareElec', 'NEC_ConsumptionLock', 'NEC_PopAtRisk'],
            'Ecological': ['ECO_LandDisturbance', 'ECO_SiteDensity', 'ECO_DeforestExposure', 'ECO_AshRisk'],
            'ArtificialSupport': ['AS_ExitPenalty', 'AS_StateOwnership', 'AS_CoverageGap', 'AS_SubsidyProxy']
        }
        
        for pillar in available_pillars:
            if pillar in pillar_submetrics:
                submetrics = pillar_submetrics[pillar]
                equal_weight = 1.0 / len(submetrics)
                pillar_subweights = {metric: equal_weight for metric in submetrics}
                
                # Optional deep persona nudges (±0.05)
                if user_profile['depth'] == 'deep':
                    if pillar == 'Emissions' and user_profile['persona'] == 'operator':
                        # Prioritize intensity over absolute for operators
                        pillar_subweights['ES_Intensity'] += 0.05
                        pillar_subweights['ES_Absolute'] -= 0.05
                    
                    # Renormalize to sum to 1
                    total = sum(pillar_subweights.values())
                    for metric in pillar_subweights:
                        pillar_subweights[metric] = pillar_subweights[metric] / total
                
                subweights[pillar] = pillar_subweights
        
        return subweights
    
    def _save_weights_log(self, explainability: Dict, user_iso3: str, run_id: str):
        """Save weights explainability to log file."""
        
        logs_dir = self.config.get('workspace_path', '.') + "/artifacts/logs"
        import os
        os.makedirs(logs_dir, exist_ok=True)
        
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f"weights_{user_iso3}_{run_id}.json"
        filepath = os.path.join(logs_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(explainability, f, indent=2)
        
        self.logger.info(f"Weights log saved to {filepath}")
    
    def get_weight_explanation(self, explainability: Dict) -> str:
        """Generate human-readable explanation of weight calculation."""
        
        profile = explainability['user_profile']
        weights = explainability['final_weights']
        deltas = explainability['deltas']
        alpha = explainability['alpha']
        
        explanation = f"Weight Calculation for {profile['user_iso3']} ({profile['persona']}, {profile['depth']} analysis):\n\n"
        
        explanation += "Starting with equal weights, the following adjustments were made:\n"
        
        # Explain persona effect
        explanation += f"• Persona ({profile['persona']}): "
        persona_effects = []
        for pillar, delta in deltas.items():
            persona_nudge = self.nudge_tables['persona'][profile['persona']].get(pillar, 0)
            if persona_nudge != 0:
                persona_effects.append(f"{pillar} {persona_nudge:+.2f}")
        explanation += ", ".join(persona_effects) + "\n"
        
        # Explain time horizon effect
        explanation += f"• Time Horizon ({profile['time_horizon']}): "
        time_effects = []
        for pillar, delta in deltas.items():
            time_nudge = self.nudge_tables['time_horizon'][profile['time_horizon']].get(pillar, 0)
            if time_nudge != 0:
                time_effects.append(f"{pillar} {time_nudge:+.2f}")
        explanation += ", ".join(time_effects) + "\n"
        
        explanation += f"• Depth scaling (α = {alpha}) applied to all deltas\n"
        explanation += f"• Floor of {self.floor} applied and renormalized\n\n"
        
        explanation += "Final weights:\n"
        for pillar, weight in weights.items():
            explanation += f"• {pillar}: {weight:.3f}\n"
        
        return explanation
