"""
FVI Facts Generator and Gemini Integration
Generates facts JSON and integrates with Gemini for explanations.
"""

import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai

class FVIFactsGenerator:
    def __init__(self, config: Dict):
        """Initialize the facts generator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configure Gemini
        if 'gemini_api_key' in config:
            genai.configure(api_key=config['gemini_api_key'])
            self.gemini_model = genai.GenerativeModel(config.get('gemini_model', 'gemini-2.0-flash'))
        else:
            self.logger.warning("No Gemini API key provided")
            self.gemini_model = None
    
    def generate_facts_json(self, 
                           user_profile: Dict[str, str],
                           pillar_scores: pd.DataFrame,
                           submetrics: pd.DataFrame,
                           composite_df: pd.DataFrame,
                           weights_result: Dict[str, Any],
                           data_gaps: Dict[str, List[str]],
                           reference_year: int,
                           run_id: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive facts JSON for the user's country.
        """
        
        user_iso3 = user_profile['user_iso3']
        self.logger.info(f"Generating facts JSON for {user_iso3}")
        
        # Get country data
        country_composite = composite_df[composite_df['iso3'] == user_iso3]
        country_pillars = pillar_scores[pillar_scores['iso3'] == user_iso3]
        country_submetrics = submetrics[submetrics['iso3'] == user_iso3]
        
        if country_composite.empty:
            raise ValueError(f"No composite data found for country {user_iso3}")
        
        composite_row = country_composite.iloc[0]
        
        # Entity information
        country_names = {
            'IND': 'India',
            'CHN': 'China',
            'USA': 'United States',
            'JPN': 'Japan',
            'ZAF': 'South Africa'
        }
        
        entity = {
            'iso3': user_iso3,
            'name': country_names.get(user_iso3, user_iso3)
        }
        
        # Pillar scores (only available ones)
        pillar_scores_dict = {}
        for _, row in country_pillars.iterrows():
            pillar_scores_dict[row['pillar']] = {
                'score': row['pillar_score'],
                'num_submetrics': row.get('num_submetrics', 0),
                'data_quality': {
                    'min': row.get('min_data_quality', 1.0),
                    'mean': row.get('mean_data_quality', 1.0)
                },
                'available_metrics': row.get('available_metrics', [])
            }
        
        # Sub-metrics details
        submetrics_dict = {}
        for pillar in pillar_scores_dict.keys():
            pillar_submetrics = country_submetrics[country_submetrics['pillar'] == pillar]
            submetrics_dict[pillar] = []
            
            for _, row in pillar_submetrics.iterrows():
                submetric_data = {
                    'name': row['metric'],
                    'raw_value': row['raw_value'],
                    'normalized_value': row['norm_value'],
                    'winsor_bounds': {
                        'low': row.get('winsor_low', row['raw_value']),
                        'high': row.get('winsor_high', row['raw_value'])
                    },
                    'data_quality': row['data_quality'],
                    'dataset_ids': row['dataset_ids']
                }
                
                # Add any special flags
                if row.get('proxy_intensity', False):
                    submetric_data['flags'] = ['proxy_intensity']
                
                submetrics_dict[pillar].append(submetric_data)
        
        # Weights information
        pillar_weights = weights_result['pillar_weights']
        subweights = weights_result['subweights']
        
        # Top drivers (largest contributions to composite)
        top_drivers = []
        composite_score = composite_row['composite']
        
        for pillar, pillar_data in pillar_scores_dict.items():
            if pillar in pillar_weights:
                contribution = pillar_weights[pillar] * pillar_data['score']
                contribution_pct = (contribution / composite_score) * 100 if composite_score > 0 else 0
                
                top_drivers.append({
                    'pillar': pillar,
                    'contribution_pct': contribution_pct,
                    'pillar_score': pillar_data['score'],
                    'weight': pillar_weights[pillar]
                })
        
        # Sort by contribution and take top 3
        top_drivers.sort(key=lambda x: x['contribution_pct'], reverse=True)
        top_drivers = top_drivers[:3]
        
        # Classification information
        classification = composite_row['classification']
        borderline_flag = composite_row.get('borderline_flag', False)
        guardrails_triggered = composite_row.get('guardrails_triggered', [])
        
        # Quality summary
        all_qualities = []
        proxies_used = []
        
        for pillar_data in submetrics_dict.values():
            for submetric in pillar_data:
                all_qualities.append(submetric['data_quality'])
                if submetric.get('flags') and 'proxy_intensity' in submetric['flags']:
                    proxies_used.append(submetric['name'])
        
        quality_summary = {
            'min_quality': min(all_qualities) if all_qualities else 1.0,
            'mean_quality': np.mean(all_qualities) if all_qualities else 1.0,
            'proxies_used': proxies_used
        }
        
        # Assumptions and configuration
        assumptions = {
            'thresholds': self.config['pillar_thresholds'],
            'alpha': weights_result['explainability']['alpha'],
            'floors': self.config['weight_engine']['floor'],
            'winsor_limits': self.config['winsor_limits_pct']
        }
        
        # Compile facts JSON
        facts = {
            'entity': entity,
            'reference_year': reference_year,
            'pillar_scores': pillar_scores_dict,
            'submetrics': submetrics_dict,
            'weights': {
                'pillars': pillar_weights,
                'subweights': subweights
            },
            'composite': composite_score,
            'classification': {
                'label': classification,
                'borderline_flag': borderline_flag,
                'guardrails_triggered': guardrails_triggered
            },
            'top_drivers': top_drivers,
            'data_gaps': data_gaps.get(user_iso3, []),
            'user_profile': user_profile,
            'assumptions': assumptions,
            'quality_summary': quality_summary,
            'generated_at': datetime.now().isoformat(),
            'run_id': run_id
        }
        
        return facts
    
    def generate_gemini_explanation(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation using Gemini API based on facts JSON.
        """
        
        if not self.gemini_model:
            self.logger.error("Gemini model not configured")
            return {
                'explanation_bullets': ['Gemini integration not available'],
                'recommended_actions': ['Configure Gemini API key'],
                'clarifying_questions': ['Please provide Gemini API configuration'],
                'error': 'Gemini not configured'
            }
        
        try:
            # Create the prompt
            prompt = self._create_gemini_prompt(facts)
            
            # Generate response
            response = self.gemini_model.generate_content(prompt)
            
            # Parse the response
            parsed_response = self._parse_gemini_response(response.text)
            
            self.logger.info(f"Generated Gemini explanation for {facts['entity']['iso3']}")
            return parsed_response
        
        except Exception as e:
            self.logger.error(f"Error generating Gemini explanation: {str(e)}")
            return {
                'explanation_bullets': [f'Error generating explanation: {str(e)}'],
                'recommended_actions': ['Review data quality and try again'],
                'clarifying_questions': ['Would you like to proceed without AI explanation?'],
                'error': str(e)
            }
    
    def _create_gemini_prompt(self, facts: Dict[str, Any]) -> str:
        """Create the prompt for Gemini based on facts JSON."""
        
        entity = facts['entity']
        classification = facts['classification']
        top_drivers = facts['top_drivers']
        user_profile = facts['user_profile']
        data_gaps = facts['data_gaps']
        
        prompt = f"""Use ONLY the Facts JSON provided below. You are analyzing the Fossil Vulnerability Index for {entity['name']} ({entity['iso3']}).

FACTS JSON:
{json.dumps(facts, indent=2)}

Based on these facts, provide:

1. THREE CONCISE BULLETS explaining the classification "{classification['label']}" for this country:
   - Focus on the top contributing pillars and their scores
   - Mention any guardrails or borderline conditions
   - Reference specific metric values from the facts

2. 2-3 ACTIONS aligned with the user's profile (persona: {user_profile['persona']}, time horizon: {user_profile['time_horizon']}, risk tolerance: {user_profile['risk_tolerance']}):
   - Provide concrete, actionable recommendations
   - Consider the user's role and constraints
   - Align with the time horizon and risk profile

3. UP TO 3 CLARIFYING QUESTIONS for any data gaps:
   - Only ask about significant missing data that affects the analysis
   - Prioritize questions that would most impact the classification
   - Be specific about what information is needed

IMPORTANT RULES:
- Use ONLY information from the Facts JSON provided
- Do not invent numbers, scores, or data sources
- Do not make assumptions about missing data
- Be specific and cite actual values from the facts
- Keep explanations concise and actionable

Format your response as:

EXPLANATION:
• [bullet 1]
• [bullet 2] 
• [bullet 3]

ACTIONS:
• [action 1]
• [action 2]
• [action 3]

QUESTIONS:
• [question 1]
• [question 2]
• [question 3]
"""
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response into structured format."""
        
        sections = {
            'explanation_bullets': [],
            'recommended_actions': [],
            'clarifying_questions': []
        }
        
        current_section = None
        
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.upper().startswith('EXPLANATION'):
                current_section = 'explanation_bullets'
                continue
            elif line.upper().startswith('ACTIONS'):
                current_section = 'recommended_actions'
                continue
            elif line.upper().startswith('QUESTIONS'):
                current_section = 'clarifying_questions'
                continue
            
            # Parse bullet points
            if line.startswith('•') or line.startswith('-'):
                content = line[1:].strip()
                if content and current_section:
                    sections[current_section].append(content)
        
        # Ensure we have at least some content
        if not any(sections.values()):
            sections['explanation_bullets'] = [response_text[:200] + "..." if len(response_text) > 200 else response_text]
        
        return sections
    
    def save_facts_json(self, facts: Dict[str, Any], workspace_path: str) -> str:
        """Save facts JSON to artifacts folder."""
        
        artifacts_dir = f"{workspace_path}/artifacts/facts"
        import os
        os.makedirs(artifacts_dir, exist_ok=True)
        
        iso3 = facts['entity']['iso3']
        run_id = facts.get('run_id', 'default')
        
        filename = f"{iso3}_facts.json"
        filepath = os.path.join(artifacts_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(facts, f, indent=2)
        
        self.logger.info(f"Facts JSON saved to {filepath}")
        return filepath
    
    def create_gemini_instruction_template(self) -> str:
        """Create the standard Gemini instruction template."""
        
        return """Use ONLY the Facts JSON. Provide:
(1) three concise bullets explaining the classification for this country;
(2) 2–3 actions aligned with the user's time horizon and risk tolerance;
(3) up to 3 clarifying questions for any data_gaps.
Do not invent numbers or sources."""
