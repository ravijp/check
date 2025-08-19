"""
business_intelligence.py - Employee Turnover Business Intelligence Module
================================================================================

Purpose: Individual-level driver analysis and causal intervention simulation for employee turnover prediction
Methodology: Causal intervention with business-logic cascade effects and SHAP attribution
Integration: SurvivalModelEngine with FeatureConfig-based preprocessing pipeline

Key Features:
- SHAP-based individual risk driver analysis with existing feature infrastructure
- Causal intervention simulation with realistic cascade effects
- Business-logic based confounder adjustment
- Population-level insights with risk stratification
- FeatureConfig-aligned modifiability classification

"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)

HIGH_RISK_THRESHOLD = 0.8
MEDIUM_RISK_THRESHOLD = 0.4

@dataclass
class InterventionConfig:
    """Intervention configuration with causal cascade effects"""
    salary_increase_multiplier: float = 1.15
    promotion_job_level_increment: int = 1
    max_job_level: int = 100
    meaningful_risk_reduction_threshold: float = 0.05

@dataclass
class CausalConfig:
    """Business-logic based causal relationships for realistic interventions"""
    
    @property
    def salary_increase_effects(self) -> Dict[str, Any]:
        return {
            'baseline_salary': 1.15,
            'compensation_percentile_company': lambda x: min(x + 15, 95),
            'peer_salary_ratio': lambda x: x * 1.10,
            'salary_growth_rate_12m': lambda x: max(x, 0.15),
            'team_avg_comp': 1.08,
        }
    
    @property
    def promotion_effects(self) -> Dict[str, Any]:
        return {
            'job_level': lambda x: min(float(x) + 1, 100),
            'role_complexity_score': lambda x: min(x + 0.5, 5.0),
            'decision_making_authority_indicator': lambda x: min(x + 1, 10),
            'time_with_current_manager': lambda x: 0.0,
            'tenure_in_current_role': lambda x: 0.0,
            'baseline_salary': 1.08,
            'compensation_percentile_company': lambda x: min(x + 8, 95),
        }

class BusinessIntelligence:
    """
    Business Intelligence module with causal intervention capabilities
    
    Uses existing engine infrastructure for consistent feature transformation
    and causal intervention simulation through strategic feature modification
    with realistic cascade effects.
    """
    
    def __init__(self, model_engine):
        """
        Initialize with existing engine infrastructure
        
        Args:
            model_engine: Trained SurvivalModelEngine with model_results
        """
        self.model_engine = model_engine
        self.config = InterventionConfig()
        self.causal_config = CausalConfig()
        
        # Use existing engine objects (zero duplication)
        self.feature_importance = model_engine.model_results.feature_importance
        self.feature_name_mapping = model_engine.model_results.feature_name_mapping
        self.feature_columns = model_engine.feature_columns
        
        # SHAP explainer initialization
        self.shap_explainer = shap.TreeExplainer(model_engine.model)
        
        # FeatureConfig-based classification
        self._classify_features_from_config()
        
        logger.info(f"BusinessIntelligence initialized with causal intervention capability")
    
    def _classify_features_from_config(self):
        """Feature modifiability classification using FeatureConfig patterns"""
        
        modifiable_raw_features = {
            'baseline_salary', 'salary_growth_rate_12m', 'team_avg_comp', 'peer_salary_ratio',
            'avg_salary_last_quarter', 'salary_growth_rate12m_to_cpi_rate',
            'job_level', 'career_stage', 'career_joiner_stage',
            'time_with_current_manager', 'manager_tenure_days', 'team_size',
            'assignment_frequency_12m', 'pay_frequency_preference'
        }
        
        self.feature_modifiability = {}
        for _, row in self.feature_importance.iterrows():
            original_feature = row['original_feature']
            is_modifiable = original_feature in modifiable_raw_features
            self.feature_modifiability[row['feature']] = 'modifiable' if is_modifiable else 'contextual'
    
    def analyze_individual_drivers(self, employee_data: pd.DataFrame) -> Dict:
        """
        Individual risk driver analysis using SHAP values
        
        Args:
            employee_data: Single employee record (1 row DataFrame)
            
        Returns:
            Dict: Risk profile with SHAP-ranked drivers and modifiability
        """
        if len(employee_data) != 1:
            raise ValueError("Employee data must contain exactly one record")
        
        risk_score = self.model_engine.predict_risk_scores(employee_data)[0]
        risk_category = self._categorize_risk_score(risk_score)
        
        X_processed = self.model_engine._get_processed_features(employee_data)
        shap_values = self.shap_explainer.shap_values(X_processed.values)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = shap_values.flatten()
        
        drivers = []
        for i, feature in enumerate(X_processed.columns):
            if i < len(shap_values):
                original_feature = self.feature_name_mapping.get(feature, feature)
                modifiability = self.feature_modifiability.get(feature, 'contextual')
                
                drivers.append({
                    'feature': feature,
                    'original_feature': original_feature,
                    'shap_value': float(shap_values[i]),
                    'current_value': X_processed.iloc[0, i],
                    'modifiability': modifiability,
                    'impact_rank': 0
                })
        
        drivers.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        for rank, driver in enumerate(drivers, 1):
            driver['impact_rank'] = rank
        
        return {
            'employee_profile': {
                'current_risk_score': float(risk_score),
                'risk_category': risk_category,
                'time_horizon_days': 365
            },
            'top_drivers': drivers[:20]
        }
    
    def _apply_causal_effects(self, employee_data: pd.DataFrame, effects_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
        """Apply causal cascade effects based on configuration"""
        modified_data = employee_data.copy()
        feature_changes = {}
        
        for feature, effect in effects_config.items():
            if feature in modified_data.columns:
                original_value = modified_data[feature].iloc[0]
                
                try:
                    if callable(effect):
                        new_value = effect(original_value)
                    else:
                        new_value = original_value * effect
                    
                    modified_data[feature].iloc[0] = new_value
                    feature_changes[feature] = {
                        'original': float(original_value),
                        'modified': float(new_value),
                        'effect_type': 'cascade'
                    }
                except (ValueError, TypeError):
                    continue
        
        return modified_data, feature_changes
    
    def _simulate_causal_salary_increase(self, employee_data: pd.DataFrame) -> Dict:
        """Causal salary intervention with realistic confounding adjustment"""
        
        modified_data, feature_changes = self._apply_causal_effects(
            employee_data, self.causal_config.salary_increase_effects
        )
        
        baseline_risk = self.model_engine.predict_risk_scores(employee_data)[0]
        causal_risk = self.model_engine.predict_risk_scores(modified_data)[0]
        
        risk_reduction = float(baseline_risk - causal_risk)
        new_category = self._categorize_risk_score(causal_risk)
        
        effect_magnitude = 'LARGE' if risk_reduction > 0.1 else 'MEDIUM' if risk_reduction > 0.05 else 'SMALL'
        
        return {
            'new_risk_score': float(causal_risk),
            'causal_risk_reduction': risk_reduction,
            'new_risk_category': new_category,
            'effect_magnitude': effect_magnitude,
            'causal_feature_changes': feature_changes,
            'features_affected': len(feature_changes),
            'intervention_type': 'causal_salary_increase'
        }
    
    def _simulate_causal_promotion(self, employee_data: pd.DataFrame) -> Dict:
        """Causal promotion intervention with realistic confounding adjustment"""
        
        modified_data, feature_changes = self._apply_causal_effects(
            employee_data, self.causal_config.promotion_effects
        )
        
        baseline_risk = self.model_engine.predict_risk_scores(employee_data)[0]
        causal_risk = self.model_engine.predict_risk_scores(modified_data)[0]
        
        risk_reduction = float(baseline_risk - causal_risk)
        new_category = self._categorize_risk_score(causal_risk)
        
        effect_magnitude = 'LARGE' if risk_reduction > 0.1 else 'MEDIUM' if risk_reduction > 0.05 else 'SMALL'
        
        return {
            'new_risk_score': float(causal_risk),
            'causal_risk_reduction': risk_reduction,
            'new_risk_category': new_category,
            'effect_magnitude': effect_magnitude,
            'causal_feature_changes': feature_changes,
            'features_affected': len(feature_changes),
            'intervention_type': 'causal_promotion'
        }
    
    def simulate_causal_intervention_scenarios(self, employee_data: pd.DataFrame, scenarios: List[str]) -> Dict:
        """
        Causal counterfactual intervention simulation with realistic confounding
        
        Args:
            employee_data: Single employee record (1 row DataFrame)
            scenarios: ['causal_salary_increase', 'causal_promotion']
            
        Returns:
            Dict: Baseline and causal intervention results
        """
        if len(employee_data) != 1:
            raise ValueError("Employee data must contain exactly one record")
        
        baseline_risk = self.model_engine.predict_risk_scores(employee_data)[0]
        baseline_category = self._categorize_risk_score(baseline_risk)
        
        results = {
            'baseline': {
                'risk_score': float(baseline_risk),
                'risk_category': baseline_category
            },
            'causal_interventions': {}
        }
        
        for scenario in scenarios:
            if scenario == 'causal_salary_increase':
                intervention_result = self._simulate_causal_salary_increase(employee_data)
            elif scenario == 'causal_promotion':
                intervention_result = self._simulate_causal_promotion(employee_data)
            else:
                raise ValueError(f"Unsupported causal scenario: {scenario}")
            
            results['causal_interventions'][scenario] = intervention_result
        
        return results
    
    def simulate_intervention_scenarios(self, employee_data: pd.DataFrame, scenarios: List[str]) -> Dict:
        """
        Legacy intervention simulation - kept for backwards compatibility
        """
        return self.simulate_causal_intervention_scenarios(employee_data, scenarios)
    
    def _analyze_risk_stratified_drivers(self, dataset: pd.DataFrame, risk_categories: List[str]) -> Dict:
        """Risk-stratified SHAP driver analysis"""
        
        risk_groups = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
        for i, category in enumerate(risk_categories):
            risk_groups[category].append(i)
        
        common_drivers = {}
        
        for category, indices in risk_groups.items():
            if not indices:
                common_drivers[category.lower() + '_risk'] = []
                continue
            
            sample_size = min(50, len(indices))
            sample_indices = np.random.choice(indices, sample_size, replace=False)
            sample_data = dataset.iloc[sample_indices]
            
            X_processed = self.model_engine._get_processed_features(sample_data)
            shap_values = self.shap_explainer.shap_values(X_processed.values)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
            
            top_drivers = []
            for idx in top_indices:
                if idx < len(X_processed.columns):
                    feature = X_processed.columns[idx]
                    original_feature = self.feature_name_mapping.get(feature, feature)
                    top_drivers.append((original_feature, float(mean_abs_shap[idx])))
            
            common_drivers[category.lower() + '_risk'] = top_drivers
        
        return common_drivers
    
    def _assess_causal_intervention_effectiveness(self, dataset: pd.DataFrame, risk_scores: np.ndarray, 
                                                risk_categories: List[str]) -> Dict:
        """Population causal intervention effectiveness assessment"""
        
        sample_size = min(200, len(dataset))
        sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
        sample_data = dataset.iloc[sample_indices]
        sample_risks = risk_scores[sample_indices]
        sample_categories = [risk_categories[i] for i in sample_indices]
        
        effectiveness = {
            'causal_salary_increase': self._evaluate_causal_salary_effectiveness(sample_data, sample_risks, sample_categories),
            'causal_promotion': self._evaluate_causal_promotion_effectiveness(sample_data, sample_risks, sample_categories)
        }
        
        return effectiveness
    
    def _evaluate_causal_salary_effectiveness(self, dataset: pd.DataFrame, risk_scores: np.ndarray, 
                                            risk_categories: List[str]) -> Dict:
        """Causal salary intervention effectiveness evaluation"""
        
        risk_reductions = []
        employees_benefiting = 0
        category_transitions = {'high_to_medium': 0, 'high_to_low': 0, 'medium_to_low': 0}
        
        for i, (idx, row) in enumerate(dataset.iterrows()):
            employee_data = pd.DataFrame([row])
            
            try:
                intervention_result = self._simulate_causal_salary_increase(employee_data)
                risk_reduction = intervention_result['causal_risk_reduction']
                
                if risk_reduction > self.config.meaningful_risk_reduction_threshold:
                    employees_benefiting += 1
                
                risk_reductions.append(risk_reduction)
                
                original_category = risk_categories[i]
                new_category = intervention_result['new_risk_category']
                
                transition_key = f"{original_category.lower()}_to_{new_category.lower()}"
                if transition_key in category_transitions:
                    category_transitions[transition_key] += 1
                    
            except Exception:
                risk_reductions.append(0.0)
        
        return {
            'employees_benefiting': employees_benefiting,
            'avg_risk_reduction': float(np.mean(risk_reductions)),
            'category_transitions': category_transitions
        }
    
    def _evaluate_causal_promotion_effectiveness(self, dataset: pd.DataFrame, risk_scores: np.ndarray, 
                                               risk_categories: List[str]) -> Dict:
        """Causal promotion intervention effectiveness evaluation"""
        
        risk_reductions = []
        employees_benefiting = 0
        category_transitions = {'high_to_medium': 0, 'high_to_low': 0, 'medium_to_low': 0}
        
        for i, (idx, row) in enumerate(dataset.iterrows()):
            employee_data = pd.DataFrame([row])
            
            try:
                intervention_result = self._simulate_causal_promotion(employee_data)
                risk_reduction = intervention_result['causal_risk_reduction']
                
                if risk_reduction > self.config.meaningful_risk_reduction_threshold:
                    employees_benefiting += 1
                
                risk_reductions.append(risk_reduction)
                
                original_category = risk_categories[i]
                new_category = intervention_result['new_risk_category']
                
                transition_key = f"{original_category.lower()}_to_{new_category.lower()}"
                if transition_key in category_transitions:
                    category_transitions[transition_key] += 1
                    
            except Exception:
                risk_reductions.append(0.0)
        
        return {
            'employees_benefiting': employees_benefiting,
            'avg_risk_reduction': float(np.mean(risk_reductions)),
            'category_transitions': category_transitions
        }
    
    def generate_population_insights(self, dataset: pd.DataFrame) -> Dict:
        """
        Population-level causal intervention effectiveness analysis
        
        Args:
            dataset: Population dataset for analysis
            
        Returns:
            Dict: Population summary and causal intervention effectiveness metrics
        """
        risk_scores = self.model_engine.predict_risk_scores(dataset)
        risk_categories = [self._categorize_risk_score(score) for score in risk_scores]
        
        total_employees = len(dataset)
        risk_distribution = {
            'high_risk_count': sum(1 for cat in risk_categories if cat == 'HIGH'),
            'medium_risk_count': sum(1 for cat in risk_categories if cat == 'MEDIUM'),
            'low_risk_count': sum(1 for cat in risk_categories if cat == 'LOW')
        }
        risk_distribution['high_risk_percentage'] = (risk_distribution['high_risk_count'] / total_employees) * 100
        
        common_drivers = self._analyze_risk_stratified_drivers(dataset, risk_categories)
        
        causal_intervention_effectiveness = self._assess_causal_intervention_effectiveness(dataset, risk_scores, risk_categories)
        
        return {
            'population_summary': {
                'total_employees': total_employees,
                'risk_distribution': risk_distribution
            },
            'common_drivers_by_risk': common_drivers,
            'causal_intervention_effectiveness': causal_intervention_effectiveness
        }
    
    def _categorize_risk_score(self, risk_score: float) -> str:
        """Risk score categorization with easily modifiable thresholds"""
        if risk_score >= HIGH_RISK_THRESHOLD:
            return 'HIGH'
        elif risk_score >= MEDIUM_RISK_THRESHOLD:
            return 'MEDIUM'
        else:
            return 'LOW'


if __name__ == "__main__":
    """
    BusinessIntelligence module demonstration with causal intervention capabilities
    """
    
    print("=== CAUSAL BUSINESS INTELLIGENCE MODULE ===")
    
    bi = BusinessIntelligence(engine)
    print(f"✓ BusinessIntelligence initialized with causal intervention capability")
    print(f"✓ Features: {len(bi.feature_importance)} ranked")
    print(f"✓ SHAP: {len(bi.feature_columns)} model features")
    print(f"✓ Causal: Business-logic cascade effects enabled")
    
    print(f"\n=== CAUSAL INTERVENTION SIMULATION ===")
    
    sample_employee = dataset_raw['val'].iloc[0:1]
    causal_results = bi.simulate_causal_intervention_scenarios(
        sample_employee, 
        ['causal_salary_increase', 'causal_promotion']
    )
    
    baseline = causal_results['baseline']
    print(f"Baseline: {baseline['risk_score']:.3f} ({baseline['risk_category']})")
    
    for scenario, results in causal_results['causal_interventions'].items():
        scenario_name = scenario.replace('causal_', '').replace('_', ' ').title()
        direction = "↓" if results['causal_risk_reduction'] > 0 else "↑"
        
        print(f"\n{scenario_name}:")
        print(f"  Impact: {direction} {abs(results['causal_risk_reduction']):6.3f} ({results['effect_magnitude']})")
        print(f"  Transition: {baseline['risk_category']} → {results['new_risk_category']}")
        print(f"  Features Affected: {results['features_affected']}")
        
        if results['causal_feature_changes']:
            print(f"  Key Changes:")
            for feature, change in list(results['causal_feature_changes'].items())[:3]:
                print(f"    {feature}: {change['original']:.1f} → {change['modified']:.1f}")
    
    print(f"\n=== CAUSAL POPULATION INSIGHTS ===")
    
    population_sample = dataset_raw['val'].head(300)
    population_results = bi.generate_population_insights(population_sample)
    
    summary = population_results['population_summary']
    distribution = summary['risk_distribution']
    
    print(f"Population Analysis:")
    print(f"  High Risk: {distribution['high_risk_count']:,} ({distribution['high_risk_percentage']:.1f}%)")
    
    effectiveness = population_results['causal_intervention_effectiveness']
    
    for intervention, impact in effectiveness.items():
        intervention_name = intervention.replace('causal_', '').replace('_', ' ').title()
        benefit_rate = impact['employees_benefiting'] / summary['total_employees'] * 100
        
        print(f"  {intervention_name}: {impact['employees_benefiting']:,} benefit ({benefit_rate:.1f}%)")
        print(f"    Avg Effect: {impact['avg_risk_reduction']:6.3f}")
    
    print(f"\n✓ CAUSAL INTERVENTION FRAMEWORK READY")
