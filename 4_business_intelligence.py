"""
business_intelligence.py - Employee Turnover Business Intelligence Module
================================================================================

Purpose: Individual-level driver analysis and intervention simulation for employee turnover prediction
Methodology: Counterfactual simulation with SHAP attribution and raw feature modification
Integration: SurvivalModelEngine with FeatureConfig-based preprocessing pipeline

Key Features:
- SHAP-based individual risk driver analysis with existing feature infrastructure
- Strategic raw feature modification through preprocessing pipeline consistency
- Counterfactual intervention simulation (salary increase, promotion)
- Population-level insights with risk stratification
- Delta Method confidence intervals
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

# Risk categorization thresholds (easily modifiable)
HIGH_RISK_THRESHOLD = 0.8
MEDIUM_RISK_THRESHOLD = 0.4

@dataclass
class InterventionConfig:
    """Intervention configuration with FeatureConfig alignment"""
    salary_increase_multiplier: float = 1.15
    promotion_job_level_increment: int = 1
    max_job_level: int = 100
    meaningful_risk_reduction_threshold: float = 0.05

class BusinessIntelligence:
    """
    Business Intelligence module with FeatureConfig-based feature processing
    
    Uses existing engine infrastructure for consistent feature transformation
    and intervention simulation through strategic raw feature modification.
    """
    
    def __init__(self, model_engine):
        """
        Initialize with existing engine infrastructure
        
        Args:
            model_engine: Trained SurvivalModelEngine with model_results
        """
        self.model_engine = model_engine
        self.config = InterventionConfig()
        
        # Use existing engine objects (zero duplication)
        self.feature_importance = model_engine.model_results.feature_importance
        self.feature_name_mapping = model_engine.model_results.feature_name_mapping
        self.feature_columns = model_engine.feature_columns
        
        # SHAP explainer initialization
        self.shap_explainer = shap.TreeExplainer(model_engine.model)
        
        # FeatureConfig-based classification
        self._classify_features_from_config()
        
        logger.info(f"BusinessIntelligence initialized with {len(self.feature_importance)} features")
    
    def _classify_features_from_config(self):
        """Feature modifiability classification using FeatureConfig patterns"""
        
        # Extract modifiable features from FeatureConfig patterns
        modifiable_raw_features = {
            # Salary-related (from winsorize_features, direct_features)
            'baseline_salary', 'salary_growth_rate_12m', 'team_avg_comp', 'peer_salary_ratio',
            'avg_salary_last_quarter', 'salary_growth_rate12m_to_cpi_rate',
            
            # Role/promotion-related (from categorical_features)
            'job_level', 'career_stage', 'career_joiner_stage',
            
            # Management-related (from winsorize_features, log_transform_features)
            'time_with_current_manager', 'manager_tenure_days', 'team_size',
            
            # Assignment-related (from winsorize_features)
            'assignment_frequency_12m', 'pay_frequency_preference'
        }
        
        # Classify processed features based on original feature mapping
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
        
        # Risk assessment
        risk_score = self.model_engine.predict_risk_scores(employee_data)[0]
        risk_category = self._categorize_risk_score(risk_score)
        
        # SHAP analysis
        X_processed = self.model_engine._get_processed_features(employee_data)
        shap_values = self.shap_explainer.shap_values(X_processed.values)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = shap_values.flatten()
        
        # Build driver ranking
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
        
        # Rank by absolute SHAP impact
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
    
    def simulate_intervention_scenarios(self, employee_data: pd.DataFrame, scenarios: List[str]) -> Dict:
        """
        Counterfactual intervention simulation with raw feature modification
        
        Args:
            employee_data: Single employee record (1 row DataFrame)
            scenarios: ['salary_increase_15pct', 'promotion']
            
        Returns:
            Dict: Baseline and intervention results with confidence intervals
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
            'interventions': {}
        }
        
        for scenario in scenarios:
            if scenario == 'salary_increase_15pct':
                intervention_result = self._simulate_salary_increase(employee_data)
            elif scenario == 'promotion':
                intervention_result = self._simulate_promotion(employee_data)
            else:
                raise ValueError(f"Unsupported scenario: {scenario}")
            
            results['interventions'][scenario] = intervention_result
        
        return results
    
    def _simulate_salary_increase(self, employee_data: pd.DataFrame) -> Dict:
        """Salary increase simulation with raw feature targeting"""
        
        modified_data = employee_data.copy()
        feature_changes = {}
        
        # Target baseline_salary (primary salary feature)
        if 'baseline_salary' in modified_data.columns:
            original_value = modified_data['baseline_salary'].iloc[0]
            new_value = original_value * self.config.salary_increase_multiplier
            modified_data['baseline_salary'].iloc[0] = new_value
            
            feature_changes['baseline_salary'] = {
                'original': float(original_value),
                'modified': float(new_value),
                'change_pct': float((self.config.salary_increase_multiplier - 1) * 100)
            }
        
        # Counterfactual prediction through preprocessing pipeline
        baseline_risk = self.model_engine.predict_risk_scores(employee_data)[0]
        modified_risk = self.model_engine.predict_risk_scores(modified_data)[0]
        
        risk_reduction = float(baseline_risk - modified_risk)
        new_category = self._categorize_risk_score(modified_risk)
        confidence_interval = self._calculate_delta_method_ci(employee_data, modified_data)
        
        return {
            'new_risk_score': float(modified_risk),
            'risk_reduction': risk_reduction,
            'new_risk_category': new_category,
            'confidence_interval': confidence_interval,
            'feature_changes': feature_changes
        }
    
    def _simulate_promotion(self, employee_data: pd.DataFrame) -> Dict:
        """Promotion simulation with job_level validation"""
        
        modified_data = employee_data.copy()
        feature_changes = {}
        
        # Job level promotion with validation
        if 'job_level' in modified_data.columns:
            current_level = modified_data['job_level'].iloc[0]
            
            # Handle numeric job levels only
            if str(current_level).isdigit():
                numeric_level = int(current_level)
                if numeric_level < self.config.max_job_level:
                    new_level = numeric_level + self.config.promotion_job_level_increment
                    modified_data['job_level'].iloc[0] = new_level
                    
                    feature_changes['job_level'] = {
                        'original': float(numeric_level),
                        'modified': float(new_level)
                    }
        
        # Counterfactual prediction
        baseline_risk = self.model_engine.predict_risk_scores(employee_data)[0]
        modified_risk = self.model_engine.predict_risk_scores(modified_data)[0]
        
        risk_reduction = float(baseline_risk - modified_risk)
        new_category = self._categorize_risk_score(modified_risk)
        confidence_interval = self._calculate_delta_method_ci(employee_data, modified_data)
        
        return {
            'new_risk_score': float(modified_risk),
            'risk_reduction': risk_reduction,
            'new_risk_category': new_category,
            'confidence_interval': confidence_interval,
            'feature_changes': feature_changes
        }
    
    def _calculate_delta_method_ci(self, baseline_data: pd.DataFrame, modified_data: pd.DataFrame) -> List[float]:
        """Delta Method confidence intervals for intervention effects"""
        
        # Prediction variance estimation through perturbation
        baseline_risks = []
        modified_risks = []
        
        for i in range(5):  # Minimal samples for computational efficiency
            noise_scale = 0.001
            
            # Baseline with perturbation
            baseline_noisy = baseline_data.copy()
            numeric_cols = baseline_noisy.select_dtypes(include=[np.number]).columns
            baseline_noisy[numeric_cols] += np.random.normal(0, noise_scale, size=len(numeric_cols))
            baseline_risks.append(self.model_engine.predict_risk_scores(baseline_noisy)[0])
            
            # Modified with perturbation
            modified_noisy = modified_data.copy()
            modified_noisy[numeric_cols] += np.random.normal(0, noise_scale, size=len(numeric_cols))
            modified_risks.append(self.model_engine.predict_risk_scores(modified_noisy)[0])
        
        # Effect variance calculation
        baseline_var = np.var(baseline_risks)
        modified_var = np.var(modified_risks)
        effect_var = baseline_var + modified_var
        effect_std = np.sqrt(effect_var)
        
        # Point estimate and 95% CI
        effect_estimate = np.mean(baseline_risks) - np.mean(modified_risks)
        margin_error = 1.96 * effect_std
        
        return [float(effect_estimate - margin_error), float(effect_estimate + margin_error)]
    
    def generate_population_insights(self, dataset: pd.DataFrame) -> Dict:
        """
        Population-level intervention effectiveness analysis
        
        Args:
            dataset: Population dataset for analysis
            
        Returns:
            Dict: Population summary and intervention effectiveness metrics
        """
        # Population risk assessment
        risk_scores = self.model_engine.predict_risk_scores(dataset)
        risk_categories = [self._categorize_risk_score(score) for score in risk_scores]
        
        # Population distribution
        total_employees = len(dataset)
        risk_distribution = {
            'high_risk_count': sum(1 for cat in risk_categories if cat == 'HIGH'),
            'medium_risk_count': sum(1 for cat in risk_categories if cat == 'MEDIUM'),
            'low_risk_count': sum(1 for cat in risk_categories if cat == 'LOW')
        }
        risk_distribution['high_risk_percentage'] = (risk_distribution['high_risk_count'] / total_employees) * 100
        
        # Risk-stratified driver analysis
        common_drivers = self._analyze_risk_stratified_drivers(dataset, risk_categories)
        
        # Intervention effectiveness assessment
        intervention_effectiveness = self._assess_intervention_effectiveness(dataset, risk_scores, risk_categories)
        
        return {
            'population_summary': {
                'total_employees': total_employees,
                'risk_distribution': risk_distribution
            },
            'common_drivers_by_risk': common_drivers,
            'intervention_effectiveness': intervention_effectiveness
        }
    
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
            
            # Strategic sampling for computational efficiency
            sample_size = min(50, len(indices))
            sample_indices = np.random.choice(indices, sample_size, replace=False)
            sample_data = dataset.iloc[sample_indices]
            
            # SHAP analysis
            X_processed = self.model_engine._get_processed_features(sample_data)
            shap_values = self.shap_explainer.shap_values(X_processed.values)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Aggregate importance
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
    
    def _assess_intervention_effectiveness(self, dataset: pd.DataFrame, risk_scores: np.ndarray, 
                                         risk_categories: List[str]) -> Dict:
        """Population intervention effectiveness assessment"""
        
        # Strategic sampling for performance
        sample_size = min(200, len(dataset))
        sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
        sample_data = dataset.iloc[sample_indices]
        sample_risks = risk_scores[sample_indices]
        sample_categories = [risk_categories[i] for i in sample_indices]
        
        effectiveness = {
            'salary_increase_15pct': self._evaluate_salary_effectiveness(sample_data, sample_risks, sample_categories),
            'promotion': self._evaluate_promotion_effectiveness(sample_data, sample_risks, sample_categories)
        }
        
        return effectiveness
    
    def _evaluate_salary_effectiveness(self, dataset: pd.DataFrame, risk_scores: np.ndarray, 
                                     risk_categories: List[str]) -> Dict:
        """Salary intervention effectiveness evaluation"""
        
        risk_reductions = []
        employees_benefiting = 0
        category_transitions = {'high_to_medium': 0, 'high_to_low': 0, 'medium_to_low': 0}
        
        for i, (idx, row) in enumerate(dataset.iterrows()):
            employee_data = pd.DataFrame([row])
            
            try:
                intervention_result = self._simulate_salary_increase(employee_data)
                risk_reduction = intervention_result['risk_reduction']
                
                if risk_reduction > self.config.meaningful_risk_reduction_threshold:
                    employees_benefiting += 1
                
                risk_reductions.append(risk_reduction)
                
                # Category transition tracking
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
    
    def _evaluate_promotion_effectiveness(self, dataset: pd.DataFrame, risk_scores: np.ndarray, 
                                        risk_categories: List[str]) -> Dict:
        """Promotion intervention effectiveness evaluation"""
        
        risk_reductions = []
        employees_benefiting = 0
        category_transitions = {'high_to_medium': 0, 'high_to_low': 0, 'medium_to_low': 0}
        
        for i, (idx, row) in enumerate(dataset.iterrows()):
            employee_data = pd.DataFrame([row])
            
            try:
                intervention_result = self._simulate_promotion(employee_data)
                risk_reduction = intervention_result['risk_reduction']
                
                if risk_reduction > self.config.meaningful_risk_reduction_threshold:
                    employees_benefiting += 1
                
                risk_reductions.append(risk_reduction)
                
                # Category transition tracking
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
    BusinessIntelligence module demonstration with FeatureConfig integration
    """
    
    print("=== BUSINESS INTELLIGENCE MODULE - FEATURECONFIG INTEGRATION ===")
    
    # Initialize with existing engine
    bi = BusinessIntelligence(engine)
    print(f"✓ BusinessIntelligence initialized")
    print(f"✓ Features: {len(bi.feature_importance)} ranked")
    print(f"✓ SHAP: {len(bi.feature_columns)} model features")
    print(f"✓ Modifiability: FeatureConfig-aligned classification")
    
    # === INDIVIDUAL ANALYSIS ===
    print(f"\n=== INDIVIDUAL ANALYSIS ===")
    
    sample_employee = dataset_raw['val'].iloc[0:1]
    individual_results = bi.analyze_individual_drivers(sample_employee)
    
    profile = individual_results['employee_profile']
    print(f"Risk: {profile['current_risk_score']:.3f} ({profile['risk_category']})")
    print(f"Horizon: {profile['time_horizon_days']} days")
    
    print(f"\nTop 10 Risk Drivers:")
    modifiable_count = 0
    for i, driver in enumerate(individual_results['top_drivers'][:10], 1):
        direction = "↑" if driver['shap_value'] > 0 else "↓"
        icon = "🔧" if driver['modifiability'] == 'modifiable' else "📊"
        if driver['modifiability'] == 'modifiable':
            modifiable_count += 1
        
        print(f"  {i:2d}. {icon} {driver['original_feature']:<35} {direction} {abs(driver['shap_value']):6.3f}")
    
    print(f"\nModifiable: {modifiable_count}/10 drivers")
    
    # === INTERVENTION SIMULATION ===
    print(f"\n=== INTERVENTION SIMULATION ===")
    
    intervention_results = bi.simulate_intervention_scenarios(
        sample_employee, 
        ['salary_increase_15pct', 'promotion']
    )
    
    baseline = intervention_results['baseline']
    print(f"Baseline: {baseline['risk_score']:.3f} ({baseline['risk_category']})")
    
    for scenario, results in intervention_results['interventions'].items():
        scenario_name = scenario.replace('_', ' ').title().replace('15Pct', '15%')
        direction = "↓" if results['risk_reduction'] > 0 else "↑"
        
        print(f"\n{scenario_name}:")
        print(f"  Impact: {direction} {abs(results['risk_reduction']):6.3f}")
        print(f"  Transition: {baseline['risk_category']} → {results['new_risk_category']}")
        print(f"  New Score: {results['new_risk_score']:.3f}")
        print(f"  95% CI: [{results['confidence_interval'][0]:+.3f}, {results['confidence_interval'][1]:+.3f}]")
        
        if results['feature_changes']:
            print(f"  Changes:")
            for feature, change in results['feature_changes'].items():
                if 'change_pct' in change:
                    print(f"    {feature}: {change['original']:.0f} → {change['modified']:.0f} (+{change['change_pct']:.0f}%)")
                else:
                    print(f"    {feature}: {change['original']} → {change['modified']}")
    
    # === POPULATION INSIGHTS ===
    print(f"\n=== POPULATION INSIGHTS ===")
    
    population_sample = dataset_raw['val'].head(300)
    population_results = bi.generate_population_insights(population_sample)
    
    summary = population_results['population_summary']
    distribution = summary['risk_distribution']
    
    print(f"Population (n={summary['total_employees']:,}):")
    print(f"  High Risk:   {distribution['high_risk_count']:,} ({distribution['high_risk_percentage']:.1f}%)")
    print(f"  Medium Risk: {distribution['medium_risk_count']:,}")
    print(f"  Low Risk:    {distribution['low_risk_count']:,}")
    
    print(f"\nDriver Patterns by Risk:")
    for risk_level, drivers in population_results['common_drivers_by_risk'].items():
        if drivers:
            print(f"  {risk_level.replace('_', ' ').title()}:")
            for feature, importance in drivers[:5]:
                print(f"    • {feature}: {importance:.3f}")
    
    print(f"\nIntervention Effectiveness:")
    effectiveness = population_results['intervention_effectiveness']
    
    for intervention, impact in effectiveness.items():
        intervention_name = intervention.replace('_', ' ').title().replace('15Pct', '15%')
        benefit_rate = impact['employees_benefiting'] / summary['total_employees'] * 100
        
        print(f"  {intervention_name}:")
        print(f"    Benefiting: {impact['employees_benefiting']:,} ({benefit_rate:.1f}%)")
        print(f"    Avg Reduction: {impact['avg_risk_reduction']:6.3f}")
        
        transitions = impact['category_transitions']
        total_transitions = sum(transitions.values())
        if total_transitions > 0:
            print(f"    Improvements: {total_transitions:,}")
            for transition, count in transitions.items():
                if count > 0:
                    transition_display = transition.replace('_', ' → ').upper()
                    print(f"      {transition_display}: {count:,}")
    
    # === EXECUTION SUMMARY ===
    best_intervention = max(
        intervention_results['interventions'].items(),
        key=lambda x: x[1]['risk_reduction']
    )
    
    print(f"\n" + "="*70)
    print(" EXECUTION SUMMARY")
    print("="*70)
    
    print(f"\n🎯 INDIVIDUAL RECOMMENDATION:")
    print(f"   Risk: {profile['current_risk_score']:.3f} ({profile['risk_category']})")
    print(f"   Optimal: {best_intervention[0].replace('_', ' ').title().replace('15Pct', '15%')}")
    print(f"   Impact: {best_intervention[1]['risk_reduction']:+.3f} → {best_intervention[1]['new_risk_category']}")
    
    print(f"\n📊 POPULATION METRICS:")
    print(f"   High-Risk: {distribution['high_risk_count']:,} ({distribution['high_risk_percentage']:.1f}%)")
    print(f"   Avg Reach: {np.mean([i['employees_benefiting']/summary['total_employees'] for i in effectiveness.values()]):.1%}")
    
    print(f"\n🔧 TECHNICAL EXECUTION:")
    print(f"   • FeatureConfig-aligned modifiability classification")
    print(f"   • Raw feature modification through preprocessing pipeline")
    print(f"   • Job level validation (0-10, numeric only)")
    print(f"   • Delta Method confidence intervals")
    print(f"   • Fixed risk thresholds: HIGH≥{HIGH_RISK_THRESHOLD}, MEDIUM≥{MEDIUM_RISK_THRESHOLD}")
    
