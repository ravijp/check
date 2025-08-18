"""
business_intelligence_integrated.py

Integrated business intelligence module combining causal inference and driver analysis.
Replaces the placeholder 4_business_intelligence.py with working implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Import the concise modules
from causal_inference_concise import CausalInterventionAnalyzer, InterventionEffect
from driver_analysis_concise import IndividualDriverAnalyzer, RiskDrivers

logger = logging.getLogger(__name__)


class BusinessIntelligence:
    """
    Integrated business intelligence framework for survival analysis.
    Combines causal inference and driver analysis for actionable insights.
    """
    
    def __init__(self, model_engine, evaluation=None):
        """
        Initialize business intelligence framework.
        
        Args:
            model_engine: Trained SurvivalModelEngine instance
            evaluation: Optional SurvivalEvaluation instance
        """
        self.model_engine = model_engine
        self.evaluation = evaluation
        
        # Initialize components
        self.causal_analyzer = CausalInterventionAnalyzer(
            model_engine, n_bootstrap=100, confidence_level=0.95
        )
        self.driver_analyzer = IndividualDriverAnalyzer(
            model_engine, self.causal_analyzer
        )
        
        logger.info("BusinessIntelligence framework initialized")
    
    def analyze_population(self, X: pd.DataFrame, sample_size: int = None) -> Dict:
        """
        Analyze intervention effects at population level.
        
        Args:
            X: Population features
            sample_size: Optional sampling for large datasets
            
        Returns:
            Dict with population-level insights
        """
        # Sample if needed for performance
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} from {len(X)} employees")
        else:
            X_sample = X
        
        # Calculate intervention effects
        salary_effect = self.causal_analyzer.estimate_salary_intervention(X_sample)
        promotion_effect = self.causal_analyzer.estimate_promotion_intervention(X_sample)
        
        # Risk distribution
        risk_scores = self.model_engine.predict_risk_scores(X_sample)
        
        return {
            'population_size': len(X_sample),
            'risk_distribution': {
                'mean': float(np.mean(risk_scores)),
                'std': float(np.std(risk_scores)),
                'high_risk_pct': float(np.mean(risk_scores >= 0.67) * 100),
                'medium_risk_pct': float(np.mean((risk_scores >= 0.33) & (risk_scores < 0.67)) * 100),
                'low_risk_pct': float(np.mean(risk_scores < 0.33) * 100)
            },
            'interventions': {
                'salary_increase_15pct': self._format_intervention_effect(salary_effect, 'Salary Increase'),
                'promotion': self._format_intervention_effect(promotion_effect, 'Promotion')
            },
            'recommendations': self._generate_population_recommendations(
                risk_scores, salary_effect, promotion_effect
            )
        }
    
    def analyze_individual(self, employee_data: pd.Series) -> Dict:
        """
        Analyze individual employee with drivers and interventions.
        
        Args:
            employee_data: Single employee features
            
        Returns:
            Dict with individual analysis for UI display
        """
        # Get driver analysis
        analysis = self.driver_analyzer.analyze_employee(employee_data)
        
        # Format for UI
        ui_data = self.driver_analyzer.format_for_ui(analysis)
        
        # Add intervention details if available
        employee_df = pd.DataFrame([employee_data])
        
        try:
            salary_ite = self.causal_analyzer.estimate_salary_intervention(
                employee_df, horizon=365
            ).ite_array[0]
            
            promotion_ite = self.causal_analyzer.estimate_promotion_intervention(
                employee_df, horizon=365
            ).ite_array[0]
            
            ui_data['intervention_details'] = {
                'salary_increase': {
                    'risk_reduction': f"{salary_ite:.1%}",
                    'recommended': salary_ite > 0.05
                },
                'promotion': {
                    'risk_reduction': f"{promotion_ite:.1%}",
                    'recommended': promotion_ite > 0.05
                }
            }
        except Exception as e:
            logger.warning(f"Could not calculate individual intervention effects: {e}")
        
        return ui_data
    
    def analyze_high_risk_cohort(self, X: pd.DataFrame, 
                                 risk_threshold: float = 0.67) -> Dict:
        """
        Focused analysis on high-risk employees.
        
        Args:
            X: Employee features
            risk_threshold: Threshold for high risk
            
        Returns:
            Dict with high-risk cohort analysis
        """
        # Identify high-risk employees
        risk_scores = self.model_engine.predict_risk_scores(X)
        high_risk_mask = risk_scores >= risk_threshold
        high_risk_data = X[high_risk_mask]
        
        if len(high_risk_data) == 0:
            return {'message': 'No high-risk employees found'}
        
        # Analyze interventions for high-risk group
        salary_effect = self.causal_analyzer.estimate_salary_intervention(high_risk_data)
        promotion_effect = self.causal_analyzer.estimate_promotion_intervention(high_risk_data)
        
        # Identify common risk factors
        sample_size = min(100, len(high_risk_data))
        sample_indices = np.random.choice(len(high_risk_data), sample_size, replace=False)
        
        common_factors = {}
        for idx in sample_indices:
            employee = high_risk_data.iloc[idx]
            analysis = self.driver_analyzer.analyze_employee(employee, n_drivers=3)
            
            for factor in analysis.top_risk_factors:
                feature = factor['feature']
                if feature not in common_factors:
                    common_factors[feature] = 0
                common_factors[feature] += 1
        
        # Sort by frequency
        common_factors = sorted(common_factors.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'cohort_size': len(high_risk_data),
            'percentage_of_population': float(np.mean(high_risk_mask) * 100),
            'intervention_effectiveness': {
                'salary_increase': self._format_intervention_effect(salary_effect, 'Salary'),
                'promotion': self._format_intervention_effect(promotion_effect, 'Promotion')
            },
            'common_risk_factors': [
                {
                    'factor': self.driver_analyzer.FEATURE_DISPLAY_NAMES.get(f[0], f[0]),
                    'frequency_pct': float(f[1] / sample_size * 100)
                }
                for f in common_factors
            ],
            'priority_actions': self._prioritize_interventions(salary_effect, promotion_effect, len(high_risk_data))
        }
    
    def _format_intervention_effect(self, effect: InterventionEffect, name: str) -> Dict:
        """Format intervention effect for reporting"""
        return {
            'name': name,
            'ate': f"{effect.ate:.2%}",
            'confidence_interval': f"({effect.ate_ci_lower:.2%}, {effect.ate_ci_upper:.2%})",
            'responders_pct': f"{effect.responders_pct:.1f}%",
            'significant': effect.significant,
            'p_value': float(effect.p_value),
            'sample_size': effect.sample_size
        }
    
    def _generate_population_recommendations(self, risk_scores: np.ndarray,
                                           salary_effect: InterventionEffect,
                                           promotion_effect: InterventionEffect) -> List[str]:
        """Generate population-level recommendations"""
        recommendations = []
        
        high_risk_pct = np.mean(risk_scores >= 0.67) * 100
        
        if high_risk_pct > 20:
            recommendations.append(f"URGENT: {high_risk_pct:.1f}% of employees are high risk")
        
        if salary_effect.significant and salary_effect.ate > 0.05:
            recommendations.append(f"Salary increases could reduce turnover by {salary_effect.ate:.1%}")
        
        if promotion_effect.significant and promotion_effect.ate > 0.05:
            recommendations.append(f"Promotions could reduce turnover by {promotion_effect.ate:.1%}")
        
        if salary_effect.ate > promotion_effect.ate:
            recommendations.append("Prioritize compensation reviews over promotions")
        elif promotion_effect.ate > salary_effect.ate:
            recommendations.append("Prioritize career development over compensation")
        
        return recommendations
    
    def _prioritize_interventions(self, salary_effect: InterventionEffect,
                                 promotion_effect: InterventionEffect,
                                 cohort_size: int) -> List[Dict]:
        """Prioritize interventions based on effectiveness and cost"""
        actions = []
        
        # Estimate costs (placeholder values - should be configured)
        salary_cost_per_person = 10000  # 15% of ~$67k average
        promotion_cost_per_person = 5000  # Training, transition costs
        
        if salary_effect.significant and salary_effect.ate > 0.03:
            prevented_turnover = int(cohort_size * salary_effect.ate)
            actions.append({
                'action': 'Implement 15% salary increases',
                'expected_retention': prevented_turnover,
                'estimated_cost': salary_cost_per_person * cohort_size,
                'roi_estimate': 'Positive' if salary_effect.ate > 0.10 else 'Neutral',
                'priority': 1 if salary_effect.ate > 0.10 else 2
            })
        
        if promotion_effect.significant and promotion_effect.ate > 0.03:
            prevented_turnover = int(cohort_size * promotion_effect.ate)
            actions.append({
                'action': 'Fast-track promotions',
                'expected_retention': prevented_turnover,
                'estimated_cost': promotion_cost_per_person * cohort_size,
                'roi_estimate': 'Positive' if promotion_effect.ate > 0.08 else 'Neutral',
                'priority': 1 if promotion_effect.ate > 0.08 else 2
            })
        
        # Sort by priority
        actions.sort(key=lambda x: x['priority'])
        
        return actions


if __name__ == "__main__":
    """
    Test suite demonstrating business intelligence capabilities
    """
    print("=== BUSINESS INTELLIGENCE MODULE TEST ===")
    print("Demonstrating integration with survival model\n")
    
    # ===== 1. SETUP =====
    print("1. SETUP")
    print("-" * 50)
    
    # Mock model engine for testing (replace with actual in production)
    class MockModelEngine:
        def __init__(self):
            self.aft_parameters = type('obj', (object,), {
                'sigma': 1.5,
                'distribution': 'normal'
            })()
            
        def predict_risk_scores(self, X):
            np.random.seed(42)
            # Generate realistic risk distribution
            return np.random.beta(2, 5, len(X))
        
        def predict_survival_curves(self, X, time_points):
            risks = self.predict_risk_scores(X)
            n_times = len(time_points)
            curves = np.zeros((len(X), n_times))
            for i in range(len(X)):
                for j in range(n_times):
                    # Exponential decay based on risk
                    curves[i, j] = np.exp(-risks[i] * time_points[j] / 365)
            return curves
    
    # Create test data
    np.random.seed(42)
    n_employees = 1000
    
    test_data = pd.DataFrame({
        'employee_id': [f'EMP{i:05d}' for i in range(n_employees)],
        'baseline_salary': np.random.uniform(40000, 150000, n_employees),
        'age_at_vantage': np.random.uniform(22, 65, n_employees),
        'tenure_at_vantage_days': np.random.uniform(30, 3650, n_employees),
        'job_level': np.random.randint(1, 8, n_employees),
        'compensation_percentile_company': np.random.uniform(10, 90, n_employees),
        'time_since_last_promotion': np.random.uniform(0, 1825, n_employees),
        'promotion_velocity': np.random.uniform(0.1, 2.0, n_employees),
        'career_stage': np.random.choice(['Early', 'Mid', 'Senior'], n_employees),
        'team_turnover_rate': np.random.uniform(0, 0.3, n_employees),
        'role_complexity_score': np.random.uniform(1, 5, n_employees),
        'peer_salary_ratio': np.random.uniform(0.8, 1.2, n_employees)
    })
    
    # Initialize business intelligence
    mock_engine = MockModelEngine()
    bi = BusinessIntelligence(mock_engine)
    
    print("✓ BusinessIntelligence initialized")
    print(f"✓ Test data: {len(test_data)} employees")
    
    # ===== 2. POPULATION ANALYSIS =====
    print("\n2. POPULATION ANALYSIS")
    print("-" * 50)
    
    pop_analysis = bi.analyze_population(test_data, sample_size=500)
    
    print(f"Population size: {pop_analysis['population_size']}")
    print(f"Risk distribution:")
    print(f"  High risk: {pop_analysis['risk_distribution']['high_risk_pct']:.1f}%")
    print(f"  Medium risk: {pop_analysis['risk_distribution']['medium_risk_pct']:.1f}%")
    print(f"  Low risk: {pop_analysis['risk_distribution']['low_risk_pct']:.1f}%")
    
    print(f"\nIntervention effects:")
    for name, effect in pop_analysis['interventions'].items():
        print(f"  {effect['name']}:")
        print(f"    ATE: {effect['ate']}")
        print(f"    CI: {effect['confidence_interval']}")
        print(f"    Significant: {effect['significant']}")
    
    print(f"\nRecommendations:")
    for rec in pop_analysis['recommendations']:
        print(f"  • {rec}")
    
    # ===== 3. INDIVIDUAL ANALYSIS =====
    print("\n3. INDIVIDUAL ANALYSIS")
    print("-" * 50)
    
    # Select a high-risk employee
    risk_scores = mock_engine.predict_risk_scores(test_data)
    high_risk_idx = np.argmax(risk_scores)
    high_risk_employee = test_data.iloc[high_risk_idx]
    
    print(f"Analyzing employee: {high_risk_employee['employee_id']}")
    
    individual_analysis = bi.analyze_individual(high_risk_employee)
    
    print(f"Risk assessment:")
    print(f"  Score: {individual_analysis['risk_summary']['score']}")
    print(f"  Category: {individual_analysis['risk_summary']['category']}")
    
    print(f"\nTop risk factors:")
    for factor in individual_analysis['top_risk_factors'][:3]:
        print(f"  • {factor['name']}: {factor['value']} (Impact: {factor['impact']})")
    
    print(f"\nRecommended interventions:")
    for rec in individual_analysis['recommendations']:
        print(f"  • {rec['action']}")
        print(f"    Expected impact: {rec['expected_impact']}")
        print(f"    Priority: {rec['priority']}")
    
    # ===== 4. HIGH-RISK COHORT ANALYSIS =====
    print("\n4. HIGH-RISK COHORT ANALYSIS")
    print("-" * 50)
    
    cohort_analysis = bi.analyze_high_risk_cohort(test_data, risk_threshold=0.67)
    
    print(f"High-risk cohort size: {cohort_analysis['cohort_size']}")
    print(f"Percentage of population: {cohort_analysis['percentage_of_population']:.1f}%")
    
    print(f"\nIntervention effectiveness for high-risk group:")
    for name, effect in cohort_analysis['intervention_effectiveness'].items():
        print(f"  {effect['name']}: {effect['ate']} reduction")
    
    print(f"\nCommon risk factors in high-risk group:")
    for factor in cohort_analysis['common_risk_factors']:
        print(f"  • {factor['factor']}: {factor['frequency_pct']:.0f}% of sample")
    
    print(f"\nPriority actions:")
    for action in cohort_analysis['priority_actions']:
        print(f"  • {action['action']}")
        print(f"    Expected retention: {action['expected_retention']} employees")
        print(f"    Estimated cost: ${action['estimated_cost']:,}")
    
    # ===== 5. EXECUTIVE SUMMARY =====
    print("\n5. EXECUTIVE SUMMARY")
    print("-" * 50)
    
    print("KEY FINDINGS:")
    print(f"• {cohort_analysis['cohort_size']} employees at high risk")
    print(f"• Salary increases could reduce turnover by {pop_analysis['interventions']['salary_increase_15pct']['ate']}")
    print(f"• Promotions could reduce turnover by {pop_analysis['interventions']['promotion']['ate']}")
    
    if cohort_analysis['priority_actions']:
        top_action = cohort_analysis['priority_actions'][0]
        print(f"\nTOP RECOMMENDATION:")
        print(f"• {top_action['action']}")
        print(f"• Expected to retain {top_action['expected_retention']} employees")
    
    print("\n=== TEST COMPLETED SUCCESSFULLY ===")
    print("\nThis module is ready for integration with your trained survival model.")
    print("Replace MockModelEngine with your actual model_engine from survival_model_engine.py")
