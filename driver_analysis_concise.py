"""
driver_analysis_concise.py

Individual-level driver analysis for employee turnover risk.
Provides actionable explanations for UI display.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskDrivers:
    """Individual employee risk drivers for UI display"""
    employee_id: Any
    risk_score: float
    risk_category: str
    top_risk_factors: List[Dict[str, Any]]
    top_protective_factors: List[Dict[str, Any]]
    intervention_recommendations: List[Dict[str, Any]]


class IndividualDriverAnalyzer:
    """
    Analyzes individual-level drivers of turnover risk.
    Focused on actionable insights for HR practitioners.
    """
    
    # Define modifiable features (from actual feature list)
    MODIFIABLE_FEATURES = [
        'baseline_salary', 'salary_growth_rate_12m', 'compensation_percentile_company',
        'peer_salary_ratio', 'time_since_last_promotion', 'promotion_velocity',
        'role_complexity_score', 'team_size', 'time_with_current_manager',
        'manager_span_control', 'comp_chang_freq_per_year', 'pay_grade_stagnation_months'
    ]
    
    # Feature display names for UI
    FEATURE_DISPLAY_NAMES = {
        'baseline_salary': 'Current Salary',
        'tenure_at_vantage_days': 'Company Tenure',
        'age_at_vantage': 'Age',
        'time_since_last_promotion': 'Time Since Promotion',
        'compensation_percentile_company': 'Salary Percentile (Company)',
        'compensation_percentile_industry': 'Salary Percentile (Industry)',
        'peer_salary_ratio': 'Salary vs Peers',
        'job_level': 'Job Level',
        'team_size': 'Team Size',
        'salary_growth_rate_12m': '12-Month Salary Growth',
        'career_stage': 'Career Stage',
        'tenure_in_current_role': 'Time in Current Role',
        'time_with_current_manager': 'Time with Manager',
        'promotion_velocity': 'Promotion Speed',
        'team_avg_comp': 'Team Average Salary',
        'team_turnover_rate': 'Team Turnover Rate',
        'role_complexity_score': 'Role Complexity',
        'pay_grade_stagnation_months': 'Pay Grade Stagnation'
    }
    
    def __init__(self, model_engine, causal_analyzer=None):
        """
        Initialize driver analyzer.
        
        Args:
            model_engine: Trained SurvivalModelEngine
            causal_analyzer: Optional CausalInterventionAnalyzer
        """
        self.model_engine = model_engine
        self.causal_analyzer = causal_analyzer
        
    def analyze_employee(self, employee_data: pd.Series, 
                        n_drivers: int = 5) -> RiskDrivers:
        """
        Analyze turnover risk drivers for an individual employee.
        
        Args:
            employee_data: Single employee's features
            n_drivers: Number of top drivers to return
            
        Returns:
            RiskDrivers object with analysis results
        """
        # Get current risk
        employee_df = pd.DataFrame([employee_data])
        risk_score = self.model_engine.predict_risk_scores(employee_df)[0]
        risk_category = self._categorize_risk(risk_score)
        
        # Calculate feature contributions using permutation
        contributions = self._calculate_feature_contributions(employee_df)
        
        # Separate risk and protective factors
        risk_factors = []
        protective_factors = []
        
        for feature, contrib_value in contributions.items():
            factor = {
                'feature': feature,
                'display_name': self.FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title()),
                'value': employee_data.get(feature),
                'contribution': contrib_value,
                'modifiable': feature in self.MODIFIABLE_FEATURES
            }
            
            if contrib_value > 0:  # Increases risk
                risk_factors.append(factor)
            else:  # Decreases risk
                protective_factors.append(factor)
        
        # Sort by absolute contribution
        risk_factors.sort(key=lambda x: abs(x['contribution']), reverse=True)
        protective_factors.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Generate intervention recommendations
        recommendations = self._generate_recommendations(
            employee_df, risk_factors[:n_drivers], risk_score
        )
        
        return RiskDrivers(
            employee_id=employee_data.name if hasattr(employee_data, 'name') else 'Unknown',
            risk_score=risk_score,
            risk_category=risk_category,
            top_risk_factors=risk_factors[:n_drivers],
            top_protective_factors=protective_factors[:n_drivers],
            intervention_recommendations=recommendations
        )
    
    def _calculate_feature_contributions(self, employee_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature contributions using permutation approach.
        """
        baseline_risk = self.model_engine.predict_risk_scores(employee_df)[0]
        contributions = {}
        
        # Get feature importance from model if available
        if hasattr(self.model_engine, 'feature_importance'):
            important_features = self.model_engine.feature_importance['feature'].tolist()[:30]
        else:
            important_features = employee_df.columns.tolist()
        
        for feature in important_features:
            if feature not in employee_df.columns:
                continue
            if feature in ['survival_time_days', 'event_indicator_vol', 'dataset_split']:
                continue
            
            # Create permuted version
            permuted_df = employee_df.copy()
            
            # Replace with median/mode (simple baseline)
            if permuted_df[feature].dtype in ['float64', 'int64', 'float32', 'int32']:
                permuted_df[feature] = 0  # Neutral value
            else:
                permuted_df[feature] = 'Unknown'
            
            try:
                permuted_risk = self.model_engine.predict_risk_scores(permuted_df)[0]
                contribution = baseline_risk - permuted_risk
                contributions[feature] = contribution
            except:
                continue
        
        return contributions
    
    def _generate_recommendations(self, employee_df: pd.DataFrame,
                                 risk_factors: List[Dict],
                                 current_risk: float) -> List[Dict]:
        """
        Generate intervention recommendations based on risk factors.
        """
        recommendations = []
        
        # Check if salary-related factors are prominent
        salary_factors = [f for f in risk_factors 
                         if f['feature'] in ['baseline_salary', 'compensation_percentile_company', 
                                            'peer_salary_ratio', 'salary_growth_rate_12m']]
        
        if salary_factors and self.causal_analyzer:
            # Use causal analyzer if available
            try:
                salary_effect = self.causal_analyzer.estimate_salary_intervention(
                    employee_df, increase_pct=0.15, horizon=365
                )
                if salary_effect.ite_array[0] > 0:  # This employee would benefit
                    recommendations.append({
                        'intervention': '15% Salary Increase',
                        'expected_risk_reduction': f"{salary_effect.ite_array[0]:.1%}",
                        'confidence': 'High' if salary_effect.significant else 'Medium',
                        'priority': 1 if salary_effect.ite_array[0] > 0.05 else 2
                    })
            except:
                pass
        elif salary_factors:
            # Fallback without causal analyzer
            recommendations.append({
                'intervention': 'Review Compensation',
                'expected_risk_reduction': 'To be determined',
                'confidence': 'Medium',
                'priority': 2
            })
        
        # Check promotion-related factors
        promotion_factors = [f for f in risk_factors
                            if f['feature'] in ['time_since_last_promotion', 'promotion_velocity',
                                               'pay_grade_stagnation_months', 'job_level']]
        
        if promotion_factors and self.causal_analyzer:
            try:
                promotion_effect = self.causal_analyzer.estimate_promotion_intervention(
                    employee_df, horizon=365
                )
                if promotion_effect.ite_array[0] > 0:
                    recommendations.append({
                        'intervention': 'Promotion',
                        'expected_risk_reduction': f"{promotion_effect.ite_array[0]:.1%}",
                        'confidence': 'High' if promotion_effect.significant else 'Medium',
                        'priority': 1 if promotion_effect.ite_array[0] > 0.05 else 2
                    })
            except:
                pass
        elif promotion_factors:
            recommendations.append({
                'intervention': 'Career Development Review',
                'expected_risk_reduction': 'To be determined',
                'confidence': 'Medium',
                'priority': 2
            })
        
        # Check team/manager factors
        team_factors = [f for f in risk_factors
                       if f['feature'] in ['time_with_current_manager', 'team_turnover_rate',
                                          'team_size', 'manager_span_control']]
        
        if team_factors:
            recommendations.append({
                'intervention': 'Team/Manager Discussion',
                'expected_risk_reduction': 'Varies',
                'confidence': 'Low',
                'priority': 3
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into High/Medium/Low"""
        if risk_score >= 0.67:  # Top tercile
            return 'High'
        elif risk_score >= 0.33:  # Middle tercile
            return 'Medium'
        else:
            return 'Low'
    
    def format_for_ui(self, analysis: RiskDrivers) -> Dict:
        """
        Format analysis results for UI display.
        
        Returns:
            Dict ready for JSON serialization and UI rendering
        """
        def format_value(feature: str, value: Any) -> str:
            """Format feature value for display"""
            if value is None:
                return 'N/A'
            
            if 'salary' in feature.lower() or 'comp' in feature.lower():
                if isinstance(value, (int, float)) and not 'percentile' in feature:
                    return f"${value:,.0f}"
            elif 'days' in feature.lower() or 'tenure' in feature.lower():
                if isinstance(value, (int, float)):
                    return f"{value/365.25:.1f} years"
            elif 'percentile' in feature.lower() or 'ratio' in feature.lower():
                if isinstance(value, (int, float)):
                    return f"{value:.1f}"
            
            return str(value)
        
        return {
            'employee_id': analysis.employee_id,
            'risk_summary': {
                'score': f"{analysis.risk_score:.1%}",
                'category': analysis.risk_category,
                'color': {'High': 'red', 'Medium': 'yellow', 'Low': 'green'}[analysis.risk_category]
            },
            'top_risk_factors': [
                {
                    'name': factor['display_name'],
                    'value': format_value(factor['feature'], factor['value']),
                    'impact': 'High' if abs(factor['contribution']) > 0.05 else 'Medium',
                    'modifiable': factor['modifiable']
                }
                for factor in analysis.top_risk_factors
            ],
            'top_protective_factors': [
                {
                    'name': factor['display_name'],
                    'value': format_value(factor['feature'], factor['value']),
                    'impact': 'High' if abs(factor['contribution']) > 0.05 else 'Medium'
                }
                for factor in analysis.top_protective_factors
            ],
            'recommendations': [
                {
                    'action': rec['intervention'],
                    'expected_impact': rec['expected_risk_reduction'],
                    'confidence': rec['confidence'],
                    'priority': ['High', 'Medium', 'Low'][rec['priority'] - 1]
                }
                for rec in analysis.intervention_recommendations
            ]
        }


if __name__ == "__main__":
    """
    Test suite for IndividualDriverAnalyzer
    """
    print("=== INDIVIDUAL DRIVER ANALYZER TEST ===")
    
    # Mock model engine for testing
    class MockModelEngine:
        def predict_risk_scores(self, X):
            # Return realistic risk scores
            np.random.seed(hash(str(X.iloc[0].values)) % 2**32)
            return np.random.beta(2, 5, len(X))  # Skewed toward lower risk
        
        def predict_survival_curves(self, X, time_points):
            base_risk = self.predict_risk_scores(X)[0]
            n_times = len(time_points)
            curves = np.zeros((len(X), n_times))
            for i in range(len(X)):
                for j in range(n_times):
                    curves[i, j] = 1 - base_risk * (1 - np.exp(-0.002 * time_points[j]))
            return curves
    
    # Create test employee data
    test_employee = pd.Series({
        'employee_id': 'EMP123456',
        'baseline_salary': 65000,
        'age_at_vantage': 35,
        'tenure_at_vantage_days': 1095,  # 3 years
        'job_level': 4,
        'compensation_percentile_company': 45,
        'compensation_percentile_industry': 50,
        'time_since_last_promotion': 730,  # 2 years
        'promotion_velocity': 0.5,
        'career_stage': 'Mid',
        'tenure_in_current_role': 365,
        'time_with_current_manager': 180,
        'team_size': 8,
        'team_turnover_rate': 0.15,
        'peer_salary_ratio': 0.92,
        'salary_growth_rate_12m': 0.03,
        'role_complexity_score': 3.5,
        'pay_grade_stagnation_months': 18,
        'manager_span_control': 12,
        'gender_cd': 'F',
        'naics_cd': '52'
    }, name='EMP123456')
    
    # Initialize analyzer
    mock_engine = MockModelEngine()
    
    # Test without causal analyzer
    print("\n1. Testing without causal analyzer:")
    print("-" * 50)
    
    analyzer = IndividualDriverAnalyzer(mock_engine)
    analysis = analyzer.analyze_employee(test_employee)
    
    print(f"Employee: {analysis.employee_id}")
    print(f"Risk Score: {analysis.risk_score:.1%}")
    print(f"Risk Category: {analysis.risk_category}")
    
    print("\nTop Risk Factors:")
    for factor in analysis.top_risk_factors[:3]:
        print(f"  • {factor['display_name']}: {factor['value']}")
        print(f"    Contribution: {factor['contribution']:.3f}, Modifiable: {factor['modifiable']}")
    
    print("\nTop Protective Factors:")
    for factor in analysis.top_protective_factors[:3]:
        print(f"  • {factor['display_name']}: {factor['value']}")
        print(f"    Contribution: {factor['contribution']:.3f}")
    
    print("\nRecommendations:")
    for rec in analysis.intervention_recommendations:
        print(f"  • {rec['intervention']}: {rec['expected_risk_reduction']}")
        print(f"    Confidence: {rec['confidence']}, Priority: {rec['priority']}")
    
    # Test with causal analyzer
    print("\n2. Testing with causal analyzer:")
    print("-" * 50)
    
    from causal_inference_concise import CausalInterventionAnalyzer
    causal_analyzer = CausalInterventionAnalyzer(mock_engine, n_bootstrap=10)  # Small for testing
    
    analyzer_with_causal = IndividualDriverAnalyzer(mock_engine, causal_analyzer)
    analysis_with_causal = analyzer_with_causal.analyze_employee(test_employee)
    
    print("\nRecommendations with causal estimates:")
    for rec in analysis_with_causal.intervention_recommendations:
        print(f"  • {rec['intervention']}: {rec['expected_risk_reduction']}")
        print(f"    Confidence: {rec['confidence']}, Priority: {rec['priority']}")
    
    # Test UI formatting
    print("\n3. UI-formatted output:")
    print("-" * 50)
    
    ui_data = analyzer.format_for_ui(analysis)
    
    print(f"Risk Summary: {ui_data['risk_summary']}")
    print(f"Top Risk Factors: {ui_data['top_risk_factors'][:2]}")
    print(f"Recommendations: {ui_data['recommendations']}")
    
    print("\n=== TEST COMPLETED SUCCESSFULLY ===")
