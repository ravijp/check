"""
4_business_intelligence.py

Concise business intelligence module for survival analysis with SHAP-based feature attribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from dataclasses import dataclass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@dataclass
class InterventionEffect:
    ate: float
    confidence_interval: Tuple[float, float]
    p_value: float
    intervention_name: str


@dataclass
class RiskAnalysis:
    employee_id: str
    risk_score: float
    risk_category: str
    feature_contributions: Dict[str, float]
    top_risk_factors: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]


class CausalAnalyzer:
    def __init__(self, model_engine, n_bootstrap=100):
        self.model_engine = model_engine
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(42)
    
    def salary_intervention(self, X: pd.DataFrame) -> pd.DataFrame:
        X_int = X.copy()
        if 'current_salary' in X_int.columns:
            X_int['current_salary'] *= 1.15
        if 'comp_ratio' in X_int.columns:
            X_int['comp_ratio'] *= 1.15
        return X_int
    
    def promotion_intervention(self, X: pd.DataFrame) -> pd.DataFrame:
        X_int = X.copy()
        if 'job_level' in X_int.columns:
            X_int['job_level'] += 1
        if 'years_in_level' in X_int.columns:
            X_int['years_in_level'] = 0
        return X_int
    
    def estimate_effect(self, X: pd.DataFrame, intervention_func) -> InterventionEffect:
        risk_baseline = self.model_engine.predict_risk_scores(X)
        risk_intervention = self.model_engine.predict_risk_scores(intervention_func(X))
        ate = np.mean(risk_baseline - risk_intervention)
        
        bootstrap_ates = []
        for _ in range(self.n_bootstrap):
            idx = self.rng.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[idx]
            risk_base = self.model_engine.predict_risk_scores(X_boot)
            risk_int = self.model_engine.predict_risk_scores(intervention_func(X_boot))
            bootstrap_ates.append(np.mean(risk_base - risk_int))
        
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)
        t_stat = ate / (np.std(bootstrap_ates) + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        return InterventionEffect(ate, (ci_lower, ci_upper), p_value, intervention_func.__name__)


class FeatureAnalyzer:
    def __init__(self, model_engine):
        self.model_engine = model_engine
        self.explainer = None
        self.use_shap = SHAP_AVAILABLE
        
        if SHAP_AVAILABLE and hasattr(model_engine, 'xgb_model'):
            self.explainer = shap.TreeExplainer(model_engine.xgb_model)
    
    def get_contributions(self, employee_data: pd.DataFrame) -> Dict[str, float]:
        if self.use_shap and self.explainer:
            return self._shap_contributions(employee_data)
        else:
            return self._permutation_contributions(employee_data)
    
    def _shap_contributions(self, X: pd.DataFrame) -> Dict[str, float]:
        if hasattr(self.model_engine, 'feature_processor'):
            X_processed = self.model_engine.feature_processor.transform(X)
        else:
            X_processed = X
        
        shap_values = self.explainer.shap_values(X_processed)
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        return dict(zip(X_processed.columns, shap_values))
    
    def _permutation_contributions(self, X: pd.DataFrame) -> Dict[str, float]:
        baseline_risk = self.model_engine.predict_risk_scores(X)[0]
        contributions = {}
        
        for feature in X.columns:
            if feature in ['survival_time_days', 'event_indicator_vol']:
                continue
                
            X_perm = X.copy()
            if X[feature].dtype in ['float64', 'int64']:
                X_perm[feature] = X[feature].median()
            else:
                X_perm[feature] = X[feature].mode().iloc[0]
            
            perm_risk = self.model_engine.predict_risk_scores(X_perm)[0]
            contributions[feature] = baseline_risk - perm_risk
        
        return {k: v for k, v in contributions.items() if abs(v) > 1e-6}


class BusinessIntelligence:
    def __init__(self, model_engine):
        self.model_engine = model_engine
        self.causal_analyzer = CausalAnalyzer(model_engine)
        self.feature_analyzer = FeatureAnalyzer(model_engine)
        
        self.interventions = {
            'salary_increase': self.causal_analyzer.salary_intervention,
            'promotion': self.causal_analyzer.promotion_intervention
        }
    
    def analyze_population(self, data: pd.DataFrame) -> Dict[str, Any]:
        results = {
            'population_size': len(data),
            'average_risk': np.mean(self.model_engine.predict_risk_scores(data)),
            'interventions': {}
        }
        
        for name, intervention_func in self.interventions.items():
            effect = self.causal_analyzer.estimate_effect(data, intervention_func)
            results['interventions'][name] = {
                'ate': f"{effect.ate:.1%}",
                'confidence_interval': [f"{effect.confidence_interval[0]:.1%}", 
                                       f"{effect.confidence_interval[1]:.1%}"],
                'p_value': effect.p_value,
                'significant': effect.p_value < 0.05
            }
        
        return results
    
    def analyze_individual(self, employee_data: Union[pd.Series, pd.DataFrame]) -> RiskAnalysis:
        if isinstance(employee_data, pd.Series):
            employee_df = employee_data.to_frame().T
            employee_id = str(employee_data.name)
        else:
            employee_df = employee_data.copy()
            employee_id = str(employee_df.index[0])
        
        risk_score = self.model_engine.predict_risk_scores(employee_df)[0]
        contributions = self.feature_analyzer.get_contributions(employee_df)
        
        # Risk category
        if risk_score < 0.33:
            risk_category = "Low"
        elif risk_score < 0.67:
            risk_category = "Medium" 
        else:
            risk_category = "High"
        
        # Top risk factors
        risk_factors = []
        for feature, contrib in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
            if contrib > 0:
                risk_factors.append({
                    'feature': feature,
                    'value': employee_df[feature].iloc[0],
                    'contribution': contrib,
                    'modifiable': self._is_modifiable(feature)
                })
        
        # Recommendations
        recommendations = []
        for name, intervention_func in self.interventions.items():
            X_counterfactual = intervention_func(employee_df)
            counterfactual_risk = self.model_engine.predict_risk_scores(X_counterfactual)[0]
            risk_reduction = risk_score - counterfactual_risk
            
            if risk_reduction > 0.01:
                recommendations.append({
                    'intervention': name.replace('_', ' ').title(),
                    'expected_reduction': f"{risk_reduction:.1%}",
                    'priority': 'High' if risk_reduction > 0.05 else 'Medium'
                })
        
        return RiskAnalysis(
            employee_id=employee_id,
            risk_score=risk_score,
            risk_category=risk_category,
            feature_contributions=contributions,
            top_risk_factors=risk_factors,
            recommendations=recommendations
        )
    
    def analyze_cohort(self, data: pd.DataFrame, risk_threshold=0.7) -> Dict[str, Any]:
        risk_scores = self.model_engine.predict_risk_scores(data)
        high_risk_mask = risk_scores >= risk_threshold
        high_risk_data = data[high_risk_mask]
        
        if len(high_risk_data) == 0:
            return {'cohort_size': 0}
        
        cohort_interventions = {}
        for name, intervention_func in self.interventions.items():
            effect = self.causal_analyzer.estimate_effect(high_risk_data, intervention_func)
            cohort_interventions[name] = {
                'ate': f"{effect.ate:.1%}",
                'expected_retention': int(len(high_risk_data) * max(effect.ate, 0))
            }
        
        return {
            'cohort_size': len(high_risk_data),
            'percentage_of_population': len(high_risk_data) / len(data) * 100,
            'average_risk': np.mean(risk_scores[high_risk_mask]),
            'interventions': cohort_interventions
        }
    
    def _is_modifiable(self, feature: str) -> bool:
        non_modifiable = {'age', 'gender', 'hire_date', 'years_of_service', 'department'}
        return feature.lower() not in non_modifiable


if __name__ == "__main__":
    """Usage example with synthetic data"""
    
    class MockModelEngine:
        def __init__(self):
            self.feature_importance = pd.DataFrame({
                'feature': ['current_salary', 'performance_score', 'years_in_role', 'job_level'],
                'importance': [0.4, 0.3, 0.2, 0.1]
            })
        
        def predict_risk_scores(self, X):
            np.random.seed(hash(str(X.iloc[0].values)) % 2147483647)
            base_risk = 0.5
            
            if 'current_salary' in X.columns:
                salary_effect = -0.3 * (X['current_salary'] / 100000 - 0.75)
                base_risk += salary_effect.iloc[0] if hasattr(salary_effect, 'iloc') else salary_effect
            
            if 'performance_score' in X.columns:
                perf_effect = -0.2 * (X['performance_score'] / 5 - 0.6)
                base_risk += perf_effect.iloc[0] if hasattr(perf_effect, 'iloc') else perf_effect
            
            noise = np.random.normal(0, 0.1, len(X))
            risk_scores = np.clip(base_risk + noise, 0.1, 0.9)
            return risk_scores if len(X) > 1 else risk_scores[0]
    
    # Create test data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'current_salary': np.random.normal(75000, 20000, 1000),
        'performance_score': np.random.normal(3.5, 0.8, 1000),
        'years_in_role': np.random.exponential(2, 1000),
        'job_level': np.random.choice([1, 2, 3, 4, 5], 1000),
        'comp_ratio': np.random.normal(1.0, 0.15, 1000),
        'years_in_level': np.random.exponential(1.5, 1000),
        'age': np.random.normal(35, 8, 1000),
        'department': np.random.choice(['Eng', 'Sales', 'Marketing'], 1000)
    })
    
    test_data['current_salary'] = np.clip(test_data['current_salary'], 40000, 150000)
    test_data['performance_score'] = np.clip(test_data['performance_score'], 1, 5)
    
    # Initialize BI system
    mock_engine = MockModelEngine()
    bi = BusinessIntelligence(mock_engine)
    
    print("=== Business Intelligence Analysis ===")
    
    # Population analysis
    pop_results = bi.analyze_population(test_data.head(100))
    print(f"\nPopulation Analysis (n={pop_results['population_size']}):")
    print(f"Average Risk: {pop_results['average_risk']:.1%}")
    
    for intervention, results in pop_results['interventions'].items():
        significance = " ***" if results['significant'] else ""
        print(f"{intervention}: {results['ate']} (p={results['p_value']:.3f}){significance}")
    
    # Individual analysis
    sample_employee = test_data.iloc[0]
    individual_analysis = bi.analyze_individual(sample_employee)
    
    print(f"\nIndividual Analysis - Employee {individual_analysis.employee_id}:")
    print(f"Risk Score: {individual_analysis.risk_score:.1%} ({individual_analysis.risk_category})")
    
    print("Top Risk Factors:")
    for factor in individual_analysis.top_risk_factors[:3]:
        mod_flag = " (modifiable)" if factor['modifiable'] else ""
        print(f"  {factor['feature']}: {factor['value']:.2f}, impact: {factor['contribution']:+.3f}{mod_flag}")
    
    print("Recommendations:")
    for rec in individual_analysis.recommendations:
        print(f"  {rec['intervention']}: {rec['expected_reduction']} reduction ({rec['priority']} priority)")
    
    # High-risk cohort
    cohort_results = bi.analyze_cohort(test_data.head(200), risk_threshold=0.65)
    
    print(f"\nHigh-Risk Cohort Analysis:")
    print(f"Cohort Size: {cohort_results['cohort_size']} ({cohort_results['percentage_of_population']:.1f}%)")
    print(f"Average Risk: {cohort_results['average_risk']:.1%}")
    
    print("Cohort Interventions:")
    for intervention, results in cohort_results['interventions'].items():
        print(f"  {intervention}: {results['ate']}, potential retention: {results['expected_retention']} employees")
    
    # Feature contributions test
    contributions = bi.feature_analyzer.get_contributions(test_data.iloc[[0]])
    non_zero_contributions = sum(1 for v in contributions.values() if abs(v) > 1e-6)
    print(f"\nFeature Contributions: {non_zero_contributions} non-zero features")
    print(f"SHAP Available: {SHAP_AVAILABLE}")
    print(f"Using SHAP: {bi.feature_analyzer.use_shap}")
    
    print("\n=== Analysis Complete ===")
