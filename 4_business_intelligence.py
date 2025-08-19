"""
4_business_intelligence.py

Expert-level business intelligence module for employee turnover analysis.
Consolidates causal inference and driver analysis with SHAP-based attribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    raise ImportError("SHAP is required for this module. Install with: pip install shap")

logger = logging.getLogger(__name__)


@dataclass
class InterventionEffect:
    """Container for intervention effect estimates with analytical confidence intervals"""
    ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    ite_distribution: np.ndarray
    median_ite: float
    responders_pct: float
    p_value: float
    significant: bool
    sample_size: int


@dataclass 
class RiskDrivers:
    """Container for individual risk driver analysis results"""
    employee_id: str
    risk_score: float
    risk_category: str
    risk_factors: List[Dict[str, Union[str, float, bool]]]
    protective_factors: List[Dict[str, Union[str, float, bool]]]
    actionable_interventions: List[Dict[str, Union[str, float]]]
    shap_contributions: Dict[str, float]


class CausalInterventionAnalyzer:
    """
    G-computation based causal inference using domain-expert selected confounders.
    Implements analytical confidence intervals for computational efficiency.
    """
    
    # Domain-expert selected confounders (Option B)
    SALARY_CONFOUNDERS = [
        'age_at_vantage_win_cap', 'tenure_at_vantage_days_log', 'job_level',
        'baseline_salary_win_cap', 'naics_cd_encoded', 'gender_cd',
        'manager_changes_count_vantage', 'career_stage'
    ]
    
    PROMOTION_CONFOUNDERS = [
        'age_at_vantage_win_cap', 'tenure_at_vantage_days_log', 'job_level',
        'days_since_promot_log', 'num_promot_2yr_win_cap', 'naics_cd_encoded',
        'gender_cd', 'career_stage'
    ]
    
    def __init__(self, model_engine):
        """
        Initialize causal analyzer.
        
        Args:
            model_engine: Trained SurvivalModelEngine with AFT model
        """
        self.model_engine = model_engine
        
    def estimate_salary_intervention(self, X: pd.DataFrame, increase_pct: float = 0.15) -> InterventionEffect:
        """
        Estimate causal effect of salary increase using G-computation.
        
        Args:
            X: Employee data with confounders
            increase_pct: Salary increase percentage (default 15%)
            
        Returns:
            InterventionEffect with ATE, ITE distribution, and statistical inference
        """
        # Validate confounders are available
        available_confounders = [c for c in self.SALARY_CONFOUNDERS if c in X.columns]
        if len(available_confounders) < 5:
            raise ValueError(f"Insufficient confounders available. Need at least 5, got {len(available_confounders)}")
        
        # Factual outcomes (observed)
        risk_factual = self.model_engine.predict_risk_scores(X)
        
        # Counterfactual outcomes (intervention)
        X_intervention = self._apply_salary_intervention(X, increase_pct)
        risk_counterfactual = self.model_engine.predict_risk_scores(X_intervention)
        
        # Individual treatment effects
        ite_array = risk_factual - risk_counterfactual
        
        # Average treatment effect
        ate = np.mean(ite_array)
        
        # Analytical confidence intervals (faster than bootstrap)
        se_ate = np.std(ite_array) / np.sqrt(len(ite_array))
        t_critical = stats.t.ppf(0.975, df=len(ite_array)-1)
        ate_ci_lower = ate - t_critical * se_ate
        ate_ci_upper = ate + t_critical * se_ate
        
        # Statistical significance test
        t_stat = ate / (se_ate + 1e-10)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(ite_array)-1))
        
        # Business metrics
        median_ite = np.median(ite_array)
        responders_pct = (ite_array > 0.01).mean() * 100  # > 1% risk reduction
        
        return InterventionEffect(
            ate=ate,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            ite_distribution=ite_array,
            median_ite=median_ite,
            responders_pct=responders_pct,
            p_value=p_value,
            significant=p_value < 0.05,
            sample_size=len(X)
        )
    
    def estimate_promotion_intervention(self, X: pd.DataFrame) -> InterventionEffect:
        """
        Estimate causal effect of promotion using G-computation.
        
        Args:
            X: Employee data with confounders
            
        Returns:
            InterventionEffect with ATE, ITE distribution, and statistical inference
        """
        # Validate confounders are available
        available_confounders = [c for c in self.PROMOTION_CONFOUNDERS if c in X.columns]
        if len(available_confounders) < 5:
            raise ValueError(f"Insufficient confounders available. Need at least 5, got {len(available_confounders)}")
        
        # Factual outcomes (observed)
        risk_factual = self.model_engine.predict_risk_scores(X)
        
        # Counterfactual outcomes (intervention)
        X_intervention = self._apply_promotion_intervention(X)
        risk_counterfactual = self.model_engine.predict_risk_scores(X_intervention)
        
        # Individual treatment effects
        ite_array = risk_factual - risk_counterfactual
        
        # Average treatment effect
        ate = np.mean(ite_array)
        
        # Analytical confidence intervals
        se_ate = np.std(ite_array) / np.sqrt(len(ite_array))
        t_critical = stats.t.ppf(0.975, df=len(ite_array)-1)
        ate_ci_lower = ate - t_critical * se_ate
        ate_ci_upper = ate + t_critical * se_ate
        
        # Statistical significance test
        t_stat = ate / (se_ate + 1e-10)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(ite_array)-1))
        
        # Business metrics
        median_ite = np.median(ite_array)
        responders_pct = (ite_array > 0.01).mean() * 100
        
        return InterventionEffect(
            ate=ate,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            ite_distribution=ite_array,
            median_ite=median_ite,
            responders_pct=responders_pct,
            p_value=p_value,
            significant=p_value < 0.05,
            sample_size=len(X)
        )
    
    def _apply_salary_intervention(self, X: pd.DataFrame, increase_pct: float) -> pd.DataFrame:
        """
        Apply salary increase intervention with realistic feature transformations.
        
        Expert decision: Modify salary-related features with spillover effects.
        """
        X_int = X.copy()
        
        # Primary effect: Increase baseline salary
        if 'baseline_salary_win_cap' in X_int.columns:
            X_int['baseline_salary_win_cap'] = X_int['baseline_salary_win_cap'] * (1 + increase_pct)
        
        # Secondary effect: Update compensation percentiles (realistic organizational impact)
        if 'compensation_percentile_company' in X_int.columns:
            # Salary increase moves employee up in company percentile distribution
            current_pct = X_int['compensation_percentile_company']
            adjustment = increase_pct * 0.3  # Partial adjustment (not full increase)
            X_int['compensation_percentile_company'] = np.clip(current_pct + adjustment, 0, 1)
        
        # Tertiary effect: Team average compensation (spillover effect)
        if 'team_avg_comp' in X_int.columns:
            # Assume 10% of individual increase affects team average
            X_int['team_avg_comp'] = X_int['team_avg_comp'] * (1 + increase_pct * 0.1)
        
        return X_int
    
    def _apply_promotion_intervention(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply promotion intervention with realistic feature transformations.
        
        Expert decision: Modify promotion-related features with career progression effects.
        """
        X_int = X.copy()
        
        # Primary effect: Advance job level
        if 'job_level' in X_int.columns:
            # Handle string job levels ('1', '2', '3', etc.)
            current_levels = X_int['job_level'].astype(str)
            new_levels = []
            for level in current_levels:
                try:
                    new_level = str(int(level) + 1)
                except ValueError:
                    new_level = level  # Keep original if not numeric
                new_levels.append(new_level)
            X_int['job_level'] = new_levels
        
        # Secondary effect: Reset time since promotion (fresh promotion)
        if 'days_since_promot_log' in X_int.columns:
            # Set to log(30) representing 30 days since recent promotion
            X_int['days_since_promot_log'] = np.log(30)
        
        # Tertiary effect: Increment promotion count
        if 'num_promot_2yr_win_cap' in X_int.columns:
            X_int['num_promot_2yr_win_cap'] = X_int['num_promot_2yr_win_cap'] + 1
        
        # Quaternary effect: Update promotion indicator
        if 'promot_2yr_ind' in X_int.columns:
            X_int['promot_2yr_ind'] = 1  # Now has promotion in 2yr window
        
        return X_int


class IndividualDriverAnalyzer:
    """
    SHAP-based individual risk driver analysis optimized for business explanations.
    Uses TreeExplainer for exact feature attributions without fallback complexity.
    """
    
    # Business-friendly feature display names
    FEATURE_DISPLAY_NAMES = {
        'num_promot_2yr_win_cap': 'Recent Promotions (2yr)',
        'tenure_at_vantage_days_log': 'Tenure Length',  
        'days_since_promot_log': 'Time Since Last Promotion',
        'baseline_salary_win_cap': 'Salary Level',
        'age_at_vantage_win_cap': 'Employee Age',
        'manager_changes_count_vantage': 'Manager Changes',
        'naics_cd_encoded': 'Industry Type',
        'job_level': 'Job Level',
        'gender_cd': 'Gender',
        'career_stage': 'Career Stage'
    }
    
    # Actionable vs non-actionable factors
    ACTIONABLE_FEATURES = {
        'baseline_salary_win_cap', 'job_level', 'days_since_promot_log',
        'num_promot_2yr_win_cap', 'manager_changes_count_vantage'
    }
    
    def __init__(self, model_engine):
        """
        Initialize driver analyzer with SHAP TreeExplainer.
        
        Args:
            model_engine: Trained SurvivalModelEngine with XGBoost model
        """
        self.model_engine = model_engine
        
        # Initialize SHAP TreeExplainer
        if hasattr(model_engine, 'xgb_model') and model_engine.xgb_model is not None:
            self.explainer = shap.TreeExplainer(model_engine.xgb_model)
        else:
            raise ValueError("Model engine must have trained XGBoost model for SHAP analysis")
    
    def analyze_employee(self, employee_data: Union[pd.Series, pd.DataFrame], 
                        n_drivers: int = 8) -> RiskDrivers:
        """
        Analyze individual employee risk drivers using SHAP values.
        
        Args:
            employee_data: Single employee's feature data
            n_drivers: Number of top drivers to return
            
        Returns:
            RiskDrivers with risk factors, protective factors, and actionable interventions
        """
        # Convert to DataFrame format
        if isinstance(employee_data, pd.Series):
            employee_df = employee_data.to_frame().T
            employee_id = str(employee_data.name) if hasattr(employee_data, 'name') else 'Unknown'
        else:
            employee_df = employee_data.copy()
            employee_id = str(employee_df.index[0]) if len(employee_df) == 1 else 'Unknown'
        
        # Get risk score
        risk_score = self.model_engine.predict_risk_scores(employee_df)[0]
        risk_category = self._categorize_risk(risk_score)
        
        # Calculate SHAP values
        shap_contributions = self._calculate_shap_contributions(employee_df)
        
        # Separate risk factors (positive SHAP) and protective factors (negative SHAP)
        risk_factors = []
        protective_factors = []
        
        # Sort by absolute SHAP value for importance ranking
        sorted_features = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature, shap_value in sorted_features:
            if feature not in employee_df.columns:
                continue
                
            factor_info = {
                'feature': feature,
                'display_name': self.FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title()),
                'value': employee_df[feature].iloc[0],
                'shap_value': shap_value,
                'actionable': feature in self.ACTIONABLE_FEATURES
            }
            
            if shap_value > 0:  # Increases risk
                risk_factors.append(factor_info)
            else:  # Protective factor
                protective_factors.append(factor_info)
        
        # Generate actionable interventions
        actionable_interventions = self._generate_interventions(
            employee_df, risk_factors[:n_drivers], risk_score
        )
        
        return RiskDrivers(
            employee_id=employee_id,
            risk_score=risk_score,
            risk_category=risk_category,
            risk_factors=risk_factors[:n_drivers],
            protective_factors=protective_factors[:min(3, len(protective_factors))],
            actionable_interventions=actionable_interventions,
            shap_contributions=shap_contributions
        )
    
    def _calculate_shap_contributions(self, employee_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate SHAP contributions using TreeExplainer for exact attributions.
        
        Args:
            employee_df: Single employee processed features
            
        Returns:
            Dictionary of feature -> SHAP value mappings
        """
        # Process features through model pipeline
        X_processed = self.model_engine._get_processed_features(employee_df)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_processed)
        
        # Handle multi-output case (take first row if needed)
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Create feature -> SHAP value mapping
        shap_dict = dict(zip(X_processed.columns, shap_values))
        
        # Filter out negligible contributions for business clarity
        significant_contributions = {
            feature: value for feature, value in shap_dict.items()
            if abs(value) > 1e-4  # Only meaningful contributions
        }
        
        return significant_contributions
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level for business communication"""
        if risk_score < 0.33:
            return "Low Risk"
        elif risk_score < 0.67:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def _generate_interventions(self, employee_df: pd.DataFrame, 
                               top_risk_factors: List[Dict], 
                               current_risk: float) -> List[Dict[str, Union[str, float]]]:
        """
        Generate specific intervention recommendations based on top risk factors.
        
        Args:
            employee_df: Employee data
            top_risk_factors: Top risk factors from SHAP analysis
            current_risk: Current risk score
            
        Returns:
            List of actionable intervention recommendations
        """
        interventions = []
        
        # Analyze top actionable risk factors
        actionable_risks = [f for f in top_risk_factors if f['actionable']]
        
        for factor in actionable_risks[:3]:  # Top 3 actionable factors
            feature = factor['feature']
            shap_impact = factor['shap_value']
            
            # Generate feature-specific interventions
            if 'salary' in feature.lower():
                interventions.append({
                    'action': 'Salary Review & Adjustment',
                    'rationale': f"Salary level is driving {shap_impact:.1%} of turnover risk",
                    'expected_impact': f"{min(shap_impact * 0.7, 0.15):.1%}",  # Conservative estimate
                    'timeline': '30-60 days',
                    'priority': 'High' if shap_impact > 0.05 else 'Medium'
                })
                
            elif 'promot' in feature.lower():
                interventions.append({
                    'action': 'Career Development Discussion',
                    'rationale': f"Promotion timing is driving {shap_impact:.1%} of turnover risk",
                    'expected_impact': f"{min(shap_impact * 0.6, 0.12):.1%}",
                    'timeline': '15-30 days',
                    'priority': 'High' if shap_impact > 0.04 else 'Medium'
                })
                
            elif 'manager' in feature.lower():
                interventions.append({
                    'action': 'Manager Relationship Review',
                    'rationale': f"Management stability is driving {shap_impact:.1%} of turnover risk",
                    'expected_impact': f"{min(shap_impact * 0.5, 0.08):.1%}",
                    'timeline': '7-14 days',
                    'priority': 'High'
                })
                
            elif 'job_level' in feature.lower():
                interventions.append({
                    'action': 'Role Advancement Planning',
                    'rationale': f"Job level is driving {shap_impact:.1%} of turnover risk",
                    'expected_impact': f"{min(shap_impact * 0.8, 0.18):.1%}",
                    'timeline': '60-90 days',
                    'priority': 'High' if shap_impact > 0.06 else 'Medium'
                })
        
        # Sort by expected impact
        interventions.sort(key=lambda x: float(x['expected_impact'].rstrip('%')), reverse=True)
        
        return interventions[:3]  # Return top 3 interventions


class BusinessIntelligence:
    """
    Expert-level business intelligence framework integrating causal inference 
    and driver analysis for comprehensive turnover insights.
    """
    
    def __init__(self, model_engine):
        """
        Initialize business intelligence framework.
        
        Args:
            model_engine: Trained SurvivalModelEngine
        """
        self.model_engine = model_engine
        self.causal_analyzer = CausalInterventionAnalyzer(model_engine)
        self.driver_analyzer = IndividualDriverAnalyzer(model_engine)
    
    def analyze_population(self, data: pd.DataFrame, 
                          sample_size: Optional[int] = None) -> Dict[str, Union[str, float, Dict]]:
        """
        Analyze population-level intervention effects.
        
        Args:
            data: Employee dataset
            sample_size: Optional sample size limit for performance
            
        Returns:
            Population analysis results with intervention effects
        """
        # Sample for computational efficiency if needed
        if sample_size and len(data) > sample_size:
            analysis_data = data.sample(sample_size, random_state=42)
        else:
            analysis_data = data
        
        # Calculate risk distribution
        risk_scores = self.model_engine.predict_risk_scores(analysis_data)
        risk_distribution = {
            'high_risk_pct': (risk_scores >= 0.67).mean() * 100,
            'medium_risk_pct': ((risk_scores >= 0.33) & (risk_scores < 0.67)).mean() * 100,
            'low_risk_pct': (risk_scores < 0.33).mean() * 100
        }
        
        # Analyze interventions
        interventions = {}
        
        # Salary intervention
        try:
            salary_effect = self.causal_analyzer.estimate_salary_intervention(analysis_data)
            interventions['salary_increase_15pct'] = {
                'name': '15% Salary Increase',
                'ate': f"{salary_effect.ate:.1%}",
                'confidence_interval': f"[{salary_effect.ate_ci_lower:.1%}, {salary_effect.ate_ci_upper:.1%}]",
                'p_value': salary_effect.p_value,
                'significant': salary_effect.significant,
                'responders_pct': f"{salary_effect.responders_pct:.0f}%",
                'median_impact': f"{salary_effect.median_ite:.1%}"
            }
        except Exception as e:
            logger.warning(f"Salary intervention analysis failed: {e}")
            interventions['salary_increase_15pct'] = {'error': str(e)}
        
        # Promotion intervention
        try:
            promotion_effect = self.causal_analyzer.estimate_promotion_intervention(analysis_data)
            interventions['promotion'] = {
                'name': 'Job Level Promotion',
                'ate': f"{promotion_effect.ate:.1%}",
                'confidence_interval': f"[{promotion_effect.ate_ci_lower:.1%}, {promotion_effect.ate_ci_upper:.1%}]",
                'p_value': promotion_effect.p_value,
                'significant': promotion_effect.significant,
                'responders_pct': f"{promotion_effect.responders_pct:.0f}%",
                'median_impact': f"{promotion_effect.median_ite:.1%}"
            }
        except Exception as e:
            logger.warning(f"Promotion intervention analysis failed: {e}")
            interventions['promotion'] = {'error': str(e)}
        
        # Generate recommendations
        recommendations = self._generate_population_recommendations(interventions, risk_distribution)
        
        return {
            'population_size': len(analysis_data),
            'risk_distribution': risk_distribution,
            'interventions': interventions,
            'recommendations': recommendations
        }
    
    def analyze_individual(self, employee_data: Union[pd.Series, pd.DataFrame]) -> Dict[str, Union[str, float, List, Dict]]:
        """
        Comprehensive individual employee analysis.
        
        Args:
            employee_data: Single employee data
            
        Returns:
            Individual analysis results with drivers and interventions
        """
        # Perform driver analysis
        driver_results = self.driver_analyzer.analyze_employee(employee_data)
        
        # Format results for business consumption
        return {
            'employee_id': driver_results.employee_id,
            'risk_summary': {
                'score': f"{driver_results.risk_score:.1%}",
                'category': driver_results.risk_category
            },
            'top_risk_factors': [
                {
                    'name': factor['display_name'],
                    'value': factor['value'],
                    'impact': f"{factor['shap_value']:+.1%}",
                    'actionable': factor['actionable']
                }
                for factor in driver_results.risk_factors
            ],
            'protective_factors': [
                {
                    'name': factor['display_name'],
                    'value': factor['value'],
                    'impact': f"{abs(factor['shap_value']):.1%} protection"
                }
                for factor in driver_results.protective_factors
            ],
            'recommendations': driver_results.actionable_interventions
        }
    
    def analyze_high_risk_cohort(self, data: pd.DataFrame, 
                                risk_threshold: float = 0.67) -> Dict[str, Union[int, float, List, Dict]]:
        """
        Analyze high-risk employee cohort for targeted interventions.
        
        Args:
            data: Employee dataset
            risk_threshold: Risk threshold for high-risk classification
            
        Returns:
            High-risk cohort analysis with targeted recommendations
        """
        # Identify high-risk employees
        risk_scores = self.model_engine.predict_risk_scores(data)
        high_risk_mask = risk_scores >= risk_threshold
        high_risk_data = data[high_risk_mask]
        
        if len(high_risk_data) == 0:
            return {'cohort_size': 0, 'message': 'No high-risk employees identified'}
        
        # Analyze interventions for high-risk cohort
        cohort_interventions = {}
        
        try:
            # Salary intervention for high-risk cohort
            salary_effect = self.causal_analyzer.estimate_salary_intervention(high_risk_data)
            cohort_interventions['salary_increase'] = {
                'ate': f"{salary_effect.ate:.1%}",
                'expected_retention': int(len(high_risk_data) * max(salary_effect.ate, 0)),
                'cost_per_employee': '$7,500',  # 15% * $50k average
                'roi_estimate': f"${int(len(high_risk_data) * max(salary_effect.ate, 0) * 75000):,}"
            }
        except Exception as e:
            logger.warning(f"High-risk salary analysis failed: {e}")
        
        try:
            # Promotion intervention for high-risk cohort  
            promotion_effect = self.causal_analyzer.estimate_promotion_intervention(high_risk_data)
            cohort_interventions['promotion'] = {
                'ate': f"{promotion_effect.ate:.1%}",
                'expected_retention': int(len(high_risk_data) * max(promotion_effect.ate, 0)),
                'cost_per_employee': '$2,500',  # Promotion administrative costs
                'roi_estimate': f"${int(len(high_risk_data) * max(promotion_effect.ate, 0) * 75000):,}"
            }
        except Exception as e:
            logger.warning(f"High-risk promotion analysis failed: {e}")
        
        return {
            'cohort_size': len(high_risk_data),
            'percentage_of_population': len(high_risk_data) / len(data) * 100,
            'average_risk': np.mean(risk_scores[high_risk_mask]),
            'interventions': cohort_interventions
        }
    
    def _generate_population_recommendations(self, interventions: Dict, 
                                           risk_distribution: Dict) -> List[str]:
        """Generate business recommendations based on population analysis"""
        recommendations = []
        
        # High-risk population recommendations
        high_risk_pct = risk_distribution['high_risk_pct']
        if high_risk_pct > 15:
            recommendations.append(f"Urgent: {high_risk_pct:.0f}% of workforce is high-risk - implement immediate retention programs")
        elif high_risk_pct > 10:
            recommendations.append(f"Attention: {high_risk_pct:.0f}% of workforce is high-risk - targeted interventions recommended")
        
        # Intervention-specific recommendations
        for intervention_name, results in interventions.items():
            if 'error' in results:
                continue
                
            if results.get('significant', False):
                ate = results.get('ate', '0%')
                responders = results.get('responders_pct', '0%')
                recommendations.append(
                    f"{results['name']}: {ate} average risk reduction, {responders} of employees benefit"
                )
        
        # Risk distribution recommendations
        if risk_distribution['low_risk_pct'] > 60:
            recommendations.append("Strong retention foundation - focus on high-performers and succession planning")
        
        return recommendations[:5]  # Top 5 recommendations


if __name__ == "__main__":
    """
    Example usage demonstrating the consolidated business intelligence module.
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    print("="*80)
    print("EXPERT BUSINESS INTELLIGENCE MODULE - DEMONSTRATION")
    print("="*80)
    
    # Mock model engine for demonstration
    class MockSurvivalEngine:
        """Mock engine simulating your trained SurvivalModelEngine"""
        
        def __init__(self):
            # Simulate feature importance from your model
            self.feature_importance = pd.DataFrame({
                'feature': ['num_promot_2yr_win_cap', 'tenure_at_vantage_days_log', 
                           'days_since_promot_log', 'baseline_salary_win_cap',
                           'age_at_vantage_win_cap', 'manager_changes_count_vantage'],
                'importance': [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
            })
            
            # Mock XGBoost model for SHAP
            import xgboost as xgb
            
            # Create minimal training data to fit a real XGB model
            np.random.seed(42)
            n_samples = 1000
            X_train = pd.DataFrame({
                'num_promot_2yr_win_cap': np.random.poisson(1, n_samples),
                'tenure_at_vantage_days_log': np.random.normal(6, 1, n_samples),
                'days_since_promot_log': np.random.normal(5, 1.5, n_samples),
                'baseline_salary_win_cap': np.random.normal(75000, 15000, n_samples),
                'age_at_vantage_win_cap': np.random.normal(35, 8, n_samples),
                'manager_changes_count_vantage': np.random.poisson(0.5, n_samples)
            })
            
            # Create realistic survival outcomes
            risk_factors = (
                -0.3 * X_train['num_promot_2yr_win_cap'] +
                0.1 * X_train['days_since_promot_log'] +
                -0.2 * (X_train['baseline_salary_win_cap'] - 75000) / 15000 +
                0.15 * X_train['manager_changes_count_vantage']
            )
            y_train = np.random.exponential(365 * np.exp(-risk_factors))
            
            # Train real XGBoost model
            self.xgb_model = xgb.XGBRegressor(random_state=42, n_estimators=50)
            self.xgb_model.fit(X_train, y_train)
        
        def predict_risk_scores(self, X):
            """Simulate risk score predictions"""
            # Use actual XGB model for consistency with SHAP
            survival_times = self.xgb_model.predict(X)
            # Convert to risk scores (shorter survival = higher risk)
            risk_scores = 1 / (1 + np.exp((survival_times - 365) / 180))
            return np.clip(risk_scores, 0.05, 0.95)
        
        def _get_processed_features(self, X):
            """Mock feature processing - return as-is for demonstration"""
            return X
    
    # Create synthetic employee data matching your feature schema
    np.random.seed(42)
    n_employees = 500
    
    employee_data = pd.DataFrame({
        'employee_id': range(1, n_employees + 1),
        'num_promot_2yr_win_cap': np.random.poisson(1, n_employees),
        'tenure_at_vantage_days_log': np.random.normal(6, 1, n_employees),
        'days_since_promot_log': np.random.normal(5, 1.5, n_employees),
        'baseline_salary_win_cap': np.random.normal(75000, 15000, n_employees),
        'age_at_vantage_win_cap': np.random.normal(35, 8, n_employees),
        'manager_changes_count_vantage': np.random.poisson(0.5, n_employees),
        'naics_cd_encoded': np.random.choice([0, 1, 2, 3], n_employees),
        'gender_cd': np.random.choice([0, 1], n_employees),
        'job_level': np.random.choice(['1', '2', '3', '4', '5'], n_employees),
        'career_stage': np.random.choice([0, 1, 2], n_employees)
    })
    
    # Ensure realistic ranges
    employee_data['baseline_salary_win_cap'] = np.clip(employee_data['baseline_salary_win_cap'], 40000, 150000)
    employee_data['age_at_vantage_win_cap'] = np.clip(employee_data['age_at_vantage_win_cap'], 22, 65)
    employee_data['tenure_at_vantage_days_log'] = np.clip(employee_data['tenure_at_vantage_days_log'], 3, 9)
    
    # Initialize components
    mock_engine = MockSurvivalEngine()
    bi = BusinessIntelligence(mock_engine)
    
    print("\n1. POPULATION ANALYSIS")
    print("-" * 50)
    
    # Analyze population with sample
    pop_results = bi.analyze_population(employee_data, sample_size=200)
    
    print(f"Population Size: {pop_results['population_size']}")
    print(f"Risk Distribution:")
    print(f"  High Risk: {pop_results['risk_distribution']['high_risk_pct']:.1f}%")
    print(f"  Medium Risk: {pop_results['risk_distribution']['medium_risk_pct']:.1f}%")
    print(f"  Low Risk: {pop_results['risk_distribution']['low_risk_pct']:.1f}%")
    
    print(f"\nIntervention Effects:")
    for intervention, results in pop_results['interventions'].items():
        if 'error' not in results:
            significance = " ***" if results['significant'] else ""
            print(f"  {results['name']}: {results['ate']} (CI: {results['confidence_interval']}){significance}")
            print(f"    Responders: {results['responders_pct']}, p={results['p_value']:.3f}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(pop_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\n2. INDIVIDUAL ANALYSIS")
    print("-" * 50)
    
    # Select high-risk employee for analysis
    risk_scores = mock_engine.predict_risk_scores(employee_data)
    high_risk_idx = np.argmax(risk_scores)
    high_risk_employee = employee_data.iloc[high_risk_idx]
    
    print(f"Analyzing Employee ID: {high_risk_employee['employee_id']}")
    
    individual_results = bi.analyze_individual(high_risk_employee)
    
    print(f"Risk Assessment:")
    print(f"  Score: {individual_results['risk_summary']['score']}")
    print(f"  Category: {individual_results['risk_summary']['category']}")
    
    print(f"\nTop Risk Factors (SHAP Attribution):")
    for i, factor in enumerate(individual_results['top_risk_factors'][:4], 1):
        actionable = " (Actionable)" if factor['actionable'] else ""
        print(f"  {i}. {factor['name']}: {factor['value']}")
        print(f"     Impact: {factor['impact']}{actionable}")
    
    print(f"\nProtective Factors:")
    for i, factor in enumerate(individual_results['protective_factors'][:2], 1):
        print(f"  {i}. {factor['name']}: {factor['value']}")
        print(f"     Protection: {factor['impact']}")
    
    print(f"\nRecommended Interventions:")
    for i, rec in enumerate(individual_results['recommendations'], 1):
        print(f"  {i}. {rec['action']} ({rec['priority']} Priority)")
        print(f"     Expected Impact: {rec['expected_impact']}")
        print(f"     Timeline: {rec['timeline']}")
        print(f"     Rationale: {rec['rationale']}")
    
    print("\n3. HIGH-RISK COHORT ANALYSIS")
    print("-" * 50)
    
    # Analyze high-risk cohort
    cohort_results = bi.analyze_high_risk_cohort(employee_data, risk_threshold=0.65)
    
    if cohort_results['cohort_size'] > 0:
        print(f"High-Risk Cohort: {cohort_results['cohort_size']} employees")
        print(f"Population %: {cohort_results['percentage_of_population']:.1f}%")
        print(f"Average Risk: {cohort_results['average_risk']:.1%}")
        
        print(f"\nCohort Interventions:")
        for intervention, results in cohort_results['interventions'].items():
            print(f"  {intervention.replace('_', ' ').title()}:")
            print(f"    Risk Reduction: {results['ate']}")
            print(f"    Expected Retention: {results['expected_retention']} employees")
            print(f"    Cost per Employee: {results['cost_per_employee']}")
            print(f"    ROI Estimate: {results['roi_estimate']}")
    else:
        print("No high-risk employees identified with current threshold")
    
    print(f"\n4. TECHNICAL VALIDATION")
    print("-" * 50)
    
    # Validate SHAP integration
    try:
        sample_employee = employee_data.iloc[0]
        shap_values = bi.driver_analyzer._calculate_shap_contributions(sample_employee.to_frame().T)
        non_zero_shap = sum(1 for v in shap_values.values() if abs(v) > 1e-4)
        print(f"SHAP Integration: Working ✓")
        print(f"Feature Contributions: {non_zero_shap} non-zero attributions")
        print(f"TreeExplainer: Successfully initialized ✓")
    except Exception as e:
        print(f"SHAP Integration: Error - {e}")
    
    # Validate causal inference
    try:
        test_sample = employee_data.head(20)
        salary_effect = bi.causal_analyzer.estimate_salary_intervention(test_sample)
        print(f"Causal Inference: Working ✓")
        print(f"Salary Intervention ATE: {salary_effect.ate:.1%}")
        print(f"Statistical Significance: {salary_effect.significant}")
    except Exception as e:
        print(f"Causal Inference: Error - {e}")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("✓ Domain-expert confounder selection (Option B)")
    print("✓ SHAP TreeExplainer for exact feature attributions")
    print("✓ Analytical confidence intervals for efficiency")
    print("✓ Business-focused risk/protective factor separation")
    print("✓ Actionable intervention recommendations")
    print("✓ Expert-level statistical rigor with business clarity")
