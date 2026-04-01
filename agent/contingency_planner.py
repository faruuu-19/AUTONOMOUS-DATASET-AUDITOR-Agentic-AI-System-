"""
agent/contingency_planner.py - Intelligent Contingency Planning

POINT 4: CONTINGENCY PLANNING
This module implements pre-planned intelligent reactions:
- "IF leakage detected THEN skip spurious + run extra leakage checks"
- "IF severe imbalance THEN boost contamination + use stratified sampling"
- "IF multiple criticals THEN switch to conservative mode"
- Automated remediation suggestions
- Risk-based escalation

The system plans ahead for common scenarios and reacts intelligently.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class TriggerCondition(Enum):
    """Conditions that trigger contingency plans"""
    LEAKAGE_DETECTED = "leakage_detected"
    SEVERE_IMBALANCE = "severe_class_imbalance"
    CONTAMINATION_FOUND = "contamination_detected"
    MULTIPLE_CRITICAL = "multiple_critical_issues"
    HIGH_SPURIOUS_CORRELATION = "high_spurious_correlation"
    LOW_FEATURE_QUALITY = "low_feature_quality"
    NO_ISSUES_FOUND = "no_issues_found_yet"
    TIME_CONSTRAINT = "approaching_time_limit"


@dataclass
class ContingencyPlan:
    """A pre-planned response to a specific scenario"""
    trigger: TriggerCondition
    priority: int  # Higher = more important
    
    # Actions to take
    skip_tools: List[str]
    boost_tools: List[str]
    add_extra_checks: List[str]
    
    # Strategy changes
    change_goal: Optional[str]
    adjust_thresholds: Dict[str, float]
    
    # Recommendations
    user_recommendations: List[str]
    auto_remediation: Optional[str]
    
    # Metadata
    reasoning: str
    confidence: float


class ContingencyPlanner:
    """
    INTELLIGENT CONTINGENCY PLANNING ENGINE
    
    Plans ahead for common scenarios and reacts intelligently:
    - Pre-defined playbooks for common issues
    - Automated escalation logic
    - Smart remediation suggestions
    - Dynamic workflow adjustments
    """
    
    def __init__(self):
        self.active_plans: List[ContingencyPlan] = []
        self.triggered_plans: List[str] = []
        self.contingency_library = self._initialize_contingency_library()
        
    def _initialize_contingency_library(self) -> Dict[TriggerCondition, ContingencyPlan]:
        """
        Initialize library of pre-planned contingencies.
        
        These are intelligent "IF-THEN" rules based on domain expertise.
        """
        library = {}
        
        # CONTINGENCY 1: Data Leakage Detected
        library[TriggerCondition.LEAKAGE_DETECTED] = ContingencyPlan(
            trigger=TriggerCondition.LEAKAGE_DETECTED,
            priority=10,  # Highest priority
            skip_tools=['spurious_detector'],  # Leakage explains spurious correlations
            boost_tools=['bias_detector', 'feature_utility'],
            add_extra_checks=['temporal_leakage_check', 'correlation_matrix_review'],
            change_goal='conservative_validation',
            adjust_thresholds={'skip_threshold': 0.6},  # Be more selective
            user_recommendations=[
                "CRITICAL: Remove leaky features before any model training",
                "Verify that leaky features wouldn't be available at prediction time",
                "Consider temporal split validation to catch time-based leakage",
                "Review feature engineering pipeline for data snooping"
            ],
            auto_remediation="generate_leakage_removal_script",
            reasoning="Data leakage invalidates model - must fix before proceeding",
            confidence=0.95
        )
        
        # CONTINGENCY 2: Severe Class Imbalance
        library[TriggerCondition.SEVERE_IMBALANCE] = ContingencyPlan(
            trigger=TriggerCondition.SEVERE_IMBALANCE,
            priority=8,
            skip_tools=[],  # Don't skip anything
            boost_tools=['contamination_detector'],  # Imbalanced data prone to leaks
            add_extra_checks=['stratification_check', 'sampling_bias_analysis'],
            change_goal=None,
            adjust_thresholds={},
            user_recommendations=[
                "Use stratified sampling for train-test split",
                "Consider SMOTE, ADASYN, or class weights",
                "Use stratified K-fold cross-validation",
                "Choose metrics appropriate for imbalanced data (F1, AUC-ROC)",
                "Check if imbalance reflects true population distribution"
            ],
            auto_remediation="generate_resampling_code",
            reasoning="Severe imbalance affects model training and evaluation",
            confidence=0.90
        )
        
        # CONTINGENCY 3: Multiple Critical Issues
        library[TriggerCondition.MULTIPLE_CRITICAL] = ContingencyPlan(
            trigger=TriggerCondition.MULTIPLE_CRITICAL,
            priority=9,
            skip_tools=[],  # Run everything to document all issues
            boost_tools=['feature_utility'],  # Understand data quality
            add_extra_checks=['comprehensive_audit', 'cross_validation'],
            change_goal='deep_investigation',  # Switch to thorough mode
            adjust_thresholds={'skip_threshold': 0.2, 'critical_threshold': 0.5},
            user_recommendations=[
                "Dataset has multiple serious issues - recommend data quality review",
                "Consider data collection process audit",
                "May need significant data cleaning before modeling",
                "Document all issues for stakeholder review"
            ],
            auto_remediation="generate_data_quality_report",
            reasoning="Multiple critical issues suggest fundamental data problems",
            confidence=0.85
        )
        
        # CONTINGENCY 4: Contamination Detected
        library[TriggerCondition.CONTAMINATION_FOUND] = ContingencyPlan(
            trigger=TriggerCondition.CONTAMINATION_FOUND,
            priority=9,
            skip_tools=[],
            boost_tools=['leakage_detector'],  # Check if leakage caused contamination
            add_extra_checks=['duplicate_analysis', 'train_test_overlap_check'],
            change_goal=None,
            adjust_thresholds={},
            user_recommendations=[
                "CRITICAL: Remove duplicate samples from test set",
                "Re-split data with proper randomization",
                "Verify no information leakage during split",
                "Use scikit-learn's train_test_split with shuffle=True"
            ],
            auto_remediation="generate_deduplication_script",
            reasoning="Train-test contamination invalidates evaluation",
            confidence=0.95
        )
        
        # CONTINGENCY 5: High Spurious Correlations
        library[TriggerCondition.HIGH_SPURIOUS_CORRELATION] = ContingencyPlan(
            trigger=TriggerCondition.HIGH_SPURIOUS_CORRELATION,
            priority=7,
            skip_tools=[],
            boost_tools=['feature_utility', 'bias_detector'],
            add_extra_checks=['feature_interaction_analysis', 'domain_validation'],
            change_goal=None,
            adjust_thresholds={},
            user_recommendations=[
                "High accuracy from single features suggests shortcut learning",
                "Validate features against domain knowledge",
                "Test model on out-of-distribution data",
                "Consider ablation studies to understand feature importance",
                "Check for confounding variables"
            ],
            auto_remediation="generate_feature_analysis_report",
            reasoning="Spurious correlations won't generalize to new data",
            confidence=0.75
        )
        
        # CONTINGENCY 6: Low Feature Quality
        library[TriggerCondition.LOW_FEATURE_QUALITY] = ContingencyPlan(
            trigger=TriggerCondition.LOW_FEATURE_QUALITY,
            priority=5,
            skip_tools=[],
            boost_tools=[],
            add_extra_checks=['feature_engineering_suggestions'],
            change_goal=None,
            adjust_thresholds={},
            user_recommendations=[
                "Many low-variance or redundant features detected",
                "Consider feature selection (SelectKBest, RFE)",
                "Remove constant or near-constant features",
                "Apply PCA or feature extraction if appropriate",
                "Review feature engineering process"
            ],
            auto_remediation="generate_feature_selection_code",
            reasoning="Low quality features waste computation and may harm model",
            confidence=0.70
        )
        
        # CONTINGENCY 7: No Issues Found (so far)
        library[TriggerCondition.NO_ISSUES_FOUND] = ContingencyPlan(
            trigger=TriggerCondition.NO_ISSUES_FOUND,
            priority=3,
            skip_tools=[],
            boost_tools=['spurious_detector'],  # Look harder for hidden issues
            add_extra_checks=['sensitivity_analysis'],
            change_goal='comprehensive_scan',  # Be more thorough
            adjust_thresholds={'skip_threshold': 0.3},  # Lower threshold = run more tools
            user_recommendations=[
                "No obvious issues found - running comprehensive checks",
                "Consider additional domain-specific validation",
                "Validate assumptions about data distribution",
                "Test model robustness with cross-validation"
            ],
            auto_remediation=None,
            reasoning="Clean data still needs thorough validation",
            confidence=0.60
        )
        
        # CONTINGENCY 8: Time Constraint
        library[TriggerCondition.TIME_CONSTRAINT] = ContingencyPlan(
            trigger=TriggerCondition.TIME_CONSTRAINT,
            priority=6,
            skip_tools=['feature_utility'],  # Skip low-priority checks
            boost_tools=[],
            add_extra_checks=[],
            change_goal='quick_validation',
            adjust_thresholds={'skip_threshold': 0.55},  # Be more selective
            user_recommendations=[
                "Time limit approaching - focusing on critical checks only",
                "Consider running full audit offline for comprehensive results"
            ],
            auto_remediation=None,
            reasoning="Time constraints require prioritization",
            confidence=0.80
        )
        
        return library
    
    def evaluate_triggers(self, current_findings: Dict[str, List],
                         tools_executed: List[str],
                         time_used: float,
                         time_budget: float,
                         goal_state: Any) -> List[ContingencyPlan]:
        """
        Evaluate which contingency plans should be triggered.
        
        Returns list of plans that should be activated.
        """
        triggered_plans = []
        
        # Check each trigger condition
        
        # 1. Leakage detected?
        leakage_findings = self._get_findings_by_tool(current_findings, 'leakage_detector')
        if self._has_critical_findings(leakage_findings):
            plan = self.contingency_library[TriggerCondition.LEAKAGE_DETECTED]
            if plan.trigger.value not in self.triggered_plans:
                triggered_plans.append(plan)
                self.triggered_plans.append(plan.trigger.value)
        
        # 2. Severe imbalance?
        bias_findings = self._get_findings_by_tool(current_findings, 'bias_detector')
        if any(f.get('type') == 'severe_imbalance' for f in bias_findings):
            plan = self.contingency_library[TriggerCondition.SEVERE_IMBALANCE]
            if plan.trigger.value not in self.triggered_plans:
                triggered_plans.append(plan)
                self.triggered_plans.append(plan.trigger.value)
        
        # 3. Contamination?
        contam_findings = self._get_findings_by_tool(current_findings, 'contamination_detector')
        if self._has_critical_findings(contam_findings):
            plan = self.contingency_library[TriggerCondition.CONTAMINATION_FOUND]
            if plan.trigger.value not in self.triggered_plans:
                triggered_plans.append(plan)
                self.triggered_plans.append(plan.trigger.value)
        
        # 4. Multiple critical issues?
        total_critical = sum(
            sum(1 for f in findings if f.get('severity') == 'critical')
            for findings in current_findings.values()
        )
        if total_critical >= 3:
            plan = self.contingency_library[TriggerCondition.MULTIPLE_CRITICAL]
            if plan.trigger.value not in self.triggered_plans:
                triggered_plans.append(plan)
                self.triggered_plans.append(plan.trigger.value)
        
        # 5. High spurious correlations?
        spurious_findings = self._get_findings_by_tool(current_findings, 'spurious_detector')
        if len(spurious_findings) >= 10:  # Many spurious correlations
            plan = self.contingency_library[TriggerCondition.HIGH_SPURIOUS_CORRELATION]
            if plan.trigger.value not in self.triggered_plans:
                triggered_plans.append(plan)
                self.triggered_plans.append(plan.trigger.value)
        
        # 6. Low feature quality?
        utility_findings = self._get_findings_by_tool(current_findings, 'feature_utility')
        if len(utility_findings) > len(tools_executed) * 3:  # Many quality issues
            plan = self.contingency_library[TriggerCondition.LOW_FEATURE_QUALITY]
            if plan.trigger.value not in self.triggered_plans:
                triggered_plans.append(plan)
                self.triggered_plans.append(plan.trigger.value)
        
        # 7. No issues found?
        if total_critical == 0 and len(tools_executed) >= 2:
            plan = self.contingency_library[TriggerCondition.NO_ISSUES_FOUND]
            if plan.trigger.value not in self.triggered_plans:
                triggered_plans.append(plan)
                self.triggered_plans.append(plan.trigger.value)
        
        # 8. Time constraint?
        if time_used > time_budget * 0.7:  # Used 70% of time budget
            plan = self.contingency_library[TriggerCondition.TIME_CONSTRAINT]
            if plan.trigger.value not in self.triggered_plans:
                triggered_plans.append(plan)
                self.triggered_plans.append(plan.trigger.value)
        
        # Sort by priority (highest first)
        triggered_plans.sort(key=lambda p: p.priority, reverse=True)
        
        return triggered_plans
    
    def apply_contingency_plan(self, plan: ContingencyPlan,
                              remaining_tools: List[str],
                              tool_scores: Dict[str, float]) -> Tuple[List[str], Dict[str, str]]:
        """
        Apply a contingency plan to adjust strategy.
        
        Returns:
            (adjusted_tools, adjustments_made)
        """
        adjustments = {}
        adjusted_tools = remaining_tools.copy()
        
        # 1. Skip tools
        for tool in plan.skip_tools:
            if tool in adjusted_tools:
                adjusted_tools.remove(tool)
                adjustments[tool] = f"Skipped: {plan.reasoning}"
        
        # 2. Boost tools (increase their scores)
        for tool in plan.boost_tools:
            if tool in tool_scores:
                tool_scores[tool] *= 1.5  # 50% boost
                adjustments[tool] = f"Boosted: {plan.reasoning}"
        
        # 3. Add extra checks (if not already planned)
        for check in plan.add_extra_checks:
            adjustments[check] = f"Added: {plan.reasoning}"
        
        # 4. Adjust thresholds (these will be applied globally)
        for threshold, value in plan.adjust_thresholds.items():
            adjustments[f"threshold_{threshold}"] = f"Set to {value}: {plan.reasoning}"
        
        # Re-sort remaining tools by adjusted scores
        adjusted_tools = sorted(
            adjusted_tools,
            key=lambda t: tool_scores.get(t, 0),
            reverse=True
        )
        
        self.active_plans.append(plan)
        
        return adjusted_tools, adjustments
    
    def get_remediation_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get automated remediation suggestions based on active contingency plans.
        
        Returns list of actionable remediation steps.
        """
        suggestions = []
        
        for plan in self.active_plans:
            if plan.auto_remediation:
                suggestions.append({
                    'trigger': plan.trigger.value,
                    'remediation_type': plan.auto_remediation,
                    'recommendations': plan.user_recommendations,
                    'priority': plan.priority,
                    'confidence': plan.confidence
                })
        
        return sorted(suggestions, key=lambda x: x['priority'], reverse=True)
    
    def _get_findings_by_tool(self, findings: Dict[str, List], tool_name: str) -> List:
        """Helper to get findings for a specific tool"""
        return findings.get(tool_name, [])
    
    def _has_critical_findings(self, findings: List) -> bool:
        """Helper to check if findings contain critical issues"""
        return any(f.get('severity') == 'critical' for f in findings)
    
    def get_contingency_summary(self) -> Dict[str, Any]:
        """Get summary of contingency planning activity"""
        return {
            'plans_triggered': len(self.triggered_plans),
            'triggers': self.triggered_plans,
            'active_plans': len(self.active_plans),
            'total_recommendations': sum(len(p.user_recommendations) for p in self.active_plans),
            'auto_remediations_available': sum(1 for p in self.active_plans if p.auto_remediation)
        }
    
    def reset(self):
        """Reset for new audit"""
        self.active_plans = []
        self.triggered_plans = []