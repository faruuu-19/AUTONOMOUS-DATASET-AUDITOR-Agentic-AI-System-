"""
auditor.py - Enhanced with Autonomous Decision-Making + Goal-Oriented Reasoning

ENHANCEMENTS:
1. Autonomous Decision-Making: Chooses which tools to run
2. Goal-Oriented Reasoning: Thinks strategically about objectives
   - Sets clear goals based on context
   - Dynamically adjusts strategy based on findings
   - Makes real-time decisions to optimize outcomes
   - Knows when to stop (goal achieved vs. keep searching)
"""

import pandas as pd
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
from agent.meta_learning_engine import MetaLearningEngine
from agent.contingency_planner import ContingencyPlanner
from agent.memory1 import AuditMemory
from agent.planner import AuditPlanner
from agent.critic import AuditCritic
from agent.strategy_engine import AutonomousStrategyEngine
from agent.goal_engine import GoalOrientedEngine  # NEW!

from tools.leakage_detector import LeakageDetector
from tools.contamination_detector import ContaminationDetector
from tools.bias_detector import BiasDetector
from tools.spurious_correlation_detector import SpuriousCorrelationDetector
from tools.feature_utility_detector import FeatureUtilityDetector


CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def read_csv_resilient(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Read a CSV, falling back through common encodings.

    Exported datasets are frequently Windows-1252 rather than UTF-8 (accented
    characters from Excel, for example). pandas defaults to UTF-8 and raises
    UnicodeDecodeError on those files, so try the usual suspects in order.
    latin-1 is last because it decodes any byte sequence and therefore always
    succeeds - it is the guaranteed fallback, not a guess.
    """
    last_error = None

    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(filepath, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    raise last_error


class AutonomousDatasetAuditor:
    """
    TRULY AUTONOMOUS Dataset Auditor with GOAL-ORIENTED REASONING
    
    Key capabilities:
    1. Profiles datasets and chooses optimal strategy
    2. Sets clear goals ("find critical issues in minimal time")
    3. Dynamically adjusts strategy based on findings
    4. Makes real-time tactical decisions
    5. Knows when to stop (goal achieved)
    6. Learns from experience to improve
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize the autonomous auditor with goal-oriented reasoning."""
        self.memory = AuditMemory()
        self.planner = AuditPlanner(self.memory)
        self.critic = AuditCritic(self.memory)
        self.strategy_engine = AutonomousStrategyEngine()
        self.goal_engine = GoalOrientedEngine(verbose=verbose)  # NEW!
        self.meta_learner = MetaLearningEngine()
        self.contingency_planner = ContingencyPlanner()
        self.verbose = verbose
        
        self.df = None
        self.target_column = None
        self.train_df = None
        self.test_df = None
        self.dataset_profile = None
        
    def load_dataset(self, filepath: str, target_column: str):
        """Load dataset and create autonomous strategy."""
        if self.verbose:
            print(f"📂 Loading dataset: {filepath}")
        
        self.df = read_csv_resilient(filepath)
        self.target_column = target_column
        
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Profile the dataset for autonomous decision-making
        self.dataset_profile = self.strategy_engine.profile_dataset(self.df, target_column)
        
        # Initialize memory
        self.memory.initialize_audit(
            dataset_shape=self.df.shape,
            target_column=target_column
        )
        
        if self.verbose:
            print(f"✓ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            print(f"✓ Target column: {target_column}")
            print(f"✓ Dataset profiled for autonomous strategy selection")
    
    def load_train_test_split(self, train_path: str, test_path: str, target_column: str):
        """Load separate train and test datasets."""
        if self.verbose:
            print(f"📂 Loading train/test split...")
        
        self.train_df = read_csv_resilient(train_path)
        self.test_df = read_csv_resilient(test_path)
        self.target_column = target_column
        
        self.df = pd.concat([self.train_df, self.test_df], ignore_index=True)
        self.dataset_profile = self.strategy_engine.profile_dataset(self.df, target_column)
        
        self.memory.initialize_audit(
            dataset_shape=self.df.shape,
            target_column=target_column
        )
        
        if self.verbose:
            print(f"✓ Train: {self.train_df.shape[0]} rows, Test: {self.test_df.shape[0]} rows")
    
    def run_audit(self) -> Dict[str, Any]:
        """Execute autonomous audit with goal-oriented reasoning."""
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")

        if self.verbose:
            print("\n" + "=" * 70)
            print("AUTONOMOUS GOAL-ORIENTED AUDIT INITIATED")
            print("=" * 70)

        self.contingency_planner.reset()
        audit_start_time = time.time()

        # STEP 1: Define goal based on dataset profile.
        num_tools_available = len(self.strategy_engine.available_tools)
        self.goal_engine.initialize_goal(
            self.dataset_profile.complexity_score,
            num_tools_available
        )

        # STEP 2: Initial strategy decision.
        tools_to_run, reasoning = self.strategy_engine.decide_audit_strategy(self.dataset_profile)
        if not tools_to_run:
            tools_to_run = list(self.strategy_engine.available_tools.keys())

        executed_tools = []
        execution_times = {}
        findings_dict = {}
        tool_scores = {tool: 0.5 for tool in tools_to_run}
        tools_remaining = list(tools_to_run)

        # STEP 3: Dynamic execution loop.
        while tools_remaining:
            if self.goal_engine.goal_state:
                total_planned = max(1, len(executed_tools) + len(tools_remaining))
                self.goal_engine.goal_state.tools_executed = len(executed_tools)
                self.goal_engine.goal_state.coverage = len(executed_tools) / total_planned

            should_continue, reason = self.goal_engine.should_continue_audit(
                tools_remaining,
                findings_dict,
                self.critic.get_overall_assessment().get("overall_confidence", 1.0),
            )
            if not should_continue:
                if self.verbose:
                    print(f"\nSTOPPING AUDIT: {reason}")
                break

            next_tool = tools_remaining.pop(0)
            tool_index = len(executed_tools) + 1

            if self.verbose:
                print(f"\n[{tool_index}/{len(tools_to_run)}] Executing: {next_tool}")
                print(f"    Reasoning: {reasoning.get(next_tool, 'selected by strategy engine')}")

            start_time = time.time()
            findings = self._execute_tool(next_tool)
            execution_time = time.time() - start_time
            self.goal_engine.update_time(execution_time)

            executed_tools.append(next_tool)
            execution_times[next_tool] = execution_time
            findings_dict[next_tool] = findings

            if findings:
                has_critical = any(f.get("severity") == "critical" for f in findings)
                status = "fail" if has_critical else "warning"
            else:
                status = "pass"

            self.memory.add_audit_step(next_tool, status, findings, execution_time)
            if self.verbose:
                print(f"    Status: {status.upper()} ({len(findings)} findings, {execution_time:.2f}s)")

            # Contingency planning.
            triggered_plans = self.contingency_planner.evaluate_triggers(
                current_findings=findings_dict,
                tools_executed=executed_tools,
                time_used=self.goal_engine.goal_state.time_used if self.goal_engine.goal_state else 0.0,
                time_budget=self.goal_engine.goal_state.time_budget if self.goal_engine.goal_state else 1.0,
                goal_state=self.goal_engine.goal_state,
            )
            for plan in triggered_plans:
                tools_remaining, _ = self.contingency_planner.apply_contingency_plan(
                    plan, tools_remaining, tool_scores
                )
                if self.verbose:
                    print(f"    Contingency activated: {plan.trigger.value}")

            # Tactical deep dive.
            should_dive, dive_reason = self.goal_engine.should_deep_dive(next_tool, findings)
            if should_dive:
                if self.verbose:
                    print(f"    Deep dive: {dive_reason}")

                critique = self.critic.evaluate_tool_results(next_tool)
                if critique.get("needs_recheck") and critique.get("confidence", 1.0) < 0.75:
                    recheck_findings = self._adaptive_recheck(next_tool, findings)
                    if recheck_findings:
                        findings.extend(recheck_findings)
                        findings_dict[next_tool] = findings
                        self.memory.findings[next_tool] = findings
                        if self.verbose:
                            print(f"    Re-check found {len(recheck_findings)} additional issue(s)")

            # Dynamic mid-audit strategy updates.
            if findings_dict and tools_remaining:
                adjusted_tools, adjustments = self.goal_engine.adjust_strategy_mid_audit(
                    findings_dict, tools_remaining, tool_scores
                )
                if adjusted_tools != tools_remaining:
                    tools_remaining = adjusted_tools
                    if self.verbose and adjustments:
                        print("    Strategy adjusted based on findings")

            # Early stopping.
            should_stop, stop_reason = self.goal_engine.evaluate_stopping_early(
                len(executed_tools),
                len(tools_remaining),
                self.goal_engine.goal_state.time_used if self.goal_engine.goal_state else 0.0,
                findings_dict,
            )
            if should_stop:
                if self.verbose:
                    print(f"\nEARLY STOP: {stop_reason}")
                break

            # Confidence guardrail.
            if self.memory.audit_steps:
                assessment = self.critic.get_overall_assessment()
                if assessment.get("overall_confidence", 1.0) < 0.4:
                    if self.verbose:
                        print("\nCONFIDENCE TOO LOW, stopping audit.")
                    break

        total_audit_time = time.time() - audit_start_time
        self.strategy_engine.record_audit_outcome(
            profile=self.dataset_profile,
            tools_executed=executed_tools,
            execution_order=tools_to_run[:len(executed_tools)],
            findings=findings_dict,
            execution_times=execution_times,
            total_time=total_audit_time,
        )

        self.memory.finalize_audit()

        # Generate report first, then run meta-learning based on this report.
        report = self._generate_report()

        try:
            learning_summary = self.meta_learner.learn_from_audit(
                audit_result=report,
                dataset_profile=self.dataset_profile.to_dict(),
                execution_data={
                    "tools_executed": executed_tools,
                    "execution_times": execution_times,
                    "findings": findings_dict,
                    "total_time": total_audit_time,
                },
            )
        except Exception as exc:
            learning_summary = {"error": str(exc)}
            if self.verbose:
                print(f"Meta-learning skipped due to error: {exc}")

        tools_skipped = list(set(self.strategy_engine.available_tools.keys()) - set(executed_tools))
        report["autonomous_strategy"] = {
            "tools_selected": tools_to_run,
            "tools_executed": executed_tools,
            "tools_skipped": tools_skipped,
            "reasoning": reasoning,
            "dataset_profile": self.dataset_profile.to_dict(),
            "learning_stats": self.strategy_engine.get_learning_stats(),
            "meta_learning_summary": learning_summary,
        }
        report["goal_oriented"] = self.goal_engine.get_goal_summary()
        report["contingency_summary"] = self.contingency_planner.get_contingency_summary()

        if self.verbose:
            print("\n" + "=" * 70)
            print("AUTONOMOUS GOAL-ORIENTED AUDIT COMPLETE")
            print("=" * 70)

        return report

    def _execute_tool(self, tool_name: str) -> list:
        """Execute a specific audit tool."""
        if tool_name == 'leakage_detector':
            detector = LeakageDetector(self.df, self.target_column)
            return detector.detect()
        elif tool_name == 'contamination_detector':
            if self.train_df is not None and self.test_df is not None:
                detector = ContaminationDetector(self.train_df, self.test_df)
            else:
                detector = ContaminationDetector(self.df)
            return detector.detect()
        elif tool_name == 'bias_detector':
            detector = BiasDetector(self.df, self.target_column)
            return detector.detect()
        elif tool_name == 'spurious_detector':
            detector = SpuriousCorrelationDetector(self.df, self.target_column)
            return detector.detect()
        elif tool_name == 'feature_utility':
            detector = FeatureUtilityDetector(self.df, self.target_column)
            return detector.detect()
        else:
            return []
    
    def _adaptive_recheck(self, tool_name: str, original_findings: list) -> list:
        """Perform deeper analysis when needed."""
        additional_findings = []
        
        if tool_name == 'spurious_detector':
            if self.verbose:
                print(f"       → Cross-validating with different parameters...")
            
            suspicious_features = [
                f.get('feature') for f in original_findings 
                if 'dominance' in f.get('type', '')
            ]
            
            for feature in suspicious_features:
                if feature and feature in self.df.columns:
                    feature_data = self.df[feature].dropna()
                    if len(feature_data) > 0:
                        cv = feature_data.std() / feature_data.mean() if feature_data.mean() != 0 else 0
                        if cv < 0.1:
                            additional_findings.append({
                                'type': 'unstable_predictor',
                                'severity': 'warning',
                                'feature': feature,
                                'message': f'Feature "{feature}" has low variance (CV={cv:.4f}) despite high accuracy - may not generalize',
                                'evidence': {'coefficient_variation': float(cv)}
                            })
        
        elif tool_name == 'leakage_detector':
            if self.verbose:
                print(f"       → Checking for temporal leakage patterns...")
            
            for col in self.df.columns:
                if col != self.target_column and col in self.df.select_dtypes(include=['object']).columns:
                    unique_vals = self.df[col].dropna().astype(str).unique()
                    target_vals = self.df[self.target_column].astype(str).unique()
                    
                    for target_val in target_vals:
                        if any(str(target_val) in str(val) for val in unique_vals):
                            additional_findings.append({
                                'type': 'encoded_target_leakage',
                                'severity': 'warning',
                                'feature': col,
                                'message': f'Feature "{col}" values may encode target information',
                                'evidence': {'sample_values': list(unique_vals[:5])}
                            })
                            break
        
        return additional_findings
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        stats = self.memory.get_summary_stats()
        assessment = self.critic.get_overall_assessment()
        all_findings = self.memory.get_all_findings()
        
        readiness_score = self._calculate_readiness_score(stats, assessment)
        
        critical_count = stats['critical_count']
        warning_count = stats['warning_count']
        
        if critical_count >= 3 or readiness_score < 30:
            verdict = "NOT READY"
            verdict_color = "🔴"
        elif critical_count > 0 or readiness_score < 60 or warning_count > 5:
            verdict = "NEEDS ATTENTION"
            verdict_color = "🟡"
        elif readiness_score >= 80 and warning_count <= 3:
            verdict = "READY"
            verdict_color = "🟢"
        else:
            verdict = "NEEDS ATTENTION"
            verdict_color = "🟡"
        
        critical_issues = [f for f in all_findings if f.get('severity') == 'critical']
        warnings = [f for f in all_findings if f.get('severity') == 'warning']
        
        report = {
            'audit_metadata': self.memory.metadata,
            'verdict': verdict,
            'readiness_score': readiness_score,
            'summary': {
                'tools_executed': stats['tools_executed'],
                'total_findings': stats['total_findings'],
                'critical_count': stats['critical_count'],
                'warning_count': stats['warning_count'],
                'info_count': stats['info_count']
            },
            'critical_blockers': critical_issues,
            'warnings': warnings,
            'critic_assessment': assessment,
            'recommendations': self._generate_recommendations(all_findings, assessment),
            'execution_timeline': self.memory.audit_steps,
            'all_findings': all_findings
        }
        
        if self.verbose:
            self._print_report(report, verdict_color)
        
        self._last_report = report
        return report
    
    def _calculate_readiness_score(self, stats: Dict, assessment: Dict) -> int:
        """Calculate readiness score with nuanced penalties."""
        score = 100
        critical = stats['critical_count']
        warnings = stats['warning_count']
        
        if critical > 0:
            if critical == 1:
                score -= 12
            elif critical == 2:
                score -= 22
            else:
                score -= 22 + (critical - 2) * 7
        
        if warnings > 0:
            if warnings <= 3:
                score -= warnings * 3
            elif warnings <= 6:
                score -= 9 + (warnings - 3) * 2
            else:
                score -= 15 + (warnings - 6) * 1
        
        confidence = assessment['overall_confidence']
        if confidence < 0.7:
            score -= 5
        elif confidence < 0.5:
            score -= 10
        
        return int(max(15, min(100, score)))
    
    def _generate_recommendations(self, findings: list, assessment: Dict) -> list:
        """Generate actionable recommendations."""
        recommendations = []
        recommendations.extend(assessment.get('actionable_recommendations', []))
        
        leakage_findings = [f for f in findings if f.get('tool') == 'leakage_detector']
        if leakage_findings:
            recommendations.append(
                "PRIORITY: Remove or investigate features flagged as data leakage before training any models."
            )
        
        bias_findings = [f for f in findings if f.get('tool') == 'bias_detector']
        severe_imbalance = any(f.get('type') == 'severe_imbalance' for f in bias_findings)
        if severe_imbalance:
            recommendations.append(
                "Address class imbalance using resampling, class weights, or stratified sampling."
            )
        
        contamination_findings = [f for f in findings if f.get('tool') == 'contamination_detector']
        if contamination_findings:
            recommendations.append(
                "Ensure proper train-test separation. Remove duplicate samples from test set."
            )
        
        return list(set(recommendations))
    
    def _print_report(self, report: Dict, verdict_color: str):
        """Print formatted audit report."""
        print(f"\n{verdict_color} VERDICT: {report['verdict']}")
        print(f"📊 READINESS SCORE: {report['readiness_score']}/100")
        
        print(f"\n📈 SUMMARY:")
        summary = report['summary']
        print(f"   Tools Executed: {summary['tools_executed']}")
        print(f"   Total Findings: {summary['total_findings']}")
        print(f"   Critical: {summary['critical_count']}")
        print(f"   Warnings: {summary['warning_count']}")
        print(f"   Info: {summary['info_count']}")
        
        if report['critical_blockers']:
            print(f"\n🚫 CRITICAL BLOCKERS ({len(report['critical_blockers'])}):")
            for i, issue in enumerate(report['critical_blockers'][:5], 1):
                print(f"   {i}. [{issue['tool']}] {issue.get('message', issue.get('type'))}")
        
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS ({len(report['recommendations'])}):")
            for i, rec in enumerate(report['recommendations'][:5], 1):
                print(f"   {i}. {rec}")
        
        print(f"\n🔍 Critic Assessment:")
        print(f"   Confidence: {report['critic_assessment']['overall_confidence']:.2f}")
        print(f"   Reliability: {report['critic_assessment']['reliability']}")
    
    def save_report(self, filepath: str = "reports/audit_report.json"):
        """Save audit report to JSON file."""
        if not hasattr(self, '_last_report'):
            raise ValueError("No audit report available. Run run_audit() first.")
        
        with open(filepath, 'w') as f:
            json.dump(self._last_report, f, indent=2, default=str)
        
        if self.verbose:
            print(f"\n💾 Report saved to: {filepath}")
    
    def export_findings_csv(self, filepath: str = "reports/findings.csv"):
        """Export findings to CSV for analysis."""
        all_findings = self.memory.get_all_findings()
        
        if not all_findings:
            print("No findings to export.")
            return
        
        df = pd.DataFrame(all_findings)
        df.to_csv(filepath, index=False)
        
        if self.verbose:
            print(f"💾 Findings exported to: {filepath}")