"""
agent/goal_engine.py - Goal-Oriented Reasoning Engine

This module implements goal-oriented behavior:
- Defines clear objectives (find critical issues in minimal time)
- Dynamically adjusts strategy mid-audit based on findings
- Reasons about progress toward goals
- Makes real-time decisions to optimize outcomes
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class AuditGoal(Enum):
    """Primary audit objectives"""
    FIND_CRITICAL_FAST = "find_critical_issues_quickly"
    COMPREHENSIVE_SCAN = "comprehensive_coverage"
    QUICK_VALIDATION = "quick_validation"
    DEEP_INVESTIGATION = "deep_investigation"


@dataclass
class GoalState:
    """Current state of goal achievement"""
    primary_goal: AuditGoal
    time_budget: float  # seconds
    time_used: float
    critical_found: int
    tools_executed: int
    coverage: float  # % of planned tools executed
    confidence: float  # confidence in findings so far
    goal_progress: float  # 0-1, how close to achieving goal
    

class GoalOrientedEngine:
    """
    GOAL-ORIENTED REASONING ENGINE
    
    This engine thinks strategically about objectives:
    - "My goal is to find critical issues in minimal time"
    - "I've found leakage - should I dig deeper or move on?"
    - "Time is running out - focus on high-impact checks only"
    - "No issues yet - expand search to be thorough"
    
    It makes real-time strategic decisions to optimize outcomes.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.goal_state: Optional[GoalState] = None
        self.decision_log = []
        self.strategy_changes = []
        
    def initialize_goal(self, dataset_complexity: float, 
                       num_tools_available: int) -> AuditGoal:
        """
        Determine primary audit goal based on context.
        
        This sets the overall objective that guides all decisions.
        """
        # Decision tree for goal selection
        if dataset_complexity > 0.7:
            # High complexity = need deep investigation
            goal = AuditGoal.DEEP_INVESTIGATION
            time_budget = 300  # 5 minutes
            reason = "High complexity dataset requires thorough investigation"
        elif dataset_complexity < 0.3:
            # Low complexity = quick validation sufficient
            goal = AuditGoal.QUICK_VALIDATION
            time_budget = 60  # 1 minute
            reason = "Low complexity dataset - quick validation sufficient"
        else:
            # Medium complexity = find critical issues efficiently
            goal = AuditGoal.FIND_CRITICAL_FAST
            time_budget = 180  # 3 minutes
            reason = "Standard audit - prioritize finding critical issues quickly"
        
        self.goal_state = GoalState(
            primary_goal=goal,
            time_budget=time_budget,
            time_used=0.0,
            critical_found=0,
            tools_executed=0,
            coverage=0.0,
            confidence=1.0,
            goal_progress=0.0
        )
        
        if self.verbose:
            print(f"\n🎯 GOAL SET: {goal.value}")
            print(f"   Reasoning: {reason}")
            print(f"   Time budget: {time_budget}s")
            print(f"   Success criteria: {self._get_success_criteria(goal)}")
        
        self._log_decision("goal_initialization", {
            'goal': goal.value,
            'reason': reason,
            'time_budget': time_budget
        })
        
        return goal
    
    def _get_success_criteria(self, goal: AuditGoal) -> str:
        """Define what success looks like for each goal"""
        criteria = {
            AuditGoal.FIND_CRITICAL_FAST: "Find at least 1 critical issue OR confirm dataset is clean, in minimal time",
            AuditGoal.COMPREHENSIVE_SCAN: "Execute all relevant tools with high confidence",
            AuditGoal.QUICK_VALIDATION: "Verify no obvious critical issues exist",
            AuditGoal.DEEP_INVESTIGATION: "Thoroughly investigate all risk areas"
        }
        return criteria.get(goal, "Complete audit successfully")
    
    def should_continue_audit(self, tools_remaining: List[str], 
                             findings_so_far: Dict[str, List],
                             current_confidence: float) -> Tuple[bool, str]:
        """
        STRATEGIC DECISION: Should we continue or stop?
        
        This is goal-oriented reasoning in action:
        - If goal achieved, stop (don't waste time)
        - If running out of time, stop low-priority work
        - If critical issues found, maybe stop or dig deeper
        - If nothing found, expand search
        """
        if not self.goal_state:
            return True, "Goal not initialized"
        
        # Update goal state
        total_critical = sum(
            sum(1 for f in findings if f.get('severity') == 'critical')
            for findings in findings_so_far.values()
        )
        self.goal_state.critical_found = total_critical
        self.goal_state.confidence = current_confidence
        
        # Calculate goal progress
        self.goal_state.goal_progress = self._calculate_goal_progress()
        
        # DECISION LOGIC based on goal
        if self.goal_state.primary_goal == AuditGoal.FIND_CRITICAL_FAST:
            # Goal: Find critical issues quickly
            
            if total_critical > 0 and self.goal_state.time_used > 60:
                # Found critical issues, spent reasonable time
                return False, f"✓ GOAL ACHIEVED: Found {total_critical} critical issue(s) in {self.goal_state.time_used:.1f}s"
            
            if self.goal_state.time_used > self.goal_state.time_budget * 0.8:
                # Running out of time
                if total_critical == 0:
                    # No critical found, but time is up
                    return False, f"⏱ TIME LIMIT: No critical issues found in {self.goal_state.time_used:.1f}s - likely clean dataset"
                else:
                    # Have findings, time is up
                    return False, f"⏱ TIME LIMIT: Stopping with {total_critical} critical issue(s) found"
            
            if len(tools_remaining) == 0:
                return False, "✓ All relevant tools executed"
            
            return True, "Continuing search for critical issues"
        
        elif self.goal_state.primary_goal == AuditGoal.QUICK_VALIDATION:
            # Goal: Quick check only
            
            if total_critical > 0:
                # Found issues - goal failed (dataset not clean)
                return False, f"⚠ VALIDATION FAILED: Found {total_critical} critical issue(s)"
            
            if self.goal_state.tools_executed >= 3:
                # Checked enough, no issues
                return False, "✓ VALIDATION PASSED: No critical issues in quick scan"
            
            if self.goal_state.time_used > self.goal_state.time_budget:
                return False, "⏱ TIME LIMIT: Quick validation complete"
            
            return True, "Continuing quick validation"
        
        elif self.goal_state.primary_goal == AuditGoal.DEEP_INVESTIGATION:
            # Goal: Thorough investigation
            
            if len(tools_remaining) == 0:
                return False, "✓ Deep investigation complete - all tools executed"
            
            if self.goal_state.time_used > self.goal_state.time_budget:
                if self.goal_state.coverage > 0.8:
                    return False, "⏱ TIME LIMIT: Deep investigation covered 80%+ of tools"
                else:
                    # Not enough coverage, but time is up
                    return True, "Time limit reached but coverage insufficient - continuing critical tools only"
            
            return True, "Continuing deep investigation"
        
        else:  # COMPREHENSIVE_SCAN
            if len(tools_remaining) == 0:
                return False, "✓ Comprehensive scan complete"
            
            if self.goal_state.time_used > self.goal_state.time_budget * 1.5:
                return False, "⏱ TIME EXCEEDED: Stopping comprehensive scan"
            
            return True, "Continuing comprehensive scan"
    
    def _calculate_goal_progress(self) -> float:
        """Calculate progress toward goal (0-1)"""
        if not self.goal_state:
            return 0.0
        
        goal = self.goal_state.primary_goal
        
        if goal == AuditGoal.FIND_CRITICAL_FAST:
            # Progress = found critical OR high coverage with confidence
            if self.goal_state.critical_found > 0:
                return 1.0
            else:
                # Progress based on coverage and confidence
                return (self.goal_state.coverage * 0.6 + 
                       self.goal_state.confidence * 0.4)
        
        elif goal == AuditGoal.QUICK_VALIDATION:
            # Progress = enough tools checked
            return min(1.0, self.goal_state.tools_executed / 3.0)
        
        elif goal == AuditGoal.DEEP_INVESTIGATION:
            # Progress = coverage
            return self.goal_state.coverage
        
        else:  # COMPREHENSIVE_SCAN
            return self.goal_state.coverage
    
    def adjust_strategy_mid_audit(self, current_findings: Dict[str, List],
                                  remaining_tools: List[str],
                                  tool_scores: Dict[str, float]) -> Tuple[List[str], Dict[str, str]]:
        """
        DYNAMIC STRATEGY ADJUSTMENT
        
        This is the heart of goal-oriented reasoning:
        - React to what we've found
        - Re-prioritize remaining work
        - Make strategic pivots
        
        Example: "I found data leakage - this changes everything!
                 I should now prioritize bias checks and skip low-priority tools"
        """
        if not current_findings:
            return remaining_tools, {}
        
        # Analyze what we've found so far
        has_leakage = any('leakage' in tool for tool in current_findings.keys())
        has_bias_issues = any('bias' in tool for tool in current_findings.keys())
        has_contamination = any('contamination' in tool for tool in current_findings.keys())
        
        total_critical = sum(
            sum(1 for f in findings if f.get('severity') == 'critical')
            for findings in current_findings.values()
        )
        
        adjustments = {}
        
        # STRATEGIC PIVOT 1: Data leakage found
        if has_leakage and total_critical > 0:
            if self.verbose:
                print(f"\n🔄 STRATEGIC PIVOT: Data leakage detected!")
                print(f"   Adjusting strategy: Deprioritizing spurious correlation check")
                print(f"   Reasoning: Leakage makes spurious correlations irrelevant")
            
            # Lower priority of spurious detector (leakage explains everything)
            if 'spurious_detector' in tool_scores:
                tool_scores['spurious_detector'] *= 0.3
                adjustments['spurious_detector'] = "Deprioritized: leakage found (would explain correlations)"
            
            # Boost feature utility (need to identify leaky features)
            if 'feature_utility' in tool_scores:
                tool_scores['feature_utility'] *= 1.5
                adjustments['feature_utility'] = "Boosted: need to identify problematic features"
            
            self._log_strategy_change("leakage_pivot", {
                'reason': 'Data leakage detected',
                'adjustments': adjustments
            })
        
        # STRATEGIC PIVOT 2: Severe bias found
        if has_bias_issues and total_critical > 0:
            if self.verbose:
                print(f"\n🔄 STRATEGIC PIVOT: Severe class imbalance detected!")
                print(f"   Adjusting strategy: Boosting contamination check")
                print(f"   Reasoning: Imbalanced datasets prone to train-test contamination")
            
            # Boost contamination detector
            if 'contamination_detector' in tool_scores:
                tool_scores['contamination_detector'] *= 1.4
                adjustments['contamination_detector'] = "Boosted: imbalanced datasets prone to contamination"
            
            self._log_strategy_change("bias_pivot", {
                'reason': 'Severe class imbalance detected',
                'adjustments': adjustments
            })
        
        # STRATEGIC PIVOT 3: Multiple critical issues found
        if total_critical >= 3:
            if self.verbose:
                print(f"\n🔄 STRATEGIC PIVOT: Multiple critical issues found!")
                print(f"   Adjusting strategy: Dataset has serious problems")
                print(f"   Reasoning: Focus on documenting issues, skip minor checks")
            
            # Lower threshold for skipping tools
            remaining_tools = [t for t in remaining_tools if tool_scores.get(t, 0) > 0.4]
            adjustments['strategy'] = "Raised skip threshold to 0.4 - focus on high-value checks only"
            
            self._log_strategy_change("critical_pivot", {
                'reason': f'{total_critical} critical issues found',
                'adjustments': adjustments
            })
        
        # STRATEGIC PIVOT 4: Nothing found yet + time running out
        if total_critical == 0 and self.goal_state:
            time_remaining = self.goal_state.time_budget - self.goal_state.time_used
            if time_remaining < 60 and len(remaining_tools) > 2:
                if self.verbose:
                    print(f"\n🔄 STRATEGIC PIVOT: Time running out, no issues found")
                    print(f"   Adjusting strategy: Focus on most likely issue sources")
                    print(f"   Reasoning: {time_remaining:.0f}s left - prioritize high-yield checks")
                
                # Keep only top 2 remaining tools
                remaining_tools = sorted(
                    remaining_tools, 
                    key=lambda t: tool_scores.get(t, 0), 
                    reverse=True
                )[:2]
                adjustments['strategy'] = f"Time constraint: focusing on top {len(remaining_tools)} tools only"
                
                self._log_strategy_change("time_pivot", {
                    'reason': 'Time running out with no findings',
                    'adjustments': adjustments
                })
        
        # Re-sort remaining tools based on adjusted scores
        remaining_tools = sorted(
            remaining_tools,
            key=lambda t: tool_scores.get(t, 0),
            reverse=True
        )
        
        return remaining_tools, adjustments
    
    def should_deep_dive(self, tool_name: str, findings: List[Dict]) -> Tuple[bool, str]:
        """
        TACTICAL DECISION: Should we investigate deeper?
        
        Goal-oriented reasoning:
        - If critical issue found and goal is FIND_CRITICAL_FAST -> Yes, dig deeper
        - If quick validation and issue found -> No, we already know it failed
        - If deep investigation mode -> Always yes
        """
        if not findings or not self.goal_state:
            return False, "No findings or no goal set"
        
        critical_findings = [f for f in findings if f.get('severity') == 'critical']
        
        goal = self.goal_state.primary_goal
        
        if goal == AuditGoal.FIND_CRITICAL_FAST:
            if critical_findings:
                return True, f"Critical issue found in {tool_name} - investigating deeper to confirm"
            return False, "No critical findings - moving on to stay fast"
        
        elif goal == AuditGoal.QUICK_VALIDATION:
            # Quick validation doesn't deep dive
            return False, "Quick validation mode - no deep dives"
        
        elif goal == AuditGoal.DEEP_INVESTIGATION:
            # Always investigate deeper in this mode
            if findings:
                return True, f"Deep investigation mode - analyzing {len(findings)} finding(s) thoroughly"
            return False, "No findings to investigate"
        
        else:  # COMPREHENSIVE_SCAN
            if critical_findings:
                return True, f"Critical findings warrant deeper analysis"
            return False, "Non-critical findings - standard analysis sufficient"
    
    def evaluate_stopping_early(self, tools_executed: int,
                                tools_remaining: int,
                                time_used: float,
                                findings: Dict[str, List]) -> Tuple[bool, str]:
        """
        STRATEGIC DECISION: Should we stop the audit early?
        
        Goal-oriented reasoning about early termination:
        - Goal achieved? Stop.
        - Diminishing returns? Stop.
        - Critical issues found and confirmed? Maybe stop.
        - Time budget exceeded with good coverage? Stop.
        """
        if not self.goal_state:
            return False, "No goal set - continue"
        
        # Update state
        self.goal_state.tools_executed = tools_executed
        self.goal_state.coverage = tools_executed / max(1, tools_executed + tools_remaining)
        self.goal_state.time_used = time_used
        
        total_critical = sum(
            sum(1 for f in findings_list if f.get('severity') == 'critical')
            for findings_list in findings.values()
        )
        self.goal_state.critical_found = total_critical
        
        goal = self.goal_state.primary_goal
        
        # EARLY STOP CONDITION 1: Goal achieved
        if self.goal_state.goal_progress >= 0.95:
            return True, f"✓ GOAL ACHIEVED: {goal.value} completed ({self.goal_state.goal_progress:.0%})"
        
        # EARLY STOP CONDITION 2: Time budget exceeded
        if time_used > self.goal_state.time_budget:
            if self.goal_state.coverage > 0.6:
                return True, f"⏱ TIME BUDGET EXCEEDED: Covered {self.goal_state.coverage:.0%} of tools in {time_used:.1f}s"
            else:
                # Not enough coverage yet
                return False, f"Time exceeded but coverage only {self.goal_state.coverage:.0%} - continuing critical tools"
        
        # EARLY STOP CONDITION 3: Severe issues found (goal dependent)
        if goal == AuditGoal.FIND_CRITICAL_FAST and total_critical >= 2:
            if tools_executed >= 3:  # Checked enough tools
                return True, f"✓ GOAL ACHIEVED: Found {total_critical} critical issues in {tools_executed} tools"
        
        # EARLY STOP CONDITION 4: Diminishing returns
        if tools_executed >= 4 and total_critical == 0:
            if goal == AuditGoal.QUICK_VALIDATION:
                return True, "✓ VALIDATION PASSED: No issues in 4+ tool checks"
        
        return False, "Continuing audit - goal not yet achieved"
    
    def update_time(self, time_increment: float):
        """Update time tracking"""
        if self.goal_state:
            self.goal_state.time_used += time_increment
    
    def _log_decision(self, decision_type: str, details: Dict):
        """Log strategic decisions for analysis"""
        self.decision_log.append({
            'timestamp': time.time(),
            'type': decision_type,
            'details': details
        })
    
    def _log_strategy_change(self, change_type: str, details: Dict):
        """Log strategy changes for analysis"""
        self.strategy_changes.append({
            'timestamp': time.time(),
            'type': change_type,
            'details': details
        })
    
    def get_goal_summary(self) -> Dict[str, Any]:
        """Get summary of goal achievement"""
        if not self.goal_state:
            return {'goal_set': False}
        
        return {
            'goal_set': True,
            'primary_goal': self.goal_state.primary_goal.value,
            'time_budget': self.goal_state.time_budget,
            'time_used': self.goal_state.time_used,
            'time_efficiency': self.goal_state.time_used / self.goal_state.time_budget if self.goal_state.time_budget > 0 else 0,
            'critical_found': self.goal_state.critical_found,
            'tools_executed': self.goal_state.tools_executed,
            'coverage': self.goal_state.coverage,
            'goal_progress': self.goal_state.goal_progress,
            'goal_achieved': self.goal_state.goal_progress >= 0.95,
            'strategy_changes_count': len(self.strategy_changes),
            'strategy_changes': self.strategy_changes,
            'decision_count': len(self.decision_log)
        }