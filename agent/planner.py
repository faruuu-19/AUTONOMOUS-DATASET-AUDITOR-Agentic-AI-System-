from typing import Optional, List, Dict, Any
from agent.memory1 import AuditMemory

class AuditPlanner:
    """
    Planner module for autonomous dataset auditor.
    Determines the sequence of audit checks based on:
    - Priority (leakage is most critical)
    - Previous findings (adapt strategy based on results)
    - Stopping criteria (halt if too many critical issues)
    """
    
    # Define tool execution priority
    TOOL_PRIORITY = [
        'leakage_detector',      # Priority 1: Most critical - target leakage
        'contamination_detector', # Priority 2: Train-test contamination
        'bias_detector',         # Priority 3: Class imbalance and bias
        'spurious_detector',     # Priority 4: Shortcut learning
        'feature_utility'        # Priority 5: Feature quality
    ]
    
    # Stopping thresholds
    MAX_CRITICAL_BEFORE_STOP = 10  # Stop if we find this many critical issues
    
    def __init__(self, memory: AuditMemory):
        """
        Initialize planner with reference to memory.
        
        Args:
            memory: AuditMemory instance to query audit state
        """
        self.memory = memory
        self.skipped_tools = []  # ENHANCEMENT 2: Track skipped tools
        self.skip_reasons = {}   # Store reasons for skipping
        
    def get_next_tool(self) -> Optional[str]:
        """
        Determine which tool should be executed next.
        
        Returns:
            Name of next tool to execute, or None if audit should stop
        """
        # Check if we should stop early
        if self._should_stop_early():
            return None
        
        # ENHANCEMENT 2: Find next tool, skipping irrelevant ones
        for tool_name in self.TOOL_PRIORITY:
            if not self.memory.has_executed(tool_name) and tool_name not in self.skipped_tools:
                # Check if this tool should be skipped
                should_skip, reason = self._should_skip_tool(tool_name)
                
                if should_skip:
                    self.skipped_tools.append(tool_name)
                    self.skip_reasons[tool_name] = reason
                    continue  # Move to next tool
                
                return tool_name
        
        # All tools have been executed or skipped
        return None
    
    def _should_stop_early(self) -> bool:
        """
        Determine if audit should stop early due to severe issues.
        
        Returns:
            True if audit should halt, False to continue
        """
        stats = self.memory.get_summary_stats()
        
        # Stop if too many critical issues found
        if stats['critical_count'] >= self.MAX_CRITICAL_BEFORE_STOP:
            return True
        
        # Check for specific blocking conditions
        leakage_findings = self.memory.get_findings_by_tool('leakage_detector')
        if leakage_findings:
            # If we found perfect target leakage (correlation = 1.0), might want to stop
            for finding in leakage_findings:
                if finding.get('type') == 'identical_to_target':
                    # Found feature identical to target - this is extremely severe
                    # But we still continue to find ALL issues
                    pass
        
        return False
    
    def _should_skip_tool(self, tool_name: str) -> tuple[bool, str]:
        """
        ENHANCEMENT 2: Determine if a tool should be skipped based on dataset characteristics.
        
        Args:
            tool_name: Tool to evaluate
            
        Returns:
            Tuple of (should_skip: bool, reason: str)
        """
        metadata = self.memory.metadata
        dataset_shape = metadata.get('dataset_shape', (0, 0))
        rows, cols = dataset_shape
        
        # Skip feature_utility if very few features
        if tool_name == 'feature_utility':
            # Target column + 1 or 2 features = not much to evaluate
            if cols <= 3:
                return True, f"Dataset has only {cols} columns (including target) - too few for utility analysis"
        
        # Skip spurious_detector if very small dataset
        if tool_name == 'spurious_detector':
            if rows < 50:
                return True, f"Dataset has only {rows} rows - insufficient for spurious correlation detection"
        
        # Skip contamination_detector if no critical findings yet
        # (This is adaptive - contamination is less critical if no other issues)
        if tool_name == 'contamination_detector':
            # Check if we have train/test split information
            # This would need to be set by the auditor
            # For now, we don't skip it
            pass
        
        return False, ""
    
    def should_continue(self) -> bool:
        """
        Check if there are more tools to execute.
        
        Returns:
            True if more tools should be run, False if complete
        """
        return self.get_next_tool() is not None
    
    def get_execution_plan(self) -> List[str]:
        """
        Get the planned execution order for remaining tools.
        
        Returns:
            List of tool names in planned execution order
        """
        plan = []
        for tool_name in self.TOOL_PRIORITY:
            if not self.memory.has_executed(tool_name):
                plan.append(tool_name)
        return plan
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get audit progress information.
        
        Returns:
            Dictionary with progress metrics
        """
        total_tools = len(self.TOOL_PRIORITY)
        completed_tools = len(self.memory.get_execution_order())
        remaining_tools = total_tools - completed_tools
        
        return {
            'total_tools': total_tools,
            'completed': completed_tools,
            'remaining': remaining_tools,
            'progress_percentage': (completed_tools / total_tools) * 100,
            'next_tool': self.get_next_tool()
        }
    
    def get_reason_for_tool(self, tool_name: str) -> str:
        """
        Get human-readable explanation for why a tool is being executed.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Explanation string
        """
        reasons = {
            'leakage_detector': (
                'Checking for data leakage - the most critical dataset flaw. '
                'Leakage occurs when features contain information about the target '
                'that would not be available at prediction time.'
            ),
            'contamination_detector': (
                'Checking for train-test contamination - when identical samples '
                'appear in both training and test sets, leading to inflated '
                'performance estimates.'
            ),
            'bias_detector': (
                'Analyzing class balance and feature distributions. Severe imbalance '
                'or bias can cause models to fail on minority classes.'
            ),
            'spurious_detector': (
                'Detecting spurious correlations and shortcut learning. Models can '
                'latch onto misleading patterns that do not generalize.'
            ),
            'feature_utility': (
                'Evaluating feature quality. Removing low-utility features '
                'improves model efficiency and reduces overfitting risk.'
            )
        }
        return reasons.get(tool_name, 'Performing standard audit check.')
    
    def get_adaptive_strategy(self) -> Dict[str, Any]:
        """
        Generate adaptive audit strategy based on findings so far.
        
        Returns:
            Dictionary with strategy recommendations
        """
        stats = self.memory.get_summary_stats()
        strategy = {
            'current_status': 'green',
            'recommendations': [],
            'priority_areas': []
        }
        
        # Determine overall status
        if stats['critical_count'] > 0:
            strategy['current_status'] = 'red'
        elif stats['warning_count'] > 3:
            strategy['current_status'] = 'yellow'
        
        # Add recommendations based on findings
        if self.memory.has_executed('leakage_detector'):
            leakage_findings = self.memory.get_findings_by_tool('leakage_detector')
            critical_leakage = [f for f in leakage_findings if f.get('severity') == 'critical']
            
            if critical_leakage:
                strategy['recommendations'].append(
                    'CRITICAL: Data leakage detected. Do not train models until resolved.'
                )
                strategy['priority_areas'].append('data_leakage')
        
        if self.memory.has_executed('bias_detector'):
            bias_findings = self.memory.get_findings_by_tool('bias_detector')
            severe_imbalance = any(
                f.get('type') == 'severe_imbalance' 
                for f in bias_findings
            )
            
            if severe_imbalance:
                strategy['recommendations'].append(
                    'Severe class imbalance detected. Consider resampling or adjusted metrics.'
                )
                strategy['priority_areas'].append('class_imbalance')
        
        return strategy
    
    def explain_execution_order(self) -> List[Dict[str, str]]:
        """
        Provide detailed explanation of tool execution order.
        
        Returns:
            List of dictionaries with tool name, priority, and reasoning
        """
        explanations = []
        for idx, tool_name in enumerate(self.TOOL_PRIORITY, 1):
            explanations.append({
                'priority': idx,
                'tool': tool_name,
                'reason': self.get_reason_for_tool(tool_name),
                'executed': self.memory.has_executed(tool_name)
            })
        return explanations
    
    def __repr__(self) -> str:
        """String representation of planner state."""
        progress = self.get_progress()
        skipped_info = f", skipped={len(self.skipped_tools)}" if self.skipped_tools else ""
        return (f"AuditPlanner(completed={progress['completed']}/{progress['total_tools']}"
                f"{skipped_info}, next={progress['next_tool']})")
    
    def get_skipped_tools(self) -> Dict[str, str]:
        """
        ENHANCEMENT 2: Get tools that were skipped and reasons.
        
        Returns:
            Dictionary mapping tool names to skip reasons
        """
        return self.skip_reasons.copy()