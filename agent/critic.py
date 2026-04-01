from typing import Dict, List, Any, Optional
from agent.memory1 import AuditMemory

class AuditCritic:
    """
    Critic module for autonomous dataset auditor.
    Evaluates the confidence and reliability of audit findings.
    Triggers deeper investigation when results are ambiguous or suspicious.
    """
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.9
    MEDIUM_CONFIDENCE = 0.7
    LOW_CONFIDENCE = 0.5
    
    def __init__(self, memory: AuditMemory):
        """
        Initialize critic with reference to memory.
        
        Args:
            memory: AuditMemory instance to query audit results
        """
        self.memory = memory
        self.critiques = []  # Store all critique evaluations
        
    def evaluate_tool_results(self, tool_name: str) -> Dict[str, Any]:
        """
        Evaluate the results from a specific tool.
        
        Args:
            tool_name: Name of the tool to evaluate
            
        Returns:
            Dictionary with confidence score, issues, and recommendations
        """
        findings = self.memory.get_findings_by_tool(tool_name)
        
        critique = {
            'tool': tool_name,
            'confidence': 1.0,  # Start with high confidence
            'concerns': [],
            'recommendations': [],
            'needs_recheck': False,
            'reliable': True
        }
        
        # Tool-specific evaluation logic
        if tool_name == 'leakage_detector':
            critique = self._evaluate_leakage_findings(findings, critique)
        elif tool_name == 'contamination_detector':
            critique = self._evaluate_contamination_findings(findings, critique)
        elif tool_name == 'bias_detector':
            critique = self._evaluate_bias_findings(findings, critique)
        elif tool_name == 'spurious_detector':
            critique = self._evaluate_spurious_findings(findings, critique)
        elif tool_name == 'feature_utility':
            critique = self._evaluate_utility_findings(findings, critique)
        
        # Store critique
        self.critiques.append(critique)
        
        return critique
    
    def _evaluate_leakage_findings(self, findings: List[Dict], critique: Dict) -> Dict:
        """Evaluate leakage detector results."""
        
        if not findings:
            return critique
        
        # Check for extremely high correlations (might be legitimate in some cases)
        perfect_corr_count = sum(
            1 for f in findings 
            if f.get('type') == 'perfect_correlation' 
            and f.get('evidence', {}).get('correlation', 0) > 0.99
        )
        
        if perfect_corr_count > 0:
            critique['concerns'].append(
                f'Found {perfect_corr_count} features with near-perfect correlation. '
                'Verify these are not legitimate derived features.'
            )
            critique['recommendations'].append(
                'Manually inspect features with perfect correlation to confirm they are truly leaked.'
            )
        
        # Check for suspicious names - these are warnings, not definitive
        suspicious_names = [
            f for f in findings 
            if f.get('type') == 'suspicious_name'
        ]
        
        if len(suspicious_names) > 0 and len(findings) == len(suspicious_names):
            # Only suspicious names, no hard evidence
            critique['confidence'] = self.MEDIUM_CONFIDENCE
            critique['concerns'].append(
                'Only found suspicious feature names without correlation evidence. '
                'Names alone are not definitive proof of leakage.'
            )
        
        # High confidence if we found identical features
        identical_count = sum(1 for f in findings if f.get('type') == 'identical_to_target')
        if identical_count > 0:
            critique['confidence'] = 1.0
            critique['concerns'].append(
                f'DEFINITIVE: {identical_count} features are identical to target. This is certain leakage.'
            )
        
        return critique
    
    def _evaluate_contamination_findings(self, findings: List[Dict], critique: Dict) -> Dict:
        """Evaluate contamination detector results."""
        
        if not findings:
            return critique
        
        # Check contamination percentage
        for finding in findings:
            if finding.get('type') == 'train_test_contamination':
                percentage = finding.get('evidence', {}).get('percentage', 0)
                
                if percentage > 10:
                    critique['confidence'] = self.HIGH_CONFIDENCE
                    critique['concerns'].append(
                        f'High contamination rate ({percentage:.1f}%) - definitely problematic.'
                    )
                elif percentage > 1:
                    critique['confidence'] = self.MEDIUM_CONFIDENCE
                    critique['concerns'].append(
                        f'Moderate contamination ({percentage:.1f}%) - investigate further.'
                    )
                else:
                    critique['confidence'] = self.LOW_CONFIDENCE
                    critique['concerns'].append(
                        f'Low contamination ({percentage:.1f}%) - might be acceptable depending on use case.'
                    )
                    critique['recommendations'].append(
                        'For critical applications, even 1% contamination should be addressed.'
                    )
        
        # Near-duplicates are less certain than exact duplicates
        near_dup_findings = [f for f in findings if f.get('type') == 'near_duplicates']
        if near_dup_findings and not any(f.get('type') == 'train_test_contamination' for f in findings):
            critique['confidence'] = self.MEDIUM_CONFIDENCE
            critique['concerns'].append(
                'Found only near-duplicates, not exact matches. Similarity threshold might need tuning.'
            )
            critique['recommendations'].append(
                'Manually inspect near-duplicate samples to verify they are truly problematic.'
            )
        
        return critique
    
    def _evaluate_bias_findings(self, findings: List[Dict], critique: Dict) -> Dict:
        """Evaluate bias detector results."""
        
        if not findings:
            return critique
        
        # Check class imbalance severity
        severe_imbalance = any(f.get('type') == 'severe_imbalance' for f in findings)
        if severe_imbalance:
            critique['confidence'] = self.HIGH_CONFIDENCE
            critique['concerns'].append(
                'Severe class imbalance confirmed. This will impact model training.'
            )
            critique['recommendations'].append(
                'Consider: SMOTE, class weights, or stratified sampling.'
            )
        
        # Small class sizes are concerning but depend on total data
        small_class = any(f.get('type') == 'small_class_size' for f in findings)
        if small_class:
            for finding in findings:
                if finding.get('type') == 'small_class_size':
                    size = finding.get('evidence', {}).get('min_class_size', 0)
                    if size < 10:
                        critique['confidence'] = self.HIGH_CONFIDENCE
                        critique['concerns'].append(
                            f'Extremely small class size ({size} samples) - likely insufficient for training.'
                        )
                    else:
                        critique['confidence'] = self.MEDIUM_CONFIDENCE
                        critique['concerns'].append(
                            f'Small class size ({size} samples) - monitor performance on minority class.'
                        )
        
        # Missing value bias needs context
        missing_bias = [f for f in findings if 'missing' in f.get('type', '')]
        if missing_bias:
            critique['recommendations'].append(
                'Missing value patterns correlate with target. Imputation strategy is critical.'
            )
        
        return critique
    
    def _evaluate_spurious_findings(self, findings: List[Dict], critique: Dict) -> Dict:
        """Evaluate spurious correlation detector results."""
        
        if not findings:
            return critique
        
        # Single feature dominance is suspicious but needs verification
        dominance_findings = [f for f in findings if 'dominance' in f.get('type', '')]
        if dominance_findings:
            critique['confidence'] = self.MEDIUM_CONFIDENCE
            critique['concerns'].append(
                'Single features achieving high accuracy alone. Could be legitimate or spurious.'
            )
            critique['recommendations'].extend([
                'Verify domain knowledge: Is this feature logically causal?',
                'Test model on out-of-distribution data to check generalization.',
                'Consider ablation studies to understand feature importance.'
            ])
            critique['needs_recheck'] = True
        
        # Simple threshold rules are red flags but need context
        threshold_findings = [f for f in findings if 'threshold' in f.get('type', '')]
        if threshold_findings:
            critique['concerns'].append(
                'Simple threshold rules work too well. Verify this is not dataset artifact.'
            )
        
        return critique
    
    def _evaluate_utility_findings(self, findings: List[Dict], critique: Dict) -> Dict:
        """Evaluate feature utility detector results."""
        
        if not findings:
            return critique
        
        # Constant features are definitive
        constant_findings = [f for f in findings if 'constant' in f.get('type', '')]
        if constant_findings:
            critique['confidence'] = self.HIGH_CONFIDENCE
            # These are straightforward - high confidence
        
        # Low information features depend on context
        low_info_findings = [f for f in findings if 'low_information' in f.get('type', '')]
        if low_info_findings and len(low_info_findings) == len(findings):
            critique['confidence'] = self.MEDIUM_CONFIDENCE
            critique['concerns'].append(
                'Many features flagged as low information. Verify with domain experts before removal.'
            )
            critique['recommendations'].append(
                'Low information features might still be useful in ensembles or interactions.'
            )
        
        return critique
    
    def get_overall_assessment(self) -> Dict[str, Any]:
        """
        Provide overall assessment of audit quality and reliability.
        
        Returns:
            Dictionary with overall confidence and recommendations
        """
        if not self.critiques:
            return {
                'overall_confidence': 1.0,
                'reliability': 'high',
                'major_concerns': [],
                'actionable_recommendations': []
            }
        
        # Calculate average confidence
        confidences = [c['confidence'] for c in self.critiques]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Collect all concerns and recommendations
        all_concerns = []
        all_recommendations = []
        needs_recheck = []
        
        for critique in self.critiques:
            all_concerns.extend(critique.get('concerns', []))
            all_recommendations.extend(critique.get('recommendations', []))
            if critique.get('needs_recheck'):
                needs_recheck.append(critique['tool'])
        
        # Determine reliability level
        if avg_confidence >= self.HIGH_CONFIDENCE:
            reliability = 'high'
        elif avg_confidence >= self.MEDIUM_CONFIDENCE:
            reliability = 'medium'
        else:
            reliability = 'low'
        
        return {
            'overall_confidence': avg_confidence,
            'reliability': reliability,
            'major_concerns': all_concerns,
            'actionable_recommendations': list(set(all_recommendations)),  # Remove duplicates
            'tools_needing_recheck': needs_recheck,
            'critique_count': len(self.critiques)
        }
    
    def should_trigger_recheck(self, tool_name: str) -> bool:
        """
        Check if a tool's results should trigger a recheck.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if recheck is recommended
        """
        for critique in self.critiques:
            if critique['tool'] == tool_name:
                return critique.get('needs_recheck', False)
        return False
    
    def get_critique_summary(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get critique summary for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Critique dictionary or None if not found
        """
        for critique in self.critiques:
            if critique['tool'] == tool_name:
                return critique
        return None
    
    def __repr__(self) -> str:
        """String representation of critic state."""
        assessment = self.get_overall_assessment()
        return (f"AuditCritic(critiques={len(self.critiques)}, "
                f"confidence={assessment['overall_confidence']:.2f}, "
                f"reliability={assessment['reliability']})")