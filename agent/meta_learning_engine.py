"""
agent/meta_learning_engine.py - Self-Improvement & Meta-Learning

POINT 3: META-LEARNING
This module enables the auditor to:
- Learn which strategies work best for which dataset types
- Track performance metrics across audits
- Auto-tune parameters based on success
- Recognize cross-dataset patterns
- Continuously improve decision-making

The system learns from EVERY audit and gets smarter over time.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime


@dataclass
class StrategyPerformance:
    """Performance metrics for a specific strategy"""
    strategy_id: str
    dataset_type: str  # "small_imbalanced", "large_balanced", etc.
    tool_sequence: List[str]
    
    # Performance metrics
    time_to_first_critical: float  # How fast did we find critical issues?
    total_time: float
    critical_found: int
    false_positive_rate: float  # How many tools found nothing?
    efficiency_score: float  # critical_found / time_used
    
    # Success indicators
    goal_achieved: bool
    early_stop_triggered: bool
    
    # Context
    timestamp: str
    complexity_score: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class LearnedPattern:
    """A pattern learned from historical data"""
    pattern_type: str  # "optimal_sequence", "skip_threshold", "tool_priority"
    dataset_conditions: Dict[str, Any]  # When does this pattern apply?
    learned_value: Any  # The learned parameter/sequence
    confidence: float  # How confident are we? (0-1)
    sample_size: int  # How many audits support this?
    last_updated: str


class MetaLearningEngine:
    """
    SELF-IMPROVEMENT ENGINE
    
    Learns from every audit to continuously improve:
    - Which tool sequences work best
    - Optimal thresholds for different datasets
    - When to skip tools
    - Which strategies are most efficient
    """
    
    def __init__(self, memory_path: str = "agent/meta_learning.pkl"):
        self.memory_path = Path(memory_path)
        
        # Performance tracking
        self.strategy_performances: List[StrategyPerformance] = []
        
        # Learned patterns
        self.learned_patterns: Dict[str, LearnedPattern] = {}
        
        # Dynamic thresholds (learned over time)
        self.adaptive_thresholds = {
            'skip_threshold': 0.45,  # Initial value
            'critical_threshold': 0.7,
            'confidence_threshold': 0.75,
        }
        
        # Tool effectiveness by dataset type
        self.tool_effectiveness: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Optimal sequences discovered
        self.optimal_sequences: Dict[str, List[str]] = {}
        
        self.load_learning()
    
    def learn_from_audit(self, audit_result: Dict[str, Any], 
                        dataset_profile: Dict[str, Any],
                        execution_data: Dict[str, Any]):
        """
        CORE LEARNING FUNCTION
        
        Extract insights from completed audit to improve future decisions.
        """
        # 1. Record performance
        performance = self._extract_performance_metrics(
            audit_result, dataset_profile, execution_data
        )
        self.strategy_performances.append(performance)
        
        # 2. Update tool effectiveness
        self._update_tool_effectiveness(performance, dataset_profile)
        
        # 3. Learn optimal sequences
        self._learn_optimal_sequences(performance, dataset_profile)
        
        # 4. Tune thresholds
        self._tune_thresholds()
        
        # 5. Identify patterns
        self._identify_patterns()
        
        # Save learning
        self.save_learning()
        
        return self._generate_learning_summary()
    
    def _extract_performance_metrics(self, audit_result: Dict, 
                                     dataset_profile: Dict,
                                     execution_data: Dict) -> StrategyPerformance:
        """Extract performance metrics from audit"""
        
        tools_executed = execution_data.get('tools_executed', [])
        execution_times = execution_data.get('execution_times', {})
        findings = execution_data.get('findings', {})
        
        # Calculate time to first critical
        time_to_critical = float('inf')
        cumulative_time = 0
        for tool in tools_executed:
            cumulative_time += execution_times.get(tool, 0)
            tool_findings = findings.get(tool, [])
            if any(f.get('severity') == 'critical' for f in tool_findings):
                time_to_critical = cumulative_time
                break
        
        # Count critical issues
        critical_count = sum(
            sum(1 for f in tool_findings if f.get('severity') == 'critical')
            for tool_findings in findings.values()
        )
        
        # Calculate false positive rate (tools that found nothing)
        tools_with_no_findings = sum(1 for t in tools_executed if not findings.get(t, []))
        false_positive_rate = tools_with_no_findings / len(tools_executed) if tools_executed else 0
        
        # Efficiency score
        total_time = sum(execution_times.values())
        efficiency = critical_count / total_time if total_time > 0 else 0
        
        # Dataset type classification
        dataset_type = self._classify_dataset_type(dataset_profile)
        
        return StrategyPerformance(
            strategy_id=f"strat_{len(self.strategy_performances)}",
            dataset_type=dataset_type,
            tool_sequence=tools_executed,
            time_to_first_critical=time_to_critical,
            total_time=total_time,
            critical_found=critical_count,
            false_positive_rate=false_positive_rate,
            efficiency_score=efficiency,
            goal_achieved=audit_result.get('goal_oriented', {}).get('goal_achieved', False),
            early_stop_triggered=len(tools_executed) < 5,
            timestamp=datetime.now().isoformat(),
            complexity_score=dataset_profile.get('complexity_score', 0)
        )
    
    def _classify_dataset_type(self, profile: Dict) -> str:
        """Classify dataset into type for pattern matching"""
        size = "small" if profile.get('num_rows', 0) < 5000 else "large"
        balance = "imbalanced" if profile.get('class_balance_ratio', 1.0) < 0.3 else "balanced"
        complexity = "simple" if profile.get('complexity_score', 0) < 0.3 else "complex"
        
        return f"{size}_{balance}_{complexity}"
    
    def _update_tool_effectiveness(self, performance: StrategyPerformance, 
                                   dataset_profile: Dict):
        """Learn which tools are most effective for which dataset types"""
        dataset_type = performance.dataset_type
        
        for i, tool in enumerate(performance.tool_sequence):
            # Earlier tools that found critical issues are more effective
            if performance.time_to_first_critical != float('inf'):
                position_weight = 1.0 / (i + 1)  # Earlier = higher weight
                effectiveness = position_weight * performance.efficiency_score
            else:
                effectiveness = 0.5  # Neutral if no critical found
            
            # Update running average
            current = self.tool_effectiveness[dataset_type][tool]
            self.tool_effectiveness[dataset_type][tool] = (current + effectiveness) / 2
    
    def _learn_optimal_sequences(self, performance: StrategyPerformance,
                                 dataset_profile: Dict):
        """
        Learn which tool sequences work best.
        
        Key insight: If a sequence found critical issues quickly,
        remember it for similar datasets.
        """
        dataset_type = performance.dataset_type
        
        # Only learn from successful, efficient audits
        if performance.critical_found > 0 and performance.efficiency_score > 0.1:
            
            # If this is better than what we know
            if dataset_type not in self.optimal_sequences:
                self.optimal_sequences[dataset_type] = performance.tool_sequence
            else:
                # Compare with current best
                current_best = self.optimal_sequences[dataset_type]
                
                # Find historical performance of current best
                best_performances = [
                    p for p in self.strategy_performances
                    if p.dataset_type == dataset_type and p.tool_sequence == current_best
                ]
                
                if best_performances:
                    avg_best_efficiency = np.mean([p.efficiency_score for p in best_performances])
                    
                    if performance.efficiency_score > avg_best_efficiency:
                        # New sequence is better!
                        self.optimal_sequences[dataset_type] = performance.tool_sequence
                        
                        # Record this as a learned pattern
                        self.learned_patterns[f"optimal_seq_{dataset_type}"] = LearnedPattern(
                            pattern_type="optimal_sequence",
                            dataset_conditions={'dataset_type': dataset_type},
                            learned_value=performance.tool_sequence,
                            confidence=min(0.9, len(best_performances) / 10),
                            sample_size=len(best_performances) + 1,
                            last_updated=datetime.now().isoformat()
                        )
    
    def _tune_thresholds(self):
        """
        AUTO-TUNE thresholds based on performance.
        
        Example: If we're skipping too many useful tools, lower skip_threshold.
        """
        if len(self.strategy_performances) < 3:
            return  # Need more data
        
        recent = self.strategy_performances[-10:]  # Last 10 audits
        
        # Tune skip threshold
        # If we're missing critical issues (high false negative rate), lower threshold
        avg_false_pos = np.mean([p.false_positive_rate for p in recent])
        
        if avg_false_pos > 0.4:  # Too many tools finding nothing
            # Increase skip threshold (be more selective)
            self.adaptive_thresholds['skip_threshold'] = min(0.6, 
                self.adaptive_thresholds['skip_threshold'] + 0.05
            )
        elif avg_false_pos < 0.1:  # We're being too selective
            # Decrease skip threshold (run more tools)
            self.adaptive_thresholds['skip_threshold'] = max(0.3,
                self.adaptive_thresholds['skip_threshold'] - 0.05
            )
        
        # Record threshold adjustment as learned pattern
        self.learned_patterns['adaptive_skip_threshold'] = LearnedPattern(
            pattern_type="skip_threshold",
            dataset_conditions={'context': 'global'},
            learned_value=self.adaptive_thresholds['skip_threshold'],
            confidence=min(0.95, len(recent) / 20),
            sample_size=len(recent),
            last_updated=datetime.now().isoformat()
        )
    
    def _identify_patterns(self):
        """
        Identify cross-dataset patterns.
        
        Example: "Datasets with ID features ALWAYS need leakage_detector first"
        """
        if len(self.strategy_performances) < 5:
            return
        
        # Pattern 1: ID features → leakage_detector priority
        id_datasets = [p for p in self.strategy_performances[-20:] 
                      if 'has_id' in str(p.dataset_type).lower() or p.complexity_score > 0.4]
        
        if len(id_datasets) >= 3:
            # How often is leakage_detector first?
            leakage_first_count = sum(
                1 for p in id_datasets 
                if p.tool_sequence and p.tool_sequence[0] == 'leakage_detector'
            )
            
            if leakage_first_count / len(id_datasets) > 0.7:
                self.learned_patterns['id_features_leakage_first'] = LearnedPattern(
                    pattern_type="tool_priority",
                    dataset_conditions={'has_id_features': True},
                    learned_value={'first_tool': 'leakage_detector'},
                    confidence=leakage_first_count / len(id_datasets),
                    sample_size=len(id_datasets),
                    last_updated=datetime.now().isoformat()
                )
        
        # Pattern 2: Imbalanced → bias_detector priority
        imbalanced = [p for p in self.strategy_performances[-20:]
                     if 'imbalanced' in p.dataset_type]
        
        if len(imbalanced) >= 3:
            bias_early_count = sum(
                1 for p in imbalanced
                if p.tool_sequence and 'bias_detector' in p.tool_sequence[:2]
            )
            
            if bias_early_count / len(imbalanced) > 0.7:
                self.learned_patterns['imbalanced_bias_priority'] = LearnedPattern(
                    pattern_type="tool_priority",
                    dataset_conditions={'imbalanced': True},
                    learned_value={'priority_tool': 'bias_detector'},
                    confidence=bias_early_count / len(imbalanced),
                    sample_size=len(imbalanced),
                    last_updated=datetime.now().isoformat()
                )
    
    def get_learned_recommendations(self, dataset_profile: Dict) -> Dict[str, Any]:
        """
        Apply learned knowledge to make better decisions.
        
        Returns recommendations based on meta-learning.
        """
        dataset_type = self._classify_dataset_type(dataset_profile)
        
        recommendations = {
            'optimal_sequence': None,
            'skip_threshold': self.adaptive_thresholds['skip_threshold'],
            'tool_boosts': {},  # Tools to prioritize
            'patterns_matched': []
        }
        
        # 1. Recommend optimal sequence if we've learned one
        if dataset_type in self.optimal_sequences:
            recommendations['optimal_sequence'] = self.optimal_sequences[dataset_type]
        
        # 2. Apply learned patterns
        for pattern_name, pattern in self.learned_patterns.items():
            # Check if pattern applies
            if self._pattern_matches(pattern, dataset_profile):
                recommendations['patterns_matched'].append(pattern_name)
                
                if pattern.pattern_type == 'tool_priority':
                    # Boost this tool's score
                    if 'first_tool' in pattern.learned_value:
                        tool = pattern.learned_value['first_tool']
                        recommendations['tool_boosts'][tool] = 0.3 * pattern.confidence
                    elif 'priority_tool' in pattern.learned_value:
                        tool = pattern.learned_value['priority_tool']
                        recommendations['tool_boosts'][tool] = 0.2 * pattern.confidence
        
        # 3. Tool effectiveness recommendations
        if dataset_type in self.tool_effectiveness:
            effectiveness = self.tool_effectiveness[dataset_type]
            # Sort tools by effectiveness
            sorted_tools = sorted(effectiveness.items(), key=lambda x: x[1], reverse=True)
            recommendations['tool_ranking'] = [tool for tool, _ in sorted_tools]
        
        return recommendations
    
    def _pattern_matches(self, pattern: LearnedPattern, 
                        dataset_profile: Dict) -> bool:
        """Check if a learned pattern applies to current dataset"""
        conditions = pattern.dataset_conditions
        
        # Check each condition
        for key, value in conditions.items():
            if key == 'dataset_type':
                if self._classify_dataset_type(dataset_profile) != value:
                    return False
            elif key == 'has_id_features':
                if dataset_profile.get('has_id_features', False) != value:
                    return False
            elif key == 'imbalanced':
                is_imbalanced = dataset_profile.get('class_balance_ratio', 1.0) < 0.3
                if is_imbalanced != value:
                    return False
        
        return True
    
    def _generate_learning_summary(self) -> Dict[str, Any]:
        """Generate summary of what was learned"""
        return {
            'total_audits_learned_from': len(self.strategy_performances),
            'patterns_discovered': len(self.learned_patterns),
            'optimal_sequences_known': len(self.optimal_sequences),
            'adaptive_thresholds': self.adaptive_thresholds.copy(),
            'recent_improvements': self._get_recent_improvements()
        }
    
    def _get_recent_improvements(self) -> List[str]:
        """Get list of recent improvements made"""
        improvements = []
        
        # Check if we've learned new sequences recently
        recent_patterns = [
            p for p in self.learned_patterns.values()
            if (datetime.now() - datetime.fromisoformat(p.last_updated)).days < 1
        ]
        
        for pattern in recent_patterns:
            if pattern.pattern_type == 'optimal_sequence':
                improvements.append(
                    f"Learned optimal sequence for {pattern.dataset_conditions['dataset_type']}"
                )
            elif pattern.pattern_type == 'skip_threshold':
                improvements.append(
                    f"Tuned skip threshold to {pattern.learned_value:.2f}"
                )
            elif pattern.pattern_type == 'tool_priority':
                improvements.append(
                    f"Identified priority pattern: {pattern.learned_value}"
                )
        
        return improvements[:5]  # Top 5 recent
    
    def get_meta_stats(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning statistics"""
        if not self.strategy_performances:
            return {'learning_active': False}
        
        recent = self.strategy_performances[-10:]
        
        return {
            'learning_active': True,
            'total_audits': len(self.strategy_performances),
            'patterns_learned': len(self.learned_patterns),
            'optimal_sequences': len(self.optimal_sequences),
            'avg_efficiency_recent': np.mean([p.efficiency_score for p in recent]),
            'avg_time_to_critical': np.mean([
                p.time_to_first_critical for p in recent 
                if p.time_to_first_critical != float('inf')
            ]) if recent else 0,
            'improvement_rate': self._calculate_improvement_rate(),
            'adaptive_thresholds': self.adaptive_thresholds,
            'dataset_types_seen': len(set(p.dataset_type for p in self.strategy_performances))
        }
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate how much the system has improved over time"""
        if len(self.strategy_performances) < 10:
            return 0.0
        
        # Compare first 5 vs last 5 audits
        early = self.strategy_performances[:5]
        recent = self.strategy_performances[-5:]
        
        early_avg = np.mean([p.efficiency_score for p in early])
        recent_avg = np.mean([p.efficiency_score for p in recent])
        
        if early_avg == 0:
            return 0.0
        
        improvement = ((recent_avg - early_avg) / early_avg) * 100
        return max(-100, min(100, improvement))  # Cap at ±100%
    
    def save_learning(self):
        """Persist all learned knowledge"""
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'strategy_performances': [p.to_dict() for p in self.strategy_performances],
            'learned_patterns': {k: asdict(v) for k, v in self.learned_patterns.items()},
            'optimal_sequences': self.optimal_sequences,
            'adaptive_thresholds': self.adaptive_thresholds,
            'tool_effectiveness': dict(self.tool_effectiveness),
        }
        
        with open(self.memory_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_learning(self):
        """Load previously learned knowledge"""
        if not self.memory_path.exists():
            return
        
        try:
            with open(self.memory_path, 'rb') as f:
                data = pickle.load(f)
            
            self.strategy_performances = [
                StrategyPerformance.from_dict(p) 
                for p in data.get('strategy_performances', [])
            ]
            
            self.learned_patterns = {
                k: LearnedPattern(**v) 
                for k, v in data.get('learned_patterns', {}).items()
            }
            
            self.optimal_sequences = data.get('optimal_sequences', {})
            self.adaptive_thresholds = data.get('adaptive_thresholds', self.adaptive_thresholds)
            self.tool_effectiveness = defaultdict(
                lambda: defaultdict(float),
                data.get('tool_effectiveness', {})
            )
            
            print(f"📚 Meta-learning loaded: {len(self.strategy_performances)} audits, "
                  f"{len(self.learned_patterns)} patterns")
            
        except Exception as e:
            print(f"⚠️  Could not load meta-learning: {e}")