"""
agent/strategy_engine.py - Autonomous Decision-Making Engine

This module enables TRUE autonomous decision-making:
- Analyzes dataset characteristics
- Chooses tools dynamically based on risk profile
- Orders tools by expected impact
- Learns from past audits to improve future decisions
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import pandas as pd


@dataclass
class DatasetProfile:
    """Profile of a dataset's characteristics"""
    num_rows: int
    num_features: int
    num_classes: int
    class_balance_ratio: float  # min_class / max_class
    missing_ratio: float  # % of missing values
    numeric_ratio: float  # % numeric features
    categorical_ratio: float  # % categorical features
    has_temporal_features: bool
    has_id_features: bool
    complexity_score: float  # composite risk score
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class AuditHistory:
    """Historical record of an audit execution"""
    dataset_profile: DatasetProfile
    tools_executed: List[str]
    execution_order: List[str]
    findings_by_tool: Dict[str, int]  # tool -> num critical findings
    execution_times: Dict[str, float]
    total_time: float
    critical_found: int
    success_metrics: Dict[str, float]  # effectiveness scores
    
    def to_dict(self) -> Dict:
        return {
            'dataset_profile': self.dataset_profile.to_dict(),
            'tools_executed': self.tools_executed,
            'execution_order': self.execution_order,
            'findings_by_tool': self.findings_by_tool,
            'execution_times': self.execution_times,
            'total_time': self.total_time,
            'critical_found': self.critical_found,
            'success_metrics': self.success_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        data['dataset_profile'] = DatasetProfile.from_dict(data['dataset_profile'])
        return cls(**data)


class AutonomousStrategyEngine:
    """
    AUTONOMOUS DECISION-MAKING ENGINE
    
    This engine makes independent decisions about:
    1. Which tools to run
    2. In what order
    3. Whether to skip certain tools
    4. How to prioritize based on dataset characteristics
    
    It learns from past audits to improve future decisions.
    """
    
    def __init__(self, memory_path: str = "agent/strategy_memory.pkl"):
        self.memory_path = Path(memory_path)
        self.audit_history: List[AuditHistory] = []
        self.tool_effectiveness: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Available tools and their base characteristics
        self.available_tools = {
            'leakage_detector': {
                'detects': ['data_leakage', 'target_correlation'],
                'cost': 0.3,  # execution time weight
                'critical_rate': 0.15,  # historical % of finding critical issues
                'best_for': ['high_correlation', 'suspicious_features']
            },
            'contamination_detector': {
                'detects': ['train_test_leak', 'duplicates'],
                'cost': 0.2,
                'critical_rate': 0.10,
                'best_for': ['split_datasets', 'duplicate_risk']
            },
            'bias_detector': {
                'detects': ['class_imbalance', 'distribution_shift'],
                'cost': 0.2,
                'critical_rate': 0.25,
                'best_for': ['imbalanced_classes', 'categorical_heavy']
            },
            'spurious_detector': {
                'detects': ['spurious_correlation', 'shortcut_learning'],
                'cost': 0.5,  # most expensive
                'critical_rate': 0.20,
                'best_for': ['complex_features', 'high_accuracy_risk']
            },
            'feature_utility': {
                'detects': ['low_variance', 'redundant_features'],
                'cost': 0.3,
                'critical_rate': 0.05,
                'best_for': ['many_features', 'quality_check']
            }
        }
        
        self.load_history()
    
    def profile_dataset(self, df: pd.DataFrame, target_column: str) -> DatasetProfile:
        """
        Analyze dataset to create risk profile.
        This is the foundation for autonomous decision-making.
        """
        num_rows = len(df)
        num_features = len(df.columns) - 1
        
        # Analyze target
        target = df[target_column]
        num_classes = target.nunique()
        class_counts = target.value_counts()
        class_balance_ratio = class_counts.min() / class_counts.max() if len(class_counts) > 1 else 1.0
        
        # Missing values
        missing_ratio = df.isnull().sum().sum() / (num_rows * len(df.columns))
        
        # Feature types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numeric_ratio = len(numeric_cols) / len(df.columns)
        categorical_ratio = len(categorical_cols) / len(df.columns)
        
        # Temporal features (common patterns)
        temporal_keywords = ['date', 'time', 'year', 'month', 'day', 'timestamp']
        has_temporal = any(
            any(kw in col.lower() for kw in temporal_keywords)
            for col in df.columns
        )
        
        # ID features (potential leakage risk)
        id_keywords = ['id', 'index', 'key', 'identifier']
        has_id_features = any(
            any(kw in col.lower() for kw in id_keywords)
            for col in df.columns
        )
        
        # Calculate complexity score (higher = more risk)
        complexity_score = self._calculate_complexity(
            num_rows, num_features, num_classes, class_balance_ratio,
            missing_ratio, has_temporal, has_id_features
        )
        
        return DatasetProfile(
            num_rows=num_rows,
            num_features=num_features,
            num_classes=num_classes,
            class_balance_ratio=class_balance_ratio,
            missing_ratio=missing_ratio,
            numeric_ratio=numeric_ratio,
            categorical_ratio=categorical_ratio,
            has_temporal_features=has_temporal,
            has_id_features=has_id_features,
            complexity_score=complexity_score
        )
    
    def _calculate_complexity(self, rows, features, classes, balance, missing,
                             temporal, has_id) -> float:
        """Calculate dataset complexity/risk score (0-1)"""
        score = 0.0
        
        # Size factors
        if rows > 100000:
            score += 0.1
        if features > 50:
            score += 0.15
        
        # Balance issues
        if balance < 0.3:
            score += 0.2
        elif balance < 0.5:
            score += 0.1
        
        # Missing data
        if missing > 0.2:
            score += 0.15
        elif missing > 0.1:
            score += 0.08
        
        # Risk factors
        if temporal:
            score += 0.15  # temporal leakage risk
        if has_id:
            score += 0.1   # ID leakage risk
        
        # Multi-class complexity
        if classes > 10:
            score += 0.1
        
        return min(1.0, score)
    
    def decide_audit_strategy(self, profile: DatasetProfile) -> Tuple[List[str], Dict[str, str]]: 
        """ 
        AUTONOMOUS DECISION: Choose which tools to run and in what order. 
        
        This is where true agency happens - the system decides independently 
        based on dataset characteristics and learned experience. 
        
        Returns: 
            (ordered_tools, reasoning) - tools to execute and why 
        """ 
        print(f"\n🧠 AUTONOMOUS STRATEGY PLANNING") 
        print(f"   Dataset Profile: {profile.num_rows} rows, {profile.num_features} features") 
        print(f"   Complexity Score: {profile.complexity_score:.2f}") 
        print(f"   Class Balance: {profile.class_balance_ratio:.2f}") 
        
        # NEW: Get learned recommendations from meta-learner
        learned_recs = self.meta_learner.get_learned_recommendations(profile)
        
        # Initialize decision scores for each tool 
        tool_scores = {} 
        reasoning = {} 
        
        for tool_name, tool_info in self.available_tools.items(): 
            score = self._score_tool_relevance(tool_name, tool_info, profile) 
            tool_scores[tool_name] = score 
            reasoning[tool_name] = self._explain_score(tool_name, tool_info, profile, score) 
        
        # NEW: Apply tool boosts from learned patterns
        for tool, boost in learned_recs.get('tool_boosts', {}).items():
            if tool in tool_scores:
                original_score = tool_scores[tool]
                tool_scores[tool] += boost
                print(f"   ⬆️  Boosting {tool}: {original_score:.2f} → {tool_scores[tool]:.2f} (learned pattern match)")
                reasoning[tool] += f" [+{boost:.2f} from learned patterns]"
        
        # Sort tools by score (highest first) 
        ordered_tools = sorted(tool_scores.keys(), key=lambda t: tool_scores[t], reverse=True) 
        
        # NEW: Apply learned optimal sequence if available
        if learned_recs.get('optimal_sequence'):
            print(f"   🧠 Using learned optimal sequence")
            ordered_tools = learned_recs['optimal_sequence']
        
        # Decide which tools to skip (score below threshold)
        SKIP_THRESHOLD = 0.45
        
        # NEW: Apply learned threshold if available
        if learned_recs.get('skip_threshold'):
            SKIP_THRESHOLD = learned_recs['skip_threshold']
            print(f"   🎚️  Using learned skip threshold: {SKIP_THRESHOLD:.2f}")
        
        tools_to_run = [t for t in ordered_tools if tool_scores[t] >= SKIP_THRESHOLD] 
        skipped_tools = [t for t in ordered_tools if tool_scores[t] < SKIP_THRESHOLD] 
        
        # Add skip reasoning 
        for tool in skipped_tools: 
            reasoning[tool] = f"SKIPPED (score={tool_scores[tool]:.2f}): {reasoning[tool]}" 
        
        print(f"\n   📋 DECISION: Running {len(tools_to_run)}/{len(self.available_tools)} tools") 
        print(f"   ⊘ Skipping: {skipped_tools if skipped_tools else 'None'}") 
        print(f"\n   🎯 Execution Order (by priority):") 
        for i, tool in enumerate(tools_to_run, 1): 
            print(f"      {i}. {tool} (score: {tool_scores[tool]:.2f})") 
        
        return tools_to_run, reasoning
    def _score_tool_relevance(self, tool_name: str, tool_info: Dict, 
                             profile: DatasetProfile) -> float:
        """
        Score how relevant a tool is for this specific dataset.
        
        Combines:
        - Dataset characteristics match
        - Historical effectiveness
        - Cost-benefit analysis
        - Learned patterns
        """
        score = 0.0
        
        # 1. BASE RELEVANCE (dataset characteristics)
        if tool_name == 'leakage_detector':
            # High priority if ID features or temporal data
            if profile.has_id_features:
                score += 0.4
            if profile.has_temporal_features:
                score += 0.3
            # Always somewhat relevant
            score += 0.3
        
        elif tool_name == 'bias_detector':
            # Critical for imbalanced datasets
            if profile.class_balance_ratio < 0.5:
                score += 0.5
            if profile.class_balance_ratio < 0.3:
                score += 0.3  # extra boost for severe imbalance
            # Missing data increases relevance
            score += profile.missing_ratio * 0.3
            # Always check for bias
            score += 0.2
        
        elif tool_name == 'contamination_detector':
            # More relevant for smaller datasets (easier to have leaks)
            if profile.num_rows < 10000:
                score += 0.3
            # Always important for train/test integrity
            score += 0.4
        
        elif tool_name == 'spurious_detector':
            # More relevant for complex datasets
            score += profile.complexity_score * 0.4
            # High feature count increases spurious risk
            if profile.num_features > 20:
                score += 0.3
            # Base relevance
            score += 0.2
        
        elif tool_name == 'feature_utility':
            # More relevant with many features
            if profile.num_features > 30:
                score += 0.4
            elif profile.num_features > 15:
                score += 0.2
            # Base relevance
            score += 0.3
        
        # 2. HISTORICAL EFFECTIVENESS
        # Learn from past audits on similar datasets
        historical_boost = self._get_historical_effectiveness(tool_name, profile)
        score += historical_boost * 0.3
        
        # 3. COST-BENEFIT ADJUSTMENT
        # Penalize expensive tools slightly if low priority
        cost_penalty = tool_info['cost'] * 0.1
        score -= cost_penalty
        
        # 4. CRITICAL RATE BOOST
        # Tools that historically find critical issues get priority
        score += tool_info['critical_rate'] * 0.2
        
        return min(1.0, max(0.0, score))
    
    def _get_historical_effectiveness(self, tool_name: str, 
                                     profile: DatasetProfile) -> float:
        """
        Learn from past audits: how effective was this tool on similar datasets?
        
        This implements meta-learning for continuous improvement.
        """
        if not self.audit_history:
            return 0.5  # neutral if no history
        
        # Find similar datasets
        similar_audits = [
            audit for audit in self.audit_history
            if self._is_similar_dataset(audit.dataset_profile, profile)
        ]
        
        if not similar_audits:
            return 0.5
        
        # Calculate average effectiveness
        effectiveness_scores = []
        for audit in similar_audits:
            if tool_name in audit.findings_by_tool:
                # Effectiveness = findings found / time spent
                findings = audit.findings_by_tool[tool_name]
                time = audit.execution_times.get(tool_name, 1.0)
                effectiveness = min(1.0, findings / max(1.0, time))
                effectiveness_scores.append(effectiveness)
        
        if effectiveness_scores:
            return np.mean(effectiveness_scores)
        return 0.5
    
    def _is_similar_dataset(self, profile1: DatasetProfile, 
                           profile2: DatasetProfile) -> bool:
        """Determine if two datasets are similar enough to compare"""
        # Similar size
        size_diff = abs(profile1.num_rows - profile2.num_rows) / max(profile1.num_rows, profile2.num_rows)
        if size_diff > 0.5:
            return False
        
        # Similar feature count
        feature_diff = abs(profile1.num_features - profile2.num_features) / max(profile1.num_features, profile2.num_features)
        if feature_diff > 0.5:
            return False
        
        # Similar complexity
        complexity_diff = abs(profile1.complexity_score - profile2.complexity_score)
        if complexity_diff > 0.3:
            return False
        
        return True
    
    def _explain_score(self, tool_name: str, tool_info: Dict,
                      profile: DatasetProfile, score: float) -> str:
        """Generate human-readable reasoning for the decision"""
        reasons = []
        
        if tool_name == 'leakage_detector':
            if profile.has_id_features:
                reasons.append("ID features detected (high leakage risk)")
            if profile.has_temporal_features:
                reasons.append("temporal features present")
            if not reasons:
                reasons.append("standard leakage screening")
        
        elif tool_name == 'bias_detector':
            if profile.class_balance_ratio < 0.3:
                reasons.append("severe class imbalance detected")
            elif profile.class_balance_ratio < 0.5:
                reasons.append("moderate class imbalance")
            if profile.missing_ratio > 0.1:
                reasons.append(f"significant missing data ({profile.missing_ratio:.1%})")
        
        elif tool_name == 'contamination_detector':
            if profile.num_rows < 10000:
                reasons.append("small dataset (higher contamination risk)")
            else:
                reasons.append("train/test integrity check")
        
        elif tool_name == 'spurious_detector':
            if profile.complexity_score > 0.6:
                reasons.append("high complexity score")
            if profile.num_features > 20:
                reasons.append(f"many features ({profile.num_features})")
        
        elif tool_name == 'feature_utility':
            if profile.num_features > 30:
                reasons.append(f"high feature count ({profile.num_features})")
            else:
                reasons.append("feature quality assessment")
        
        if not reasons:
            reasons.append("baseline audit coverage")
        
        return ", ".join(reasons)
    
    def record_audit_outcome(self, profile: DatasetProfile, 
                            tools_executed: List[str],
                            execution_order: List[str],
                            findings: Dict[str, List],
                            execution_times: Dict[str, float],
                            total_time: float):
        """
        Record audit results for future learning.
        
        This builds the knowledge base for autonomous improvement.
        """
        # Calculate findings per tool
        findings_by_tool = {}
        for tool, tool_findings in findings.items():
            critical_count = sum(1 for f in tool_findings if f.get('severity') == 'critical')
            findings_by_tool[tool] = critical_count
        
        total_critical = sum(findings_by_tool.values())
        
        # Calculate success metrics
        success_metrics = {
            'efficiency': len(tools_executed) / total_time if total_time > 0 else 0,
            'critical_per_tool': total_critical / len(tools_executed) if tools_executed else 0,
            'coverage': len(tools_executed) / len(self.available_tools)
        }
        
        # Create history record
        history = AuditHistory(
            dataset_profile=profile,
            tools_executed=tools_executed,
            execution_order=execution_order,
            findings_by_tool=findings_by_tool,
            execution_times=execution_times,
            total_time=total_time,
            critical_found=total_critical,
            success_metrics=success_metrics
        )
        
        self.audit_history.append(history)
        
        # Update effectiveness scores
        for tool in tools_executed:
            self._update_tool_effectiveness(tool, history)
        
        # Save updated history
        self.save_history()
        
        print(f"\n📚 LEARNING: Audit outcome recorded for future strategy optimization")
        print(f"   Tools executed: {len(tools_executed)}")
        print(f"   Critical findings: {total_critical}")
        print(f"   Total history: {len(self.audit_history)} audits")
    
    def _update_tool_effectiveness(self, tool_name: str, history: AuditHistory):
        """Update learned effectiveness scores"""
        profile_key = self._get_profile_key(history.dataset_profile)
        
        # Effectiveness = findings / time
        findings = history.findings_by_tool.get(tool_name, 0)
        time = history.execution_times.get(tool_name, 1.0)
        effectiveness = findings / max(0.1, time)
        
        # Update running average
        current = self.tool_effectiveness[tool_name].get(profile_key, effectiveness)
        self.tool_effectiveness[tool_name][profile_key] = (current + effectiveness) / 2
    
    def _get_profile_key(self, profile: DatasetProfile) -> str:
        """Generate key for grouping similar datasets"""
        size_bucket = "small" if profile.num_rows < 10000 else "large"
        feature_bucket = "few" if profile.num_features < 20 else "many"
        balance_bucket = "balanced" if profile.class_balance_ratio > 0.5 else "imbalanced"
        
        return f"{size_bucket}_{feature_bucket}_{balance_bucket}"
    
    def save_history(self):
        """Persist learning history"""
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'audit_history': [h.to_dict() for h in self.audit_history],
            'tool_effectiveness': dict(self.tool_effectiveness)
        }
        
        with open(self.memory_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_history(self):
        """Load previous learning history"""
        if not self.memory_path.exists():
            return
        
        try:
            with open(self.memory_path, 'rb') as f:
                data = pickle.load(f)
            
            self.audit_history = [
                AuditHistory.from_dict(h) for h in data.get('audit_history', [])
            ]
            self.tool_effectiveness = defaultdict(
                lambda: defaultdict(float),
                data.get('tool_effectiveness', {})
            )
            
            print(f"📚 Loaded {len(self.audit_history)} historical audits for autonomous learning")
        except Exception as e:
            print(f"⚠️  Could not load history: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about what the system has learned"""
        if not self.audit_history:
            return {'total_audits': 0, 'learning_active': False}
        
        # Calculate tool usage patterns
        tool_usage = defaultdict(int)
        for audit in self.audit_history:
            for tool in audit.tools_executed:
                tool_usage[tool] += 1
        
        # Find most effective tools
        avg_effectiveness = {}
        for tool, profiles in self.tool_effectiveness.items():
            if profiles:
                avg_effectiveness[tool] = np.mean(list(profiles.values()))
        
        return {
            'total_audits': len(self.audit_history),
            'learning_active': True,
            'tool_usage_count': dict(tool_usage),
            'avg_tool_effectiveness': avg_effectiveness,
            'unique_dataset_profiles': len(set(
                self._get_profile_key(a.dataset_profile) for a in self.audit_history
            ))
        }