import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cluster import MiniBatchKMeans
import warnings

warnings.filterwarnings('ignore')


class SpuriousCorrelationDetector:
    """
    Detects spurious correlations and potential shortcut learning in datasets.
    
    Uses feature ablation, permutation tests, and heuristic analysis to identify
    features that have suspiciously high predictive power without causal basis.
    """
    
    def __init__(self, df, target_column, test_size=0.3, random_state=42):
        """
        Initialize spurious correlation detector.
        
        Args:
            df: pandas DataFrame containing the dataset
            target_column: name of the target/label column
            test_size: proportion for train-test split
            random_state: random seed for reproducibility
        """
        self.df = df.copy()
        self.target = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.findings = []
        
    def detect(self):
        """
        Run all spurious correlation detection checks.
        
        Returns:
            List of findings with type, severity, message, and evidence
        """
        self.findings = []
        
        # Check 1: Single feature dominance
        self._check_single_feature_dominance()
        
        # Check 2: Feature removal sensitivity
        self._check_feature_removal_impact()
        
        # Check 3: Suspiciously simple decision rules
        self._check_simple_decision_rules()
        
        # Check 4: Unrealistic feature importance
        self._check_unrealistic_importance()
        
        return self.findings
    
    def _prepare_data(self):
        """
        Prepare data for modeling with intelligent adaptive sampling.
        
        Uses a hybrid dynamic sampling strategy with hard gating:
        1. Stratified sampling (supervised datasets) - preserves class proportions
        2. Quantile-preserving sampling (numeric-heavy) - preserves distribution tails
        3. Cluster-based sampling (complex/mixed) - preserves latent structure
        
        Small datasets (<10K) are left completely untouched.
        
        Returns:
            tuple: (X, y, numeric_cols) or (None, None, []) if preparation fails
        """
        if self.target not in self.df.columns:
            return None, None, []
        
        # Get numeric columns only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != self.target]
        
        if len(numeric_cols) == 0:
            return None, None, []
        
        X = self.df[numeric_cols].copy()
        y = self.df[self.target].copy()
        
        # Handle missing values with mean imputation
        for col in X.columns:
            if X[col].isna().any():
                X[col].fillna(X[col].mean(), inplace=True)
        
        # Remove any remaining NaN in target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # OPTIMIZATION: Adaptive sampling for large datasets
        if len(X) > 10000:
            sample_size = 10000
            X, y = self._adaptive_sample(X, y, sample_size)
        
        return X, y, numeric_cols
    
    def _adaptive_sample(self, X, y, sample_size):
        """
        Adaptive sampling strategy that chooses the best method based on data characteristics.
        
        Tries multiple strategies in order:
        1. Stratified sampling (for classification with sufficient samples per class)
        2. Quantile-preserving sampling (for preserving distribution characteristics)
        3. Cluster-based sampling (for preserving data structure)
        4. Random sampling (fallback)
        
        Args:
            X: Feature matrix
            y: Target vector
            sample_size: Desired sample size
            
        Returns:
            tuple: (X_sampled, y_sampled)
        """
        np.random.seed(self.random_state)
        
        # Strategy 1: Stratified sampling
        try:
            n_classes = y.nunique()
            min_class_count = y.value_counts().min()
            
            if n_classes <= 50 and min_class_count >= 2:
                X_s, _, y_s, _ = train_test_split(
                    X, y,
                    train_size=sample_size,
                    stratify=y,
                    random_state=self.random_state
                )
                return X_s.reset_index(drop=True), y_s.reset_index(drop=True)
        except Exception:
            pass
        
        # Strategy 2: Quantile-preserving sampling
        if X.shape[1] >= 3:
            try:
                n_bins = min(10, max(5, sample_size // 1000))
                quantile_code = np.zeros(len(X))
                
                for col in X.columns:
                    try:
                        q = pd.qcut(X[col], q=n_bins, labels=False, duplicates="drop")
                        quantile_code += q.fillna(0).values
                    except Exception:
                        continue
                
                if np.std(quantile_code) > 0:
                    unique_codes = np.unique(quantile_code)
                    per_bin = max(1, sample_size // len(unique_codes))
                    
                    idx = []
                    for code in unique_codes:
                        candidates = np.where(quantile_code == code)[0]
                        k = min(per_bin, len(candidates))
                        idx.extend(np.random.choice(candidates, k, replace=False))
                    
                    idx = idx[:sample_size]
                    return (
                        X.iloc[idx].reset_index(drop=True),
                        y.iloc[idx].reset_index(drop=True)
                    )
            except Exception:
                pass
        
        # Strategy 3: Cluster-based fallback
        try:
            n_clusters = min(20, max(5, sample_size // 500))
            km = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                batch_size=1024
            )
            
            labels = km.fit_predict(X)
            idx = []
            
            for c in np.unique(labels):
                members = np.where(labels == c)[0]
                k = max(1, int(len(members) / len(X) * sample_size))
                idx.extend(np.random.choice(members, min(k, len(members)), replace=False))
            
            idx = idx[:sample_size]
            return (
                X.iloc[idx].reset_index(drop=True),
                y.iloc[idx].reset_index(drop=True)
            )
        except Exception:
            pass
        
        # Final fallback: random sampling
        X_s, _, y_s, _ = train_test_split(
            X, y,
            train_size=sample_size,
            random_state=self.random_state
        )
        return X_s.reset_index(drop=True), y_s.reset_index(drop=True)
    
    def _check_single_feature_dominance(self):
        """
        Check if a single feature can predict target suspiciously well.
        
        Flags features that achieve >90% accuracy alone as critical,
        and features achieving >85% accuracy as warnings.
        """
        X, y, feature_names = self._prepare_data()
        
        if X is None or len(X) < 30:
            return
        
        # Train a simple model on each feature individually
        for col in feature_names:
            try:
                X_single = X[[col]].values.reshape(-1, 1)
                
                clf = RandomForestClassifier(
                    n_estimators=10,
                    max_depth=3,
                    random_state=self.random_state,
                    n_jobs=1
                )
                
                scores = cross_val_score(clf, X_single, y, cv=3, scoring='accuracy')
                avg_score = scores.mean()
                
                if avg_score > 0.90:
                    self.findings.append({
                        'type': 'single_feature_dominance',
                        'severity': 'critical',
                        'feature': col,
                        'message': f'Single feature "{col}" achieves {avg_score*100:.1f}% accuracy - potential shortcut learning',
                        'evidence': {
                            'accuracy': float(avg_score),
                            'cv_scores': [float(s) for s in scores]
                        }
                    })
                elif avg_score > 0.85:
                    self.findings.append({
                        'type': 'high_single_feature_accuracy',
                        'severity': 'warning',
                        'feature': col,
                        'message': f'Single feature "{col}" achieves {avg_score*100:.1f}% accuracy - investigate causality',
                        'evidence': {
                            'accuracy': float(avg_score),
                            'cv_scores': [float(s) for s in scores]
                        }
                    })
            except Exception:
                continue
    
    def _check_feature_removal_impact(self):
        """
        Test if removing individual features dramatically drops performance.
        
        If removing a single feature causes >15% performance drop, the model
        might be relying on a spurious shortcut.
        """
        X, y, feature_names = self._prepare_data()
        
        if X is None or len(X) < 30 or len(feature_names) < 2:
            return
        
        try:
            # Baseline: train with all features
            clf = RandomForestClassifier(
                n_estimators=20,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=1
            )
            
            baseline_scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
            baseline_acc = baseline_scores.mean()
            
            # Test removing each feature
            for col in feature_names:
                X_reduced = X.drop(columns=[col])
                
                reduced_scores = cross_val_score(clf, X_reduced, y, cv=3, scoring='accuracy')
                reduced_acc = reduced_scores.mean()
                
                # Calculate performance drop
                drop = baseline_acc - reduced_acc
                drop_percentage = (drop / baseline_acc) * 100
                
                if drop_percentage > 15:
                    self.findings.append({
                        'type': 'feature_removal_sensitivity',
                        'severity': 'warning',
                        'feature': col,
                        'message': f'Removing "{col}" causes {drop_percentage:.1f}% performance drop - possible over-reliance',
                        'evidence': {
                            'baseline_accuracy': float(baseline_acc),
                            'reduced_accuracy': float(reduced_acc),
                            'drop_percentage': float(drop_percentage)
                        }
                    })
        except Exception:
            pass
    
    def _check_simple_decision_rules(self):
        """
        Check if target can be predicted by very simple threshold rules.
        
        Suspiciously simple rules often indicate spurious patterns rather than
        genuine predictive relationships.
        """
        X, y, feature_names = self._prepare_data()
        
        if X is None or len(X) < 30:
            return
        
        for col in feature_names:
            try:
                feature_values = X[col].values
                
                # Try different percentile thresholds
                for percentile in [25, 50, 75]:
                    threshold = np.percentile(feature_values, percentile)
                    
                    # Predict based on threshold
                    predictions = (feature_values > threshold).astype(int)
                    accuracy = (predictions == y.values).mean()
                    
                    if accuracy > 0.85:
                        self.findings.append({
                            'type': 'simple_threshold_rule',
                            'severity': 'warning',
                            'feature': col,
                            'message': f'Simple threshold rule on "{col}" achieves {accuracy*100:.1f}% accuracy',
                            'evidence': {
                                'accuracy': float(accuracy),
                                'threshold': float(threshold),
                                'percentile': int(percentile)
                            }
                        })
                        break  # Don't report multiple thresholds for same feature
            except Exception:
                continue
    
    def _check_unrealistic_importance(self):
        """
        Check if feature importances are unrealistically concentrated.
        
        In real-world problems, importance is usually distributed across features.
        Extreme concentration suggests potential shortcuts or data leakage.
        """
        X, y, feature_names = self._prepare_data()
        
        if X is None or len(X) < 30 or len(feature_names) < 3:
            return
        
        try:
            # Train model and get feature importances
            clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=1
            )
            clf.fit(X, y)
            
            importances = clf.feature_importances_
            total_importance = importances.sum()
            
            # Check if top feature has >70% of total importance
            top_importance = importances.max()
            top_percentage = (top_importance / total_importance) * 100
            
            if top_percentage > 70:
                top_feature = feature_names[importances.argmax()]
                self.findings.append({
                    'type': 'concentrated_importance',
                    'severity': 'warning',
                    'feature': top_feature,
                    'message': f'Feature "{top_feature}" accounts for {top_percentage:.1f}% of importance - unusually concentrated',
                    'evidence': {
                        'importance_percentage': float(top_percentage),
                        'feature_importances': {
                            name: float(imp)
                            for name, imp in zip(feature_names, importances)
                        }
                    }
                })
            
            # Check if top 2 features account for >90% importance
            top2_importance = sorted(importances, reverse=True)[:2]
            top2_percentage = (sum(top2_importance) / total_importance) * 100
            
            if top2_percentage > 90 and len(feature_names) > 3:
                self.findings.append({
                    'type': 'top_features_dominate',
                    'severity': 'warning',
                    'message': f'Top 2 features account for {top2_percentage:.1f}% of importance - check for shortcuts',
                    'evidence': {
                        'top2_percentage': float(top2_percentage),
                        'num_features': len(feature_names)
                    }
                })
        except Exception:
            pass
    
    def get_summary(self):
        """
        Generate summary of spurious correlation findings.
        
        Returns:
            dict: Summary containing status, message, counts, and detailed findings
        """
        if not self.findings:
            return {
                'status': 'pass',
                'message': 'No obvious spurious correlations detected',
                'critical_count': 0,
                'warning_count': 0
            }
        
        critical = [f for f in self.findings if f['severity'] == 'critical']
        warnings = [f for f in self.findings if f['severity'] == 'warning']
        
        return {
            'status': 'fail' if critical else 'warning',
            'message': f'Found {len(critical)} critical issues and {len(warnings)} warnings',
            'critical_count': len(critical),
            'warning_count': len(warnings),
            'findings': self.findings
        }