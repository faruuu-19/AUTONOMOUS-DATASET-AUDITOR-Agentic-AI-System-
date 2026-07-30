import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans


class BiasDetector:
    """
    Detects bias and imbalance issues in datasets.
    Analyzes class distribution, feature skewness, missing patterns,
    and potential demographic biases.
    """
    
    def __init__(self, df, target_column, random_state=42, max_cols=50):
        """
        Initialize bias detector.
        
        Args:
            df: pandas DataFrame containing the dataset
            target_column: name of the target/label column
            random_state: random seed for reproducibility
            max_cols: maximum number of columns to analyze (excluding target)
        """
        self.df = df.copy()
        self.target = target_column
        self.random_state = random_state
        self.max_cols = max_cols
        self.findings = []
        self.original_size = len(df)
        self.original_cols = len(df.columns)
        
        # Apply column sampling for wide datasets
        if len(self.df.columns) > max_cols + 1:  # +1 for target column
            self.df = self._sample_columns(self.df)
        
        # Apply row sampling for large datasets (>10000 rows)
        # Force sampling to avoid performance issues
        if len(self.df) > 10000:
            self.df = self._adaptive_sample(self.df, sample_size=10000)
            print(f"Sampled dataset from {self.original_size} to {len(self.df)} rows")
        
    def detect(self):
        """
        Run all bias and imbalance detection checks.
        
        Returns:
            List of findings with type, severity, message, and evidence
        """
        self.findings = []
        
        # Check 1: Class imbalance
        self._check_class_imbalance()
        
        # Check 2: Feature distribution bias across classes
        self._check_feature_distribution_bias()
        
        # Check 3: Missing value patterns correlated with target
        self._check_missing_value_bias()
        
        # Check 4: Extreme feature skewness
        self._check_feature_skewness()
        
        # Check 5: Sample size issues
        self._check_sample_size()
        
        return self.findings
    
    def _sample_columns(self, df):
        """
        Sample columns intelligently to reduce dimensionality while preserving analysis quality.
        
        Strategy:
        1. Always keep the target column
        2. Prioritize columns with high variance (more informative)
        3. Ensure representation from different data types
        4. Sample from both numeric and categorical features
        
        Args:
            df: DataFrame to sample columns from
            
        Returns:
            DataFrame: Column-sampled data
        """
        np.random.seed(self.random_state)
        
        # Always keep target
        selected_cols = [self.target] if self.target in df.columns else []
        remaining_budget = self.max_cols
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Remove target from both lists
        numeric_cols = [c for c in numeric_cols if c != self.target]
        categorical_cols = [c for c in categorical_cols if c != self.target]
        
        # Calculate how many of each type to keep (proportional to original distribution)
        total_features = len(numeric_cols) + len(categorical_cols)
        if total_features == 0:
            return df[selected_cols] if selected_cols else df
        
        numeric_ratio = len(numeric_cols) / total_features
        n_numeric = min(len(numeric_cols), int(remaining_budget * numeric_ratio))
        n_categorical = min(len(categorical_cols), remaining_budget - n_numeric)
        
        # Sample numeric columns by variance (high variance = more informative)
        if numeric_cols and n_numeric > 0:
            try:
                variances = df[numeric_cols].var()
                variances = variances.fillna(0)
                top_numeric = variances.nlargest(n_numeric).index.tolist()
                selected_cols.extend(top_numeric)
            except Exception:
                # Fallback to random sampling
                selected_cols.extend(np.random.choice(
                    numeric_cols, 
                    size=min(n_numeric, len(numeric_cols)), 
                    replace=False
                ).tolist())
        
        # Sample categorical columns by uniqueness (more unique values = more informative)
        if categorical_cols and n_categorical > 0:
            try:
                uniqueness = df[categorical_cols].nunique()
                top_categorical = uniqueness.nlargest(n_categorical).index.tolist()
                selected_cols.extend(top_categorical)
            except Exception:
                # Fallback to random sampling
                selected_cols.extend(np.random.choice(
                    categorical_cols,
                    size=min(n_categorical, len(categorical_cols)),
                    replace=False
                ).tolist())
        
        return df[selected_cols]
    
    def _adaptive_sample(self, df, sample_size):
        """
        Adaptive sampling strategy that chooses the best method based on data characteristics.
        
        Tries multiple strategies in order:
        1. Stratified sampling (for classification with sufficient samples per class)
        2. Quantile-preserving sampling (for preserving distribution characteristics)
        3. Cluster-based sampling (for preserving data structure)
        4. Random sampling (fallback)
        
        Args:
            df: DataFrame to sample
            sample_size: Desired sample size
            
        Returns:
            DataFrame: Sampled data
        """
        np.random.seed(self.random_state)
        
        # Ensure sample_size doesn't exceed data size
        sample_size = min(sample_size, len(df))
        
        # Strategy 1: Stratified sampling
        if self.target in df.columns:
            try:
                y = df[self.target]
                n_classes = y.nunique()
                min_class_count = y.value_counts().min()
                
                if n_classes <= 50 and min_class_count >= 2:
                    sampled_df, _ = train_test_split(
                        df,
                        train_size=sample_size,
                        stratify=y,
                        random_state=self.random_state
                    )
                    return sampled_df.reset_index(drop=True)
            except Exception:
                pass
        
        # Strategy 2: Quantile-preserving sampling
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            try:
                n_bins = min(10, max(5, sample_size // 1000))
                quantile_code = np.zeros(len(df))
                
                for col in numeric_cols:
                    try:
                        q = pd.qcut(df[col], q=n_bins, labels=False, duplicates="drop")
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
                    return df.iloc[idx].reset_index(drop=True)
            except Exception:
                pass
        
        # Strategy 3: Cluster-based fallback
        if len(numeric_cols) > 0:
            try:
                n_clusters = min(20, max(5, sample_size // 500))
                X = df[numeric_cols].fillna(df[numeric_cols].mean())
                
                km = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    batch_size=1024
                )
                
                labels = km.fit_predict(X)
                idx = []
                
                for c in np.unique(labels):
                    members = np.where(labels == c)[0]
                    k = max(1, int(len(members) / len(df) * sample_size))
                    idx.extend(np.random.choice(members, min(k, len(members)), replace=False))
                
                idx = idx[:sample_size]
                return df.iloc[idx].reset_index(drop=True)
            except Exception:
                pass
        
        # Final fallback: random sampling
        sampled_df, _ = train_test_split(
            df,
            train_size=sample_size,
            random_state=self.random_state
        )
        return sampled_df.reset_index(drop=True)
    
    def _check_class_imbalance(self):
        """Detect severe class imbalance in target variable."""
        
        if self.target not in self.df.columns:
            return
        
        target_counts = self.df[self.target].value_counts()
        total = len(self.df)
        
        if len(target_counts) < 2:
            self.findings.append({
                'type': 'single_class',
                'severity': 'critical',
                'message': 'Target variable has only one class - cannot train classifier',
                'evidence': {
                    'class_count': len(target_counts),
                    'unique_values': target_counts.index.tolist()
                }
            })
            return
        
        # Calculate imbalance ratio
        majority_count = target_counts.max()
        minority_count = target_counts.min()
        imbalance_ratio = majority_count / minority_count
        
        # Calculate percentages
        percentages = (target_counts / total * 100).to_dict()
        
        if imbalance_ratio > 100:
            self.findings.append({
                'type': 'severe_imbalance',
                'severity': 'critical',
                'message': f'Severe class imbalance detected (ratio {imbalance_ratio:.1f}:1)',
                'evidence': {
                    'imbalance_ratio': float(imbalance_ratio),
                    'class_distribution': percentages,
                    'majority_class': str(target_counts.idxmax()),
                    'minority_class': str(target_counts.idxmin())
                }
            })
        elif imbalance_ratio > 10:
            self.findings.append({
                'type': 'moderate_imbalance',
                'severity': 'warning',
                'message': f'Moderate class imbalance detected (ratio {imbalance_ratio:.1f}:1)',
                'evidence': {
                    'imbalance_ratio': float(imbalance_ratio),
                    'class_distribution': percentages
                }
            })
        elif imbalance_ratio > 3:
            self.findings.append({
                'type': 'mild_imbalance',
                'severity': 'info',
                'message': f'Mild class imbalance detected (ratio {imbalance_ratio:.1f}:1)',
                'evidence': {
                    'imbalance_ratio': float(imbalance_ratio),
                    'class_distribution': percentages
                }
            })
    
    def _check_feature_distribution_bias(self):
        """Check if feature distributions differ significantly across classes."""
        
        if self.target not in self.df.columns:
            return
        
        target_values = self.df[self.target].unique()
        if len(target_values) < 2 or len(target_values) > 50:
            return  # Skip if too few or too many classes
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target]
        
        # Limit columns to check
        if len(numeric_cols) > 50:
            np.random.seed(self.random_state)
            numeric_cols = np.random.choice(numeric_cols, size=50, replace=False).tolist()
        
        for col in numeric_cols:
            if self.df[col].isna().all():
                continue
            
            try:
                # Perform Kruskal-Wallis test (non-parametric alternative to ANOVA)
                groups = [self.df[self.df[self.target] == val][col].dropna() 
                         for val in target_values]
                
                # Filter out empty groups
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) < 2:
                    continue
                
                statistic, p_value = stats.kruskal(*groups)
                
                # Calculate effect size (eta-squared approximation)
                n = sum(len(g) for g in groups)
                k = len(groups)
                if n > k:
                    eta_squared = (statistic - k + 1) / (n - k)
                else:
                    eta_squared = 0
                
                # Significant difference with large effect size
                if p_value < 0.001 and eta_squared > 0.14:
                    self.findings.append({
                        'type': 'feature_distribution_bias',
                        'severity': 'warning',
                        'feature': col,
                        'message': f'Feature "{col}" has significantly different distributions across classes',
                        'evidence': {
                            'p_value': float(p_value),
                            'effect_size': float(eta_squared),
                            'test': 'kruskal_wallis'
                        }
                    })
                    
            except Exception:
                continue
    
    def _check_missing_value_bias(self):
        """Check if missing values correlate with target variable."""
        
        if self.target not in self.df.columns:
            return
        
        target_data = self.df[self.target]
        
        # Limit number of columns to check for performance
        cols_to_check = [col for col in self.df.columns if col != self.target]
        if len(cols_to_check) > 100:
            # Sample columns if too many
            np.random.seed(self.random_state)
            cols_to_check = np.random.choice(cols_to_check, size=100, replace=False).tolist()
        
        for col in cols_to_check:
            missing_mask = self.df[col].isna()
            missing_count = missing_mask.sum()
            
            if missing_count == 0 or missing_count == len(self.df):
                continue
            
            # Skip if missing rate is very low or very high (not informative)
            missing_rate = missing_count / len(self.df)
            if missing_rate < 0.01 or missing_rate > 0.99:
                continue
            
            try:
                # Calculate missing rate by class
                missing_by_class = {}
                for target_val in target_data.unique():
                    class_mask = target_data == target_val
                    if class_mask.sum() == 0:
                        continue
                    class_missing_rate = missing_mask[class_mask].sum() / class_mask.sum()
                    missing_by_class[str(target_val)] = float(class_missing_rate * 100)
                
                if len(missing_by_class) < 2:
                    continue
                
                # Check if missing rates differ significantly
                max_rate = max(missing_by_class.values())
                min_rate = min(missing_by_class.values())
                
                if max_rate - min_rate > 20:  # More than 20% difference
                    self.findings.append({
                        'type': 'missing_value_bias',
                        'severity': 'warning',
                        'feature': col,
                        'message': f'Missing values in "{col}" are unevenly distributed across classes',
                        'evidence': {
                            'missing_by_class': missing_by_class,
                            'max_difference': float(max_rate - min_rate),
                            'overall_missing_rate': float(missing_count / len(self.df) * 100)
                        }
                    })
                
                # Perform chi-square test only if reasonable number of classes
                if len(target_data.unique()) <= 20:
                    try:
                        # Create contingency table
                        contingency = pd.crosstab(missing_mask, target_data)
                        chi2, p_value, dof, expected = chi2_contingency(contingency)
                        
                        if p_value < 0.01:
                            self.findings.append({
                                'type': 'missing_pattern_correlation',
                                'severity': 'warning',
                                'feature': col,
                                'message': f'Missing pattern in "{col}" significantly correlates with target',
                                'evidence': {
                                    'chi2': float(chi2),
                                    'p_value': float(p_value),
                                    'missing_by_class': missing_by_class
                                }
                            })
                    except Exception:
                        continue
            except Exception:
                continue
    
    def _check_feature_skewness(self):
        """Check for extreme skewness in numeric features."""
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target]
        
        # Limit columns to check
        if len(numeric_cols) > 50:
            np.random.seed(self.random_state)
            numeric_cols = np.random.choice(numeric_cols, size=50, replace=False).tolist()
        
        for col in numeric_cols:
            if self.df[col].isna().all():
                continue
            
            try:
                skewness = self.df[col].skew()
                
                if abs(skewness) > 5:
                    self.findings.append({
                        'type': 'extreme_skewness',
                        'severity': 'warning',
                        'feature': col,
                        'message': f'Feature "{col}" has extreme skewness ({skewness:.2f})',
                        'evidence': {
                            'skewness': float(skewness),
                            'mean': float(self.df[col].mean()),
                            'median': float(self.df[col].median())
                        }
                    })
            except Exception:
                continue
    
    def _check_sample_size(self):
        """Check if dataset has sufficient samples for each class."""
        
        if self.target not in self.df.columns:
            return
        
        # Use original size for this check, not sampled size
        total_samples = self.original_size
        target_counts = self.df[self.target].value_counts()
        
        # Scale up counts if we sampled
        if len(self.df) < self.original_size:
            scale_factor = self.original_size / len(self.df)
            target_counts = (target_counts * scale_factor).astype(int)
        
        # Check overall sample size
        if total_samples < 100:
            self.findings.append({
                'type': 'small_dataset',
                'severity': 'warning',
                'message': f'Dataset has only {total_samples} samples - may be insufficient for robust training',
                'evidence': {
                    'total_samples': int(total_samples)
                }
            })
        
        # Check per-class sample size
        min_class_size = target_counts.min()
        if min_class_size < 30:
            self.findings.append({
                'type': 'small_class_size',
                'severity': 'critical' if min_class_size < 10 else 'warning',
                'message': f'Smallest class has only {min_class_size} samples',
                'evidence': {
                    'min_class_size': int(min_class_size),
                    'class_distribution': target_counts.to_dict()
                }
            })
    
    def get_summary(self):
        """Generate summary of bias findings."""
        
        if not self.findings:
            return {
                'status': 'pass',
                'message': 'No significant bias or imbalance detected',
                'critical_count': 0,
                'warning_count': 0,
                'info_count': 0
            }
        
        critical = [f for f in self.findings if f['severity'] == 'critical']
        warnings = [f for f in self.findings if f['severity'] == 'warning']
        info = [f for f in self.findings if f['severity'] == 'info']
        
        return {
            'status': 'fail' if critical else ('warning' if warnings else 'info'),
            'message': f'Found {len(critical)} critical, {len(warnings)} warnings, {len(info)} info',
            'critical_count': len(critical),
            'warning_count': len(warnings),
            'info_count': len(info),
            'findings': self.findings
        }