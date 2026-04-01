import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class FeatureUtilityDetector:
    """
    Evaluates feature utility and identifies low-value features.
    Detects constant features, low-variance features, redundant features,
    and features with low predictive information.
    """
    
    def __init__(self, df, target_column):
        """
        Initialize feature utility detector.
        
        Args:
            df: pandas DataFrame containing the dataset
            target_column: name of the target/label column
        """
        self.df = df.copy()
        self.target = target_column
        self.findings = []
        
    def detect(self):
        """
        Run all feature utility checks.
        
        Returns:
            List of findings with type, severity, message, and evidence
        """
        self.findings = []
        
        # Check 1: Constant and near-constant features
        self._check_constant_features()
        
        # Check 2: Low variance features
        self._check_low_variance()
        
        # Check 3: Highly correlated (redundant) features
        self._check_redundant_features()
        
        # Check 4: Low information features
        self._check_low_information()
        
        # Check 5: High cardinality categorical features
        self._check_high_cardinality()
        
        # Check 6: Features with excessive missing values
        self._check_excessive_missing()
        
        return self.findings
    
    def _check_constant_features(self):
        """Detect features with constant or near-constant values."""
        
        for col in self.df.columns:
            if col == self.target:
                continue
            
            try:
                unique_count = self.df[col].nunique()
                total_count = len(self.df)
                
                # Constant feature (only 1 unique value)
                if unique_count == 1:
                    self.findings.append({
                        'type': 'constant_feature',
                        'severity': 'critical',
                        'feature': col,
                        'message': f'Feature "{col}" has only one unique value - provides no information',
                        'evidence': {
                            'unique_values': int(unique_count),
                            'constant_value': str(self.df[col].iloc[0])
                        }
                    })
                
                # Near-constant feature (>95% same value)
                elif unique_count > 1:
                    value_counts = self.df[col].value_counts()
                    most_common_freq = value_counts.iloc[0] / total_count
                    
                    if most_common_freq > 0.95:
                        self.findings.append({
                            'type': 'near_constant_feature',
                            'severity': 'warning',
                            'feature': col,
                            'message': f'Feature "{col}" has {most_common_freq*100:.1f}% same value - very low variance',
                            'evidence': {
                                'unique_values': int(unique_count),
                                'most_common_frequency': float(most_common_freq),
                                'most_common_value': str(value_counts.index[0])
                            }
                        })
            except Exception as e:
                continue
    
    def _check_low_variance(self):
        """Check for numeric features with very low variance."""
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target]
        
        for col in numeric_cols:
            try:
                if self.df[col].isna().all():
                    continue
                
                # Calculate coefficient of variation (CV = std / mean)
                mean = self.df[col].mean()
                std = self.df[col].std()
                
                if abs(mean) > 1e-10:  # Avoid division by zero
                    cv = abs(std / mean)
                    
                    # Very low coefficient of variation
                    if cv < 0.01:
                        self.findings.append({
                            'type': 'low_variance_numeric',
                            'severity': 'warning',
                            'feature': col,
                            'message': f'Feature "{col}" has very low variance (CV={cv:.4f})',
                            'evidence': {
                                'coefficient_of_variation': float(cv),
                                'mean': float(mean),
                                'std': float(std)
                            }
                        })
            except Exception as e:
                continue
    
    def _check_redundant_features(self):
        """Detect highly correlated (redundant) feature pairs."""
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target]
        
        if len(numeric_cols) < 2:
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr().abs()
        
        # Find highly correlated pairs
        redundant_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    redundant_pairs.append((col1, col2, corr_value))
        
        for col1, col2, corr_value in redundant_pairs:
            self.findings.append({
                'type': 'redundant_features',
                'severity': 'warning',
                'feature': f'{col1}, {col2}',
                'message': f'Features "{col1}" and "{col2}" are highly correlated ({corr_value:.3f}) - redundant',
                'evidence': {
                    'feature1': col1,
                    'feature2': col2,
                    'correlation': float(corr_value)
                }
            })
    
    def _check_low_information(self):
        """Check for features with low mutual information with target."""
        
        if self.target not in self.df.columns:
            return
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target]
        
        if len(numeric_cols) == 0:
            return
        
        try:
            X = self.df[numeric_cols].copy()
            y = self.df[self.target].copy()
            
            # Fill missing values
            for col in X.columns:
                if X[col].isna().any():
                    X[col].fillna(X[col].mean(), inplace=True)
            
            # Remove samples with missing target
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                return
            
            # Calculate mutual information
            mi_scores = mutual_info_classif(X, y, random_state=42)
            
            # Normalize MI scores
            max_mi = mi_scores.max() if mi_scores.max() > 0 else 1.0
            normalized_mi = mi_scores / max_mi
            
            # Identify features with very low MI
            for col, mi, norm_mi in zip(numeric_cols, mi_scores, normalized_mi):
                if norm_mi < 0.05 and mi < 0.01:  # Both absolute and relative thresholds
                    self.findings.append({
                        'type': 'low_information_feature',
                        'severity': 'info',
                        'feature': col,
                        'message': f'Feature "{col}" has very low mutual information with target',
                        'evidence': {
                            'mutual_information': float(mi),
                            'normalized_mi': float(norm_mi)
                        }
                    })
        except Exception as e:
            pass
    
    def _check_high_cardinality(self):
        """Check for categorical features with too many unique values."""
        
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != self.target]
        
        for col in categorical_cols:
            try:
                unique_count = self.df[col].nunique()
                total_count = len(self.df)
                cardinality_ratio = unique_count / total_count
                
                # High cardinality if >50% unique values
                if cardinality_ratio > 0.5 and unique_count > 20:
                    self.findings.append({
                        'type': 'high_cardinality',
                        'severity': 'warning',
                        'feature': col,
                        'message': f'Feature "{col}" has {unique_count} unique values ({cardinality_ratio*100:.1f}% of dataset) - may need encoding',
                        'evidence': {
                            'unique_count': int(unique_count),
                            'total_count': int(total_count),
                            'cardinality_ratio': float(cardinality_ratio)
                        }
                    })
                
                # Extremely high cardinality (near-unique identifier)
                if cardinality_ratio > 0.95:
                    self.findings.append({
                        'type': 'identifier_feature',
                        'severity': 'critical',
                        'feature': col,
                        'message': f'Feature "{col}" appears to be an identifier ({cardinality_ratio*100:.1f}% unique) - should be removed',
                        'evidence': {
                            'unique_count': int(unique_count),
                            'cardinality_ratio': float(cardinality_ratio)
                        }
                    })
            except Exception as e:
                continue
    
    def _check_excessive_missing(self):
        """Check for features with excessive missing values."""
        
        for col in self.df.columns:
            if col == self.target:
                continue
            
            try:
                missing_count = self.df[col].isna().sum()
                missing_rate = missing_count / len(self.df)
                
                if missing_rate > 0.7:  # More than 70% missing
                    self.findings.append({
                        'type': 'excessive_missing',
                        'severity': 'critical',
                        'feature': col,
                        'message': f'Feature "{col}" has {missing_rate*100:.1f}% missing values - too sparse to be useful',
                        'evidence': {
                            'missing_count': int(missing_count),
                            'missing_rate': float(missing_rate),
                            'total_count': len(self.df)
                        }
                    })
                elif missing_rate > 0.4:  # 40-70% missing
                    self.findings.append({
                        'type': 'high_missing',
                        'severity': 'warning',
                        'feature': col,
                        'message': f'Feature "{col}" has {missing_rate*100:.1f}% missing values - consider imputation or removal',
                        'evidence': {
                            'missing_count': int(missing_count),
                            'missing_rate': float(missing_rate)
                        }
                    })
            except Exception as e:
                continue
    
    def get_summary(self):
        """Generate summary of feature utility findings."""
        
        if not self.findings:
            return {
                'status': 'pass',
                'message': 'All features appear to have reasonable utility',
                'critical_count': 0,
                'warning_count': 0,
                'info_count': 0
            }
        
        critical = [f for f in self.findings if f['severity'] == 'critical']
        warnings = [f for f in self.findings if f['severity'] == 'warning']
        info = [f for f in self.findings if f['severity'] == 'info']
        
        # Recommend features to remove
        features_to_remove = set()
        for finding in critical:
            if 'feature' in finding and ',' not in finding['feature']:
                features_to_remove.add(finding['feature'])
        
        return {
            'status': 'fail' if critical else ('warning' if warnings else 'info'),
            'message': f'Found {len(critical)} critical, {len(warnings)} warnings, {len(info)} info',
            'critical_count': len(critical),
            'warning_count': len(warnings),
            'info_count': len(info),
            'findings': self.findings,
            'recommended_removals': list(features_to_remove)
        }