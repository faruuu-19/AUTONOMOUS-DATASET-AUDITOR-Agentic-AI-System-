import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score

class LeakageDetector:
    """
    Detects potential data leakage in datasets.
    Leakage occurs when training data contains information about the target
    that would not be available at prediction time.
    """
    
    def __init__(self, df, target_column):
        """
        Initialize the leakage detector.
        
        Args:
            df: pandas DataFrame containing the dataset
            target_column: name of the target/label column
        """
        self.df = df.copy()
        self.target = target_column
        self.findings = []
        
    def detect(self):
        """
        Run all leakage detection checks.
        
        Returns:
            List of findings, each containing:
            - type: type of issue
            - severity: 'critical', 'warning', 'info'
            - feature: affected feature name
            - message: human-readable description
            - evidence: supporting data/metrics
        """
        self.findings = []
        
        # Check 1: Perfect or near-perfect correlations
        self._check_perfect_correlations()
        
        # Check 2: Suspicious feature names
        self._check_suspicious_names()
        
        # Check 3: Features with impossibly high predictive power
        self._check_predictive_power()
        
        # Check 4: Duplicate or derived features
        self._check_duplicate_features()
        
        return self.findings
    
    def _check_perfect_correlations(self):
        """Detect features with suspiciously high correlation to target."""
        if self.target not in self.df.columns:
            return
        
        target_data = self.df[self.target]
        
        for col in self.df.columns:
            if col == self.target:
                continue
            
            try:
                # For numeric features
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    # Remove NaN for correlation calculation
                    mask = ~(self.df[col].isna() | target_data.isna())
                    if mask.sum() < 2:
                        continue
                    
                    corr = np.corrcoef(self.df[col][mask], target_data[mask])[0, 1]
                    
                    if abs(corr) > 0.99:
                        self.findings.append({
                            'type': 'perfect_correlation',
                            'severity': 'critical',
                            'feature': col,
                            'message': f'Feature "{col}" has near-perfect correlation ({corr:.4f}) with target',
                            'evidence': {'correlation': float(corr)}
                        })
                    elif abs(corr) > 0.95:
                        self.findings.append({
                            'type': 'high_correlation',
                            'severity': 'warning',
                            'feature': col,
                            'message': f'Feature "{col}" has suspiciously high correlation ({corr:.4f}) with target',
                            'evidence': {'correlation': float(corr)}
                        })
                
                # For categorical features - use mutual information
                elif pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_categorical_dtype(self.df[col]):
                    # Check if feature perfectly predicts target
                    mask = ~(self.df[col].isna() | target_data.isna())
                    if mask.sum() < 2:
                        continue
                    
                    # Calculate normalized mutual information
                    col_data = self.df[col][mask].astype(str)
                    target_masked = target_data[mask].astype(str)
                    
                    mi = mutual_info_score(col_data, target_masked)
                    # Normalize by target entropy
                    target_entropy = -sum((target_masked.value_counts() / len(target_masked)) * 
                                        np.log2(target_masked.value_counts() / len(target_masked)))
                    
                    if target_entropy > 0:
                        normalized_mi = mi / target_entropy
                        
                        if normalized_mi > 0.95:
                            self.findings.append({
                                'type': 'perfect_mutual_information',
                                'severity': 'critical',
                                'feature': col,
                                'message': f'Feature "{col}" has near-perfect mutual information ({normalized_mi:.4f}) with target',
                                'evidence': {'normalized_mi': float(normalized_mi)}
                            })
                        elif normalized_mi > 0.85:
                            self.findings.append({
                                'type': 'high_mutual_information',
                                'severity': 'warning',
                                'feature': col,
                                'message': f'Feature "{col}" has suspiciously high mutual information ({normalized_mi:.4f}) with target',
                                'evidence': {'normalized_mi': float(normalized_mi)}
                            })
                            
            except Exception as e:
                # Skip features that cause errors
                continue
    
    def _check_suspicious_names(self):
        """Check for feature names that suggest leakage."""
        suspicious_keywords = [
            'target', 'label', 'outcome', 'result', 'actual', 
            'ground_truth', 'gt', 'prediction', 'pred', 'forecast',
            'future', 'after', 'post', 'final', 'end'
        ]
        
        for col in self.df.columns:
            if col == self.target:
                continue
            
            col_lower = col.lower()
            for keyword in suspicious_keywords:
                if keyword in col_lower:
                    self.findings.append({
                        'type': 'suspicious_name',
                        'severity': 'warning',
                        'feature': col,
                        'message': f'Feature "{col}" has suspicious name containing "{keyword}"',
                        'evidence': {'keyword': keyword}
                    })
                    break
    
    def _check_predictive_power(self):
        """Check if any feature alone can predict target too well."""
        if self.target not in self.df.columns:
            return
        
        target_data = self.df[self.target]
        
        for col in self.df.columns:
            if col == self.target:
                continue
            
            try:
                # For categorical/object features
                if pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_categorical_dtype(self.df[col]):
                    mask = ~(self.df[col].isna() | target_data.isna())
                    if mask.sum() < 10:
                        continue
                    
                    # Check if each unique value maps to only one target class
                    col_data = self.df[col][mask]
                    target_masked = target_data[mask]
                    
                    value_counts = col_data.nunique()
                    if value_counts > 1:
                        # For each unique value, check target variance
                        perfect_mapping = True
                        for val in col_data.unique():
                            target_for_val = target_masked[col_data == val]
                            if target_for_val.nunique() > 1:
                                perfect_mapping = False
                                break
                        
                        if perfect_mapping and value_counts > 2:
                            self.findings.append({
                                'type': 'perfect_predictor',
                                'severity': 'critical',
                                'feature': col,
                                'message': f'Feature "{col}" perfectly predicts target (1-to-1 mapping)',
                                'evidence': {'unique_values': int(value_counts)}
                            })
                            
            except Exception as e:
                continue
    
    def _check_duplicate_features(self):
        """Check for features that are duplicates or transformations of target."""
        if self.target not in self.df.columns:
            return
        
        target_data = self.df[self.target]
        
        for col in self.df.columns:
            if col == self.target:
                continue
            
            try:
                # Check if feature is identical to target
                if self.df[col].equals(target_data):
                    self.findings.append({
                        'type': 'identical_to_target',
                        'severity': 'critical',
                        'feature': col,
                        'message': f'Feature "{col}" is identical to target column',
                        'evidence': {}
                    })
                
                # Check if feature is a simple transformation of target
                if pd.api.types.is_numeric_dtype(self.df[col]) and pd.api.types.is_numeric_dtype(target_data):
                    mask = ~(self.df[col].isna() | target_data.isna())
                    if mask.sum() < 2:
                        continue
                    
                    # Check for linear relationship (y = ax + b)
                    col_vals = self.df[col][mask].values
                    target_vals = target_data[mask].values
                    
                    if len(np.unique(col_vals)) > 1:
                        # Simple linear regression
                        a = np.corrcoef(col_vals, target_vals)[0, 1] * (np.std(target_vals) / np.std(col_vals))
                        b = np.mean(target_vals) - a * np.mean(col_vals)
                        
                        predicted = a * col_vals + b
                        r2 = 1 - (np.sum((target_vals - predicted) ** 2) / np.sum((target_vals - np.mean(target_vals)) ** 2))
                        
                        if r2 > 0.999:
                            self.findings.append({
                                'type': 'linear_transformation',
                                'severity': 'critical',
                                'feature': col,
                                'message': f'Feature "{col}" appears to be a linear transformation of target (R²={r2:.6f})',
                                'evidence': {'r2': float(r2), 'coefficient': float(a), 'intercept': float(b)}
                            })
                            
            except Exception as e:
                continue
    
    def get_summary(self):
        """Generate a summary of all findings."""
        if not self.findings:
            return {
                'status': 'pass',
                'message': 'No data leakage detected',
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