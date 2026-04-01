import pandas as pd
import numpy as np
import hashlib
from sklearn.metrics.pairwise import cosine_similarity

class ContaminationDetector:
    """
    Detects train-test contamination in datasets.
    Contamination occurs when identical or near-identical samples appear
    in both training and test sets, leading to overly optimistic performance.
    """
    
    def __init__(self, train_df, test_df=None):
        """
        Initialize contamination detector.
        
        Args:
            train_df: Training dataset as pandas DataFrame
            test_df: Test dataset as pandas DataFrame (optional)
                    If None, will check for duplicates within train_df only
        """
        self.train_df = train_df.copy()
        self.test_df = test_df.copy() if test_df is not None else None
        self.findings = []
        
    def detect(self):
        """
        Run all contamination detection checks.
        
        Returns:
            List of findings with type, severity, message, and evidence
        """
        self.findings = []
        
        # Check 1: Exact duplicate rows
        self._check_exact_duplicates()
        
        # Check 2: Hash-based duplicate detection
        self._check_hash_duplicates()
        
        # Check 3: Near-duplicate detection using similarity
        self._check_near_duplicates()
        
        # Check 4: Check for duplicate indices/IDs
        self._check_duplicate_indices()
        
        return self.findings
    
    def _check_exact_duplicates(self):
        """Check for exact duplicate rows."""
        
        # Check within training set
        train_duplicates = self.train_df.duplicated().sum()
        if train_duplicates > 0:
            duplicate_percentage = (train_duplicates / len(self.train_df)) * 100
            self.findings.append({
                'type': 'exact_duplicates_train',
                'severity': 'warning',
                'message': f'Found {train_duplicates} exact duplicate rows in training set ({duplicate_percentage:.2f}%)',
                'evidence': {
                    'count': int(train_duplicates),
                    'percentage': float(duplicate_percentage)
                }
            })
        
        # Check between train and test
        if self.test_df is not None:
            # Check within test set
            test_duplicates = self.test_df.duplicated().sum()
            if test_duplicates > 0:
                duplicate_percentage = (test_duplicates / len(self.test_df)) * 100
                self.findings.append({
                    'type': 'exact_duplicates_test',
                    'severity': 'warning',
                    'message': f'Found {test_duplicates} exact duplicate rows in test set ({duplicate_percentage:.2f}%)',
                    'evidence': {
                        'count': int(test_duplicates),
                        'percentage': float(duplicate_percentage)
                    }
                })
            
            # Check for rows appearing in both train and test
            # Create a string representation of each row for comparison
            train_strings = self.train_df.apply(lambda x: '|'.join(x.astype(str)), axis=1)
            test_strings = self.test_df.apply(lambda x: '|'.join(x.astype(str)), axis=1)
            
            contaminated = test_strings.isin(train_strings).sum()
            
            if contaminated > 0:
                contamination_percentage = (contaminated / len(self.test_df)) * 100
                self.findings.append({
                    'type': 'train_test_contamination',
                    'severity': 'critical',
                    'message': f'Found {contaminated} exact matches between train and test sets ({contamination_percentage:.2f}% of test set)',
                    'evidence': {
                        'count': int(contaminated),
                        'percentage': float(contamination_percentage)
                    }
                })
    
    def _check_hash_duplicates(self):
        """Use hashing to detect duplicates more efficiently."""
        
        def hash_row(row):
            """Create hash of a row."""
            row_string = '|'.join(str(val) for val in row)
            return hashlib.md5(row_string.encode()).hexdigest()
        
        # Hash training rows
        train_hashes = self.train_df.apply(hash_row, axis=1)
        
        # Check for duplicate hashes in training set
        duplicate_hashes = train_hashes.duplicated().sum()
        if duplicate_hashes > 0:
            # This is redundant with exact_duplicates check, so only report if not already found
            pass
        
        # Check train-test contamination using hashes
        if self.test_df is not None:
            test_hashes = self.test_df.apply(hash_row, axis=1)
            
            # Find test hashes that exist in training hashes
            contaminated_hashes = test_hashes.isin(train_hashes).sum()
            
            if contaminated_hashes > 0:
                # Only report if different from exact duplicate count
                # (this might catch cases where column order differs)
                percentage = (contaminated_hashes / len(self.test_df)) * 100
                if percentage > 0.1:  # Only report if > 0.1%
                    self.findings.append({
                        'type': 'hash_contamination',
                        'severity': 'critical',
                        'message': f'Hash-based detection found {contaminated_hashes} potential duplicates between train and test',
                        'evidence': {
                            'count': int(contaminated_hashes),
                            'percentage': float(percentage)
                        }
                    })
    
    def _check_near_duplicates(self, threshold=0.95):
        """
        Detect near-duplicate rows using cosine similarity.
        Only checks numeric columns for efficiency.
        """
        
        if self.test_df is None:
            return
        
        # Get numeric columns only
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return  # Need at least 2 numeric columns
        
        # Limit to reasonable sample sizes for performance
        max_samples = 1000
        train_sample = self.train_df[numeric_cols].head(max_samples)
        test_sample = self.test_df[numeric_cols].head(max_samples)
        
        # Handle missing values
        train_sample = train_sample.fillna(train_sample.mean())
        test_sample = test_sample.fillna(test_sample.mean())
        
        if len(train_sample) == 0 or len(test_sample) == 0:
            return
        
        try:
            # Calculate cosine similarity between test and train samples
            similarities = cosine_similarity(test_sample, train_sample)
            
            # Find maximum similarity for each test sample
            max_similarities = similarities.max(axis=1)
            
            # Count near-duplicates
            near_duplicates = (max_similarities >= threshold).sum()
            
            if near_duplicates > 0:
                percentage = (near_duplicates / len(test_sample)) * 100
                self.findings.append({
                    'type': 'near_duplicates',
                    'severity': 'warning' if percentage < 5 else 'critical',
                    'message': f'Found {near_duplicates} near-duplicate samples (similarity >= {threshold}) in test set',
                    'evidence': {
                        'count': int(near_duplicates),
                        'percentage': float(percentage),
                        'threshold': float(threshold),
                        'avg_similarity': float(max_similarities.mean())
                    }
                })
        except Exception as e:
            # Skip if similarity calculation fails
            pass
    
    def _check_duplicate_indices(self):
        """Check for duplicate index values between train and test."""
        
        if self.test_df is None:
            return
        
        # Check if there are any overlapping indices
        train_indices = set(self.train_df.index)
        test_indices = set(self.test_df.index)
        
        overlap = train_indices.intersection(test_indices)
        
        if len(overlap) > 0:
            percentage = (len(overlap) / len(test_indices)) * 100
            self.findings.append({
                'type': 'duplicate_indices',
                'severity': 'warning',
                'message': f'Found {len(overlap)} overlapping index values between train and test sets ({percentage:.2f}%)',
                'evidence': {
                    'count': int(len(overlap)),
                    'percentage': float(percentage)
                }
            })
    
    def get_summary(self):
        """Generate summary of contamination findings."""
        
        if not self.findings:
            return {
                'status': 'pass',
                'message': 'No train-test contamination detected',
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
    
    def get_detailed_report(self):
        """Generate a detailed report for user review."""
        
        report = {
            'summary': self.get_summary(),
            'train_size': len(self.train_df),
            'test_size': len(self.test_df) if self.test_df is not None else 0,
            'findings': self.findings
        }
        
        return report