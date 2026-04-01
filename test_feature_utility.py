import pandas as pd
import numpy as np
from tools.feature_utility_detector import FeatureUtilityDetector

# Create dataset with MULTIPLE feature utility issues
np.random.seed(42)

df = pd.DataFrame({
    # GOOD features
    'age': np.random.randint(20, 70, 100),
    'income': np.random.normal(60000, 20000, 100),
    
    # BAD features
    'constant_col': [1] * 100,  # Constant!
    'near_constant': [1] * 96 + [2] * 4,  # 96% same value
    'customer_id': range(1000, 1100),  # Identifier (100% unique)
    'duplicate_age': np.random.randint(20, 70, 100),  # Will correlate with age
    'mostly_missing': [np.nan] * 80 + list(range(20)),  # 80% missing
    'low_variance': [100.0] * 50 + [100.1] * 50,  # Very low variance
    
    'target': np.random.randint(0, 2, 100)
})

# Make duplicate_age exactly match age for perfect correlation
df['duplicate_age'] = df['age']

print("Dataset shape:", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isna().sum())

print("\n" + "="*60)
print("RUNNING FEATURE UTILITY DETECTION...")
print("="*60)

detector = FeatureUtilityDetector(df, 'target')
findings = detector.detect()

if findings:
    print(f"\n Found {len(findings)} issue(s):\n")
    for i, finding in enumerate(findings, 1):
        print(f"{i}. [{finding['severity'].upper()}] {finding['type']}")
        print(f"   Feature: {finding['feature']}")
        print(f"   {finding['message']}\n")
else:
    print("\n All features have good utility!")

summary = detector.get_summary()
print("="*60)
print("SUMMARY")
print("="*60)
print(f"Status: {summary['status']}")
print(f"Critical: {summary['critical_count']}")
print(f"Warnings: {summary['warning_count']}")
print(f"Info: {summary['info_count']}")

if summary.get('recommended_removals'):
    print(f"\n Recommended to remove: {summary['recommended_removals']}")