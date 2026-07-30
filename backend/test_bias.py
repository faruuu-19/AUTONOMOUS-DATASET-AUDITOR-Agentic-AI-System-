import pandas as pd
import numpy as np
from tools.bias_detector import BiasDetector

# Create dataset with MULTIPLE bias issues
np.random.seed(42)

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 10,  # 100 samples
    'income': np.random.normal(60000, 20000, 100),
    'credit_score': np.random.normal(700, 50, 100),
    'target': [0] * 95 + [1] * 5  # SEVERE IMBALANCE: 95% class 0, 5% class 1
})

# Add missing values that correlate with target
df.loc[df['target'] == 1, 'credit_score'] = np.nan  # All class 1 missing credit_score!

# Add extremely skewed feature
df['skewed_feature'] = np.exp(np.random.normal(0, 3, 100))  # Exponential = high skew

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df['target'].value_counts())
print("\nMissing values:")
print(df.isna().sum())

print("\n" + "="*60)
print("RUNNING BIAS DETECTION...")
print("="*60)

detector = BiasDetector(df, 'target')
findings = detector.detect()

if findings:
    print(f"\n  Found {len(findings)} issue(s):\n")
    for i, finding in enumerate(findings, 1):
        print(f"{i}. [{finding['severity'].upper()}] {finding['type']}")
        if 'feature' in finding:
            print(f"   Feature: {finding['feature']}")
        print(f"   {finding['message']}")
        print(f"   Evidence: {finding['evidence']}\n")
else:
    print("\n No bias detected!")

summary = detector.get_summary()
print("="*60)
print("SUMMARY")
print("="*60)
print(f"Status: {summary['status']}")
print(f"Critical: {summary['critical_count']}")
print(f"Warnings: {summary['warning_count']}")