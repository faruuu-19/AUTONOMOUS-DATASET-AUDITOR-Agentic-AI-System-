import pandas as pd
import numpy as np
from tools.spurious_correlation_detector import SpuriousCorrelationDetector

# Create dataset with SPURIOUS correlation
np.random.seed(42)

n_samples = 200

# Create a "magic feature" that perfectly predicts target
# This simulates shortcut learning (e.g., image background instead of object)
magic_feature = np.random.rand(n_samples)
target = (magic_feature > 0.5).astype(int)

# Add some legitimate features that have weak correlation
df = pd.DataFrame({
    'age': np.random.randint(20, 70, n_samples),
    'income': np.random.normal(60000, 20000, n_samples),
    'magic_shortcut': magic_feature,  # This is the spurious feature!
    'credit_score': np.random.normal(700, 50, n_samples),
    'target': target
})

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df['target'].value_counts())

print("\n  NOTE: 'magic_shortcut' was artificially created to perfectly predict target")
print("This simulates real-world spurious correlations like:")
print("- Image classifier using background instead of object")
print("- Model using data collection artifact instead of true signal\n")

print("="*60)
print("RUNNING SPURIOUS CORRELATION DETECTION...")
print("="*60)
print("(This may take 10-20 seconds as it trains models...)\n")

detector = SpuriousCorrelationDetector(df, 'target')
findings = detector.detect()

if findings:
    print(f"  Found {len(findings)} issue(s):\n")
    for i, finding in enumerate(findings, 1):
        print(f"{i}. [{finding['severity'].upper()}] {finding['type']}")
        if 'feature' in finding:
            print(f"   Feature: {finding['feature']}")
        print(f"   {finding['message']}")
        print(f"   Evidence: {finding['evidence']}\n")
else:
    print("\n No spurious correlations detected!")

summary = detector.get_summary()
print("="*60)
print("SUMMARY")
print("="*60)
print(f"Status: {summary['status']}")
print(f"Critical: {summary['critical_count']}, Warnings: {summary['warning_count']}")