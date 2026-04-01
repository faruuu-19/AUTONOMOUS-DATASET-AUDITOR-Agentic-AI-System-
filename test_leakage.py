import pandas as pd
from tools.leakage_detector import LeakageDetector

# Create a dataset with INTENTIONAL leakage
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'income': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
    'credit_score': [650, 700, 750, 800, 720, 680, 760, 790],
    
    # LEAKED FEATURES (intentionally bad!)
    'will_default': [0, 0, 1, 1, 0, 1, 0, 1],  # This IS the target!
    'final_outcome': [0, 0, 1, 1, 0, 1, 0, 1],  # Duplicate of target
    'prediction_score': [0.1, 0.2, 0.9, 0.95, 0.15, 0.88, 0.12, 0.92],  # Perfect correlation
    
    # TARGET
    'defaulted': [0, 0, 1, 1, 0, 1, 0, 1]
})

print("Dataset shape:", df.shape)
print("\nDataset preview:")
print(df.head())

# Run leakage detection
print("\n" + "="*60)
print("RUNNING LEAKAGE DETECTION...")
print("="*60)

detector = LeakageDetector(df, 'defaulted')
findings = detector.detect()

# Print findings
print(f"\nFound {len(findings)} potential leakage issues:\n")

for i, finding in enumerate(findings, 1):
    print(f"{i}. [{finding['severity'].upper()}] {finding['type']}")
    print(f"   Feature: {finding['feature']}")
    print(f"   Message: {finding['message']}")
    print(f"   Evidence: {finding['evidence']}")
    print()

# Print summary
summary = detector.get_summary()
print("="*60)
print("SUMMARY")
print("="*60)
print(f"Status: {summary['status']}")
print(f"Critical Issues: {summary['critical_count']}")
print(f"Warnings: {summary['warning_count']}")