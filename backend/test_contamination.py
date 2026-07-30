import pandas as pd
from tools.contamination_detector import ContaminationDetector

# Create training set
train_df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50],
    'income': [50000, 60000, 70000, 80000, 90000, 100000],
    'score': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    'target': [0, 0, 1, 1, 0, 1]
})

# Create test set WITH contamination
test_df = pd.DataFrame({
    'age': [30, 55, 35, 60],  # Row 2 matches train row 2!
    'income': [60000, 110000, 70000, 120000],
    'score': [0.6, 0.92, 0.7, 0.88],
    'target': [0, 1, 1, 0]
})

print("Train set:")
print(train_df)
print("\nTest set:")
print(test_df)

print("\n" + "="*60)
print("RUNNING CONTAMINATION DETECTION...")
print("="*60)

detector = ContaminationDetector(train_df, test_df)
findings = detector.detect()

if findings:
    print(f"\n⚠️  Found {len(findings)} issue(s):\n")
    for i, finding in enumerate(findings, 1):
        print(f"{i}. [{finding['severity'].upper()}] {finding['type']}")
        print(f"   {finding['message']}")
        print(f"   Evidence: {finding['evidence']}\n")
else:
    print("\n✅ No contamination detected!")

summary = detector.get_summary()
print("="*60)
print(f"Status: {summary['status']}")
print(f"Critical: {summary['critical_count']}, Warnings: {summary['warning_count']}")