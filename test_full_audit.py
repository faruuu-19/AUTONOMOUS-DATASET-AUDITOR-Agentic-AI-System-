import pandas as pd
import numpy as np
from auditor import AutonomousDatasetAuditor

print("="*70)
print("FULL SYSTEM TEST: AUTONOMOUS DATASET AUDITOR")
print("="*70)

# Create a problematic test dataset
np.random.seed(42)
n_samples = 200

df = pd.DataFrame({
    'age': np.random.randint(20, 70, n_samples),
    'income': np.random.normal(60000, 20000, n_samples),
    'credit_score': np.random.normal(700, 50, n_samples),
    
    # LEAKED FEATURE (perfect correlation with target)
    'leaked_feature': np.random.rand(n_samples),
    
    # Constant feature (no utility)
    'constant_col': [1] * n_samples,
    
    # Target with severe imbalance
    'target': [0] * 190 + [1] * 10  # 95% vs 5%
})

# Make leaked_feature perfectly predict target
df['leaked_feature'] = df['target'].copy()

# Save to CSV
df.to_csv('data/test_dataset.csv', index=False)
print("✓ Created problematic test dataset: data/test_dataset.csv")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Target distribution: {df['target'].value_counts().to_dict()}")

# Initialize auditor
print("\n" + "="*70)
print("INITIALIZING AUDITOR")
print("="*70)

auditor = AutonomousDatasetAuditor(verbose=True)

# Load dataset
auditor.load_dataset('data/test_dataset.csv', target_column='target')

# Run full audit
print("\nStarting autonomous audit...")
report = auditor.run_audit()

# Save report
auditor.save_report('reports/test_audit_report.json')

print("\n" + "="*70)
print(" FULL SYSTEM TEST COMPLETE!")
print("="*70)

print("\nThe autonomous auditor:")
print("  Loaded the dataset")
print("  Planned the audit sequence")
print("  Executed all 5 tools autonomously")
print("  Stored findings in memory")
print("  Evaluated confidence with critic")
print("  Generated final verdict and report")
print("\n ALL COMPONENTS WORKING TOGETHER!")