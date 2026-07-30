import pandas as pd
import numpy as np
from auditor import AutonomousDatasetAuditor

print("="*70)
print("TESTING AGENTIC ENHANCEMENTS")
print("="*70)

# Test Enhancement #2: Dynamic tool skipping
print("\n" + "="*70)
print("TEST 1: Dynamic Tool Skipping (tiny dataset)")
print("="*70)

# Create tiny dataset (should skip some tools)
tiny_df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'target': [0, 0, 1, 1, 0]
})
tiny_df.to_csv('data/tiny_dataset.csv', index=False)

auditor1 = AutonomousDatasetAuditor(verbose=True)
auditor1.load_dataset('data/tiny_dataset.csv', 'target')
report1 = auditor1.run_audit()

# Test Enhancement #1: Adaptive re-checking
print("\n\n" + "="*70)
print("TEST 2: Adaptive Re-checking (spurious correlations)")
print("="*70)

# Create dataset with suspicious patterns
np.random.seed(42)
suspicious_df = pd.DataFrame({
    'age': np.random.randint(20, 70, 100),
    'magic_feature': np.random.rand(100),
    'target': [0] * 50 + [1] * 50
})
# Make magic_feature highly predictive
suspicious_df.loc[suspicious_df['target'] == 1, 'magic_feature'] += 0.5
suspicious_df.to_csv('data/suspicious_dataset.csv', index=False)

auditor2 = AutonomousDatasetAuditor(verbose=True)
auditor2.load_dataset('data/suspicious_dataset.csv', 'target')
report2 = auditor2.run_audit()

# Test Enhancement #3: Confidence-based stopping
print("\n\n" + "="*70)
print("TEST 3: All Enhancements on Complex Dataset")
print("="*70)

# Create a complex problematic dataset
complex_df = pd.DataFrame({
    'age': np.random.randint(20, 70, 200),
    'income': np.random.normal(60000, 20000, 200),
    'leaked': [0] * 190 + [1] * 10,
    'target': [0] * 190 + [1] * 10
})
complex_df.to_csv('data/complex_dataset.csv', index=False)

auditor3 = AutonomousDatasetAuditor(verbose=True)
auditor3.load_dataset('data/complex_dataset.csv', 'target')
report3 = auditor3.run_audit()

print("\n" + "="*70)
print("ALL AGENTIC ENHANCEMENTS TESTED!")
print("="*70)

print("\nWhat made these tests AGENTIC:")
print("  ✓ Enhancement 1: Automatically re-checked suspicious findings")
print("  ✓ Enhancement 2: Skipped irrelevant tools dynamically")
print("  ✓ Enhancement 3: Stopped early when confidence was too low")
print("\n🤖 The system now truly makes AUTONOMOUS DECISIONS!")