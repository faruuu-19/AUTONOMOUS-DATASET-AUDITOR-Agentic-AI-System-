"""
generate_test_datasets.py - Generate Test Datasets for UI Testing

Run this script to create test datasets that demonstrate:
1. Autonomous decision-making
2. Goal-oriented reasoning
3. Dynamic strategy changes

Usage:
    python generate_test_datasets.py
    
Then upload these CSV files to your Streamlit app!
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_test_datasets():
    """Generate all test datasets"""
    
    # Create test_data directory
    Path("test_data").mkdir(exist_ok=True)
    
    print("🔧 Generating test datasets for autonomous auditor...")
    print("="*70)
    
    # Dataset 1: Data Leakage (should detect quickly and stop)
    print("\n1️⃣  Creating: data_leakage_test.csv")
    print("   Expected behavior:")
    print("   - Goal: FIND_CRITICAL_FAST")
    print("   - Should prioritize leakage_detector")
    print("   - Should find leakage and STOP early")
    print("   - Should skip low-priority tools")
    
    np.random.seed(42)
    n = 1000
    
    df1 = pd.DataFrame({
        'customer_id': range(n),  # ID column (risk flag)
        'age': np.random.randint(18, 80, n),
        'income': np.random.randint(20000, 150000, n),
        'credit_score': np.random.randint(300, 850, n),
    })
    
    # Create target
    df1['default'] = (df1['credit_score'] < 600).astype(int)
    
    # CRITICAL ISSUE: Perfect leakage!
    df1['will_default'] = df1['default']  # This shouldn't exist at prediction time!
    
    df1.to_csv('test_data/data_leakage_test.csv', index=False)
    print("   ✓ Saved: test_data/data_leakage_test.csv")
    print(f"   Size: {df1.shape[0]} rows, {df1.shape[1]} columns")
    
    
    # Dataset 2: Severe Class Imbalance
    print("\n2️⃣  Creating: class_imbalance_test.csv")
    print("   Expected behavior:")
    print("   - Goal: FIND_CRITICAL_FAST")
    print("   - Should prioritize bias_detector")
    print("   - Should find imbalance, then PIVOT strategy")
    print("   - Should boost contamination_detector (imbalanced data prone to leaks)")
    
    np.random.seed(43)
    n = 2000
    
    df2 = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n),
        'feature_4': np.random.uniform(0, 100, n),
        'feature_5': np.random.exponential(2, n),
    })
    
    # CRITICAL ISSUE: Severe imbalance (97% vs 3%)
    df2['target'] = 0
    minority_indices = np.random.choice(n, size=int(n * 0.03), replace=False)
    df2.loc[minority_indices, 'target'] = 1
    
    df2.to_csv('test_data/class_imbalance_test.csv', index=False)
    print("   ✓ Saved: test_data/class_imbalance_test.csv")
    print(f"   Size: {df2.shape[0]} rows, {df2.shape[1]} columns")
    print(f"   Class distribution: {(df2['target']==0).sum()} vs {(df2['target']==1).sum()}")
    
    
    # Dataset 3: Clean & Simple (should validate quickly)
    print("\n3️⃣  Creating: clean_simple_test.csv")
    print("   Expected behavior:")
    print("   - Goal: QUICK_VALIDATION (low complexity)")
    print("   - Should run 2-3 tools max")
    print("   - Should find NO issues and STOP early")
    print("   - Time: <60 seconds")
    
    np.random.seed(44)
    n = 500
    
    df3 = pd.DataFrame({
        'height': np.random.normal(170, 10, n),
        'weight': np.random.normal(70, 15, n),
        'age': np.random.randint(18, 65, n),
    })
    
    # Balanced, clean target
    df3['category'] = np.random.choice(['A', 'B'], size=n, p=[0.5, 0.5])
    
    df3.to_csv('test_data/clean_simple_test.csv', index=False)
    print("   ✓ Saved: test_data/clean_simple_test.csv")
    print(f"   Size: {df3.shape[0]} rows, {df3.shape[1]} columns")
    
    
    # Dataset 4: Complex & Large (deep investigation needed)
    print("\n4️⃣  Creating: complex_large_test.csv")
    print("   Expected behavior:")
    print("   - Goal: DEEP_INVESTIGATION (high complexity)")
    print("   - Should run MOST/ALL tools")
    print("   - Should find multiple issues")
    print("   - Time: 3-5 minutes")
    
    np.random.seed(45)
    n = 12000
    
    # Many features
    df4 = pd.DataFrame({
        f'feature_{i}': np.random.randn(n) for i in range(40)
    })
    
    # Add temporal features (risk flag)
    df4['timestamp'] = pd.date_range('2020-01-01', periods=n, freq='H')
    df4['year'] = df4['timestamp'].dt.year
    df4['month'] = df4['timestamp'].dt.month
    
    # Add ID (risk flag)
    df4['transaction_id'] = range(n)
    
    # Imbalanced target
    df4['fraud'] = 0
    fraud_indices = np.random.choice(n, size=int(n * 0.08), replace=False)
    df4['fraud'].iloc[fraud_indices] = 1
    
    # Add subtle leakage
    df4['future_flag'] = df4['fraud'] + np.random.randn(n) * 0.3
    df4['future_flag'] = (df4['future_flag'] > 0.5).astype(int)
    
    df4 = df4.drop('timestamp', axis=1)  # Remove datetime for CSV
    df4.to_csv('test_data/complex_large_test.csv', index=False)
    print("   ✓ Saved: test_data/complex_large_test.csv")
    print(f"   Size: {df4.shape[0]} rows, {df4.shape[1]} columns")
    
    
    # Dataset 5: Multiple Issues (should trigger multiple pivots)
    print("\n5️⃣  Creating: multiple_issues_test.csv")
    print("   Expected behavior:")
    print("   - Goal: FIND_CRITICAL_FAST")
    print("   - Should find leakage → PIVOT (deprioritize spurious)")
    print("   - Should find imbalance → PIVOT (boost contamination)")
    print("   - Should make 2-3 strategic pivots")
    print("   - Should stop early after finding 3+ critical issues")
    
    np.random.seed(46)
    n = 1500
    
    df5 = pd.DataFrame({
        'user_id': range(n),  # ID
        'feature_A': np.random.randn(n),
        'feature_B': np.random.randn(n),
        'feature_C': np.random.uniform(0, 100, n),
    })
    
    # ISSUE 1: Severe imbalance
    df5['churned'] = 0
    churn_indices = np.random.choice(n, size=int(n * 0.05), replace=False)
    df5.loc[churn_indices, 'churned'] = 1
    
    # ISSUE 2: Data leakage
    df5['churn_probability'] = df5['churned']  # Leakage!
    
    # ISSUE 3: Add duplicates (contamination)
    duplicates = df5.iloc[:100].copy()
    df5 = pd.concat([df5, duplicates], ignore_index=True)
    
    df5.to_csv('test_data/multiple_issues_test.csv', index=False)
    print("   ✓ Saved: test_data/multiple_issues_test.csv")
    print(f"   Size: {df5.shape[0]} rows, {df5.shape[1]} columns")
    
    
    # Dataset 6: Learning Test (run this multiple times)
    print("\n6️⃣  Creating: learning_test_v1.csv and learning_test_v2.csv")
    print("   Expected behavior:")
    print("   - First audit: Learning from scratch")
    print("   - Second audit: Should remember v1 and adapt")
    print("   - Third audit: Should be even smarter!")
    
    np.random.seed(47)
    n = 800
    
    # V1: Imbalanced
    df6a = pd.DataFrame({
        'feat_1': np.random.randn(n),
        'feat_2': np.random.randn(n),
        'feat_3': np.random.randn(n),
    })
    df6a['label'] = 0
    df6a.loc[np.random.choice(n, size=int(n*0.1), replace=False), 'label'] = 1
    df6a.to_csv('test_data/learning_test_v1.csv', index=False)
    
    # V2: Similar profile (also imbalanced, similar size)
    df6b = pd.DataFrame({
        'var_1': np.random.randn(n),
        'var_2': np.random.randn(n),
        'var_3': np.random.randn(n),
    })
    df6b['outcome'] = 0
    df6b.loc[np.random.choice(n, size=int(n*0.12), replace=False), 'outcome'] = 1
    df6b.to_csv('test_data/learning_test_v2.csv', index=False)
    
    print("   ✓ Saved: learning_test_v1.csv and learning_test_v2.csv")
    
    
    print("\n" + "="*70)
    print("✅ ALL TEST DATASETS CREATED!")
    print("="*70)
    print("\nNext steps:")
    print("1. Start your Streamlit app: streamlit run app.py")
    print("2. Upload test datasets from test_data/ folder")
    print("3. Watch autonomous decisions and strategy changes!")
    print("\nWhat to observe:")
    print("  📊 Different GOALS for different datasets")
    print("  🎯 Tool selection based on dataset profile")
    print("  🔄 STRATEGIC PIVOTS when issues are found")
    print("  🛑 EARLY STOPPING when goals achieved")
    print("  📚 LEARNING improvement across similar datasets")


if __name__ == '__main__':
    create_test_datasets()