"""
test_autonomous_system.py - Automated Test Suite

This script:
1. Generates proper test datasets
2. Runs autonomous audits on all datasets
3. Validates autonomous decision-making
4. Validates goal-oriented reasoning
5. Generates comparison report

Usage:
    python test_autonomous_system.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from auditor import AutonomousDatasetAuditor


class TestDatasetGenerator:
    """Generate test datasets with known characteristics"""
    
    @staticmethod
    def create_leakage_dataset():
        """Dataset 1: Data Leakage - should prioritize leakage_detector"""
        np.random.seed(42)
        n = 1000
        
        df = pd.DataFrame({
            'customer_id': range(n),  # ID column (risk flag)
            'age': np.random.randint(18, 80, n),
            'income': np.random.randint(20000, 150000, n),
            'credit_score': np.random.randint(300, 850, n),
        })
        
        # Create PROPER categorical target
        df['default'] = (df['credit_score'] < 600).astype(int)
        
        # CRITICAL ISSUE: Perfect leakage!
        df['will_default'] = df['default'].copy()  # This shouldn't exist at prediction time!
        
        return df, 'default', 'data_leakage_test.csv'
    
    @staticmethod
    def create_imbalanced_dataset():
        """Dataset 2: Severe Class Imbalance - should prioritize bias_detector"""
        np.random.seed(43)
        n = 2000
        
        df = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'feature_3': np.random.randn(n),
            'feature_4': np.random.uniform(0, 100, n),
            'feature_5': np.random.exponential(2, n),
        })
        
        # CRITICAL ISSUE: Severe imbalance (97% vs 3%)
        df['target'] = 0
        minority_indices = np.random.choice(n, size=int(n * 0.03), replace=False)
        df.loc[minority_indices, 'target'] = 1
        
        return df, 'target', 'class_imbalance_test.csv'
    
    @staticmethod
    def create_clean_dataset():
        """Dataset 3: Clean & Simple - should set QUICK_VALIDATION goal"""
        np.random.seed(44)
        n = 500
        
        df = pd.DataFrame({
            'height': np.random.normal(170, 10, n),
            'weight': np.random.normal(70, 15, n),
            'age': np.random.randint(18, 65, n),
        })
        
        # Balanced, clean target
        df['category'] = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
        
        return df, 'category', 'clean_simple_test.csv'
    
    @staticmethod
    def create_complex_dataset():
        """Dataset 4: Complex & Large - should set DEEP_INVESTIGATION goal"""
        np.random.seed(45)
        n = 12000
        
        # Many features
        data = {f'feature_{i}': np.random.randn(n) for i in range(40)}
        df = pd.DataFrame(data)
        
        # Add temporal features (risk flag)
        df['year'] = np.random.choice([2020, 2021, 2022, 2023], n)
        df['month'] = np.random.randint(1, 13, n)
        
        # Add ID (risk flag)
        df['transaction_id'] = range(n)
        
        # Imbalanced target
        df['fraud'] = 0
        fraud_indices = np.random.choice(n, size=int(n * 0.08), replace=False)
        df.loc[fraud_indices, 'fraud'] = 1
        
        # Add subtle leakage
        df['future_flag'] = df['fraud'] + np.random.randn(n) * 0.3
        df['future_flag'] = (df['future_flag'] > 0.5).astype(int)
        
        return df, 'fraud', 'complex_large_test.csv'
    
    @staticmethod
    def create_multiple_issues_dataset():
        """Dataset 5: Multiple Issues - should trigger multiple strategic pivots"""
        np.random.seed(46)
        n = 1500
        
        df = pd.DataFrame({
            'user_id': range(n),  # ID
            'feature_A': np.random.randn(n),
            'feature_B': np.random.randn(n),
            'feature_C': np.random.uniform(0, 100, n),
        })
        
        # ISSUE 1: Severe imbalance
        df['churned'] = 0
        churn_indices = np.random.choice(n, size=int(n * 0.05), replace=False)
        df.loc[churn_indices, 'churned'] = 1
        
        # ISSUE 2: Data leakage
        df['churn_probability'] = df['churned'].copy()  # Leakage!
        
        # ISSUE 3: Add duplicates (contamination)
        duplicates = df.iloc[:100].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        
        return df, 'churned', 'multiple_issues_test.csv'


class AutonomousTestValidator:
    """Validate autonomous behavior of the auditor"""
    
    def __init__(self):
        self.results = []
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("="*80)
        print("AUTONOMOUS AUDITOR - AUTOMATED TEST SUITE")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Generate and test all datasets
        datasets = [
            TestDatasetGenerator.create_leakage_dataset(),
            TestDatasetGenerator.create_imbalanced_dataset(),
            TestDatasetGenerator.create_clean_dataset(),
            TestDatasetGenerator.create_complex_dataset(),
            TestDatasetGenerator.create_multiple_issues_dataset(),
        ]
        
        for df, target, filename in datasets:
            self.test_dataset(df, target, filename)
        
        # Generate summary report
        self.generate_report()
        
    def test_dataset(self, df, target_column, filename):
        """Test a single dataset"""
        filepath = self.test_dir / filename
        df.to_csv(filepath, index=False)
        
        print("\n" + "─"*80)
        print(f"TESTING: {filename}")
        print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Target: {target_column}")
        print("─"*80)
        
        # Run audit
        start_time = time.time()
        auditor = AutonomousDatasetAuditor(verbose=True)
        
        try:
            auditor.load_dataset(str(filepath), target_column)
            report = auditor.run_audit()
            execution_time = time.time() - start_time
            
            # Extract key metrics for validation
            result = self.extract_metrics(filename, report, execution_time, df, target_column)
            self.results.append(result)
            
            # Validate autonomous behavior
            self.validate_autonomy(result)
            
            print(f"\n✅ Test completed in {execution_time:.2f}s")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            
    def extract_metrics(self, filename, report, execution_time, df, target_column):
        """Extract key metrics from audit report"""
        
        # Get autonomous strategy info
        auto_strategy = report.get('autonomous_strategy', {})
        goal_info = report.get('goal_oriented', {})
        
        # Get dataset profile
        profile = auto_strategy.get('dataset_profile', {})
        
        return {
            'filename': filename,
            'dataset_size': len(df),
            'num_features': len(df.columns),
            'target_column': target_column,
            'execution_time': execution_time,
            
            # Autonomous Strategy Metrics
            'tools_available': 5,
            'tools_selected': len(auto_strategy.get('tools_selected', [])),
            'tools_executed': len(auto_strategy.get('tools_executed', [])),
            'tools_skipped': auto_strategy.get('tools_skipped', []),
            'tool_order': auto_strategy.get('tools_executed', []),
            
            # Dataset Profile
            'complexity_score': profile.get('complexity_score', 0),
            'class_balance': profile.get('class_balance_ratio', 1.0),
            'has_temporal': profile.get('has_temporal_features', False),
            'has_id': profile.get('has_id_features', False),
            
            # Goal-Oriented Metrics
            'primary_goal': goal_info.get('primary_goal', 'unknown'),
            'goal_achieved': goal_info.get('goal_achieved', False),
            'goal_progress': goal_info.get('goal_progress', 0),
            'time_efficiency': goal_info.get('time_efficiency', 0),
            'strategy_changes': goal_info.get('strategy_changes_count', 0),
            
            # Audit Results
            'verdict': report.get('verdict', 'UNKNOWN'),
            'readiness_score': report.get('readiness_score', 0),
            'critical_count': report['summary']['critical_count'],
            'warning_count': report['summary']['warning_count'],
            
            # Learning
            'learning_stats': auto_strategy.get('learning_stats', {}),
        }
    
    def validate_autonomy(self, result):
        """Validate that autonomous decisions make sense"""
        print(f"\n🔍 VALIDATION CHECKS:")
        
        checks_passed = 0
        total_checks = 0
        
        # Check 1: Tool selection is dataset-specific
        total_checks += 1
        if result['tools_selected'] != result['tools_available']:
            print(f"  ✓ Autonomous tool selection: {result['tools_selected']}/{result['tools_available']} tools chosen")
            checks_passed += 1
        else:
            print(f"  ⚠ Running all tools - not truly autonomous")
        
        # Check 2: Leakage dataset prioritizes leakage detector
        if 'leakage' in result['filename']:
            total_checks += 1
            if result['tool_order'] and result['tool_order'][0] == 'leakage_detector':
                print(f"  ✓ Leakage dataset: Correctly prioritized leakage_detector first")
                checks_passed += 1
            else:
                print(f"  ✗ Leakage dataset: Should prioritize leakage_detector, got {result['tool_order'][:2]}")
        
        # Check 3: Imbalanced dataset prioritizes bias detector
        if 'imbalance' in result['filename']:
            total_checks += 1
            if 'bias_detector' in result['tool_order'][:2]:
                print(f"  ✓ Imbalanced dataset: Correctly prioritized bias_detector")
                checks_passed += 1
            else:
                print(f"  ✗ Imbalanced dataset: Should prioritize bias_detector early")
        
        # Check 4: Clean dataset uses quick validation
        if 'clean' in result['filename']:
            total_checks += 1
            if result['primary_goal'] == 'quick_validation':
                print(f"  ✓ Clean dataset: Correctly set QUICK_VALIDATION goal")
                checks_passed += 1
            else:
                print(f"  ⚠ Clean dataset: Expected QUICK_VALIDATION, got {result['primary_goal']}")
        
        # Check 5: Complex dataset uses deep investigation
        if 'complex' in result['filename']:
            total_checks += 1
            if result['primary_goal'] == 'deep_investigation':
                print(f"  ✓ Complex dataset: Correctly set DEEP_INVESTIGATION goal")
                checks_passed += 1
            else:
                print(f"  ⚠ Complex dataset: Expected DEEP_INVESTIGATION, got {result['primary_goal']}")
        
        # Check 6: Multiple issues trigger strategy changes
        if 'multiple' in result['filename']:
            total_checks += 1
            if result['strategy_changes'] >= 1:
                print(f"  ✓ Multiple issues: Triggered {result['strategy_changes']} strategic pivot(s)")
                checks_passed += 1
            else:
                print(f"  ⚠ Multiple issues: Should trigger strategy changes")
        
        # Check 7: Tool skipping happens
        total_checks += 1
        if result['tools_skipped']:
            print(f"  ✓ Autonomous skipping: Skipped {len(result['tools_skipped'])} tool(s)")
            checks_passed += 1
        else:
            print(f"  ⚠ No tools skipped - system may not be autonomous enough")
        
        # Check 8: Execution time varies by goal
        total_checks += 1
        if result['execution_time'] < 10:  # Quick validation should be fast
            if 'clean' in result['filename']:
                print(f"  ✓ Quick validation completed in {result['execution_time']:.1f}s")
                checks_passed += 1
        else:
            if 'complex' in result['filename']:
                print(f"  ✓ Deep investigation took {result['execution_time']:.1f}s (thorough)")
                checks_passed += 1
        
        print(f"\n  📊 Validation: {checks_passed}/{total_checks} checks passed")
        result['validation_score'] = checks_passed / total_checks if total_checks > 0 else 0
        
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("AUTONOMOUS BEHAVIOR TEST REPORT")
        print("="*80)
        
        if not self.results:
            print("No test results available")
            return
        
        # Summary table
        print("\n📊 SUMMARY TABLE:")
        print("-"*80)
        print(f"{'Dataset':<30} {'Goal':<20} {'Tools':<10} {'Time':<10} {'Valid%':<10}")
        print("-"*80)
        
        for r in self.results:
            filename_short = r['filename'].replace('_test.csv', '')[:28]
            goal_short = r['primary_goal'][:18] if r['primary_goal'] != 'unknown' else 'N/A'
            tools_str = f"{r['tools_executed']}/{r['tools_available']}"
            time_str = f"{r['execution_time']:.1f}s"
            valid_str = f"{r.get('validation_score', 0)*100:.0f}%"
            
            print(f"{filename_short:<30} {goal_short:<20} {tools_str:<10} {time_str:<10} {valid_str:<10}")
        
        print("-"*80)
        
        # Key findings
        print("\n🎯 KEY FINDINGS:")
        
        # 1. Goal diversity
        goals = [r['primary_goal'] for r in self.results]
        unique_goals = set(goals)
        print(f"\n1. Goal Diversity:")
        print(f"   ✓ {len(unique_goals)} different goals set across {len(self.results)} datasets")
        for goal in unique_goals:
            count = goals.count(goal)
            print(f"     - {goal}: {count} dataset(s)")
        
        # 2. Tool selection variance
        print(f"\n2. Tool Selection:")
        tool_counts = [r['tools_executed'] for r in self.results]
        print(f"   Range: {min(tool_counts)} to {max(tool_counts)} tools executed")
        print(f"   Average: {sum(tool_counts)/len(tool_counts):.1f} tools per dataset")
        
        # 3. Strategy changes
        total_pivots = sum(r['strategy_changes'] for r in self.results)
        print(f"\n3. Strategic Pivots:")
        print(f"   Total strategy changes: {total_pivots}")
        for r in self.results:
            if r['strategy_changes'] > 0:
                print(f"     - {r['filename']}: {r['strategy_changes']} pivot(s)")
        
        # 4. Learning progress
        learning_active = any(r['learning_stats'].get('learning_active', False) for r in self.results)
        if learning_active:
            last_stats = self.results[-1]['learning_stats']
            print(f"\n4. Learning System:")
            print(f"   ✓ Learning active")
            print(f"   ✓ Historical audits: {last_stats.get('total_audits', 0)}")
            print(f"   ✓ Dataset profiles learned: {last_stats.get('unique_dataset_profiles', 0)}")
        
        # 5. Overall validation
        avg_validation = sum(r.get('validation_score', 0) for r in self.results) / len(self.results)
        print(f"\n5. Overall Autonomy Score:")
        print(f"   {'✓' if avg_validation > 0.7 else '⚠'} Average validation: {avg_validation*100:.1f}%")
        
        if avg_validation > 0.8:
            print(f"   🎉 EXCELLENT: System demonstrates strong autonomous behavior!")
        elif avg_validation > 0.6:
            print(f"   👍 GOOD: System shows autonomous decision-making")
        else:
            print(f"   ⚠️  NEEDS IMPROVEMENT: Increase autonomous behavior")
        
        # Save detailed report
        report_file = self.test_dir / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'results': self.results,
                'summary': {
                    'total_tests': len(self.results),
                    'avg_validation_score': avg_validation,
                    'unique_goals': list(unique_goals),
                    'total_strategy_pivots': total_pivots,
                }
            }, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        print("\n" + "="*80)


def main():
    """Main test execution"""
    validator = AutonomousTestValidator()
    
    try:
        validator.run_all_tests()
        print("\n✅ All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()