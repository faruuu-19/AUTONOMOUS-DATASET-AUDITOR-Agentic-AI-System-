#!/usr/bin/env python3
"""
Autonomous Dataset Auditor - Main Entry Point

This script provides a command-line interface for auditing datasets
using the autonomous agentic AI system.

Usage:
    python main.py --dataset path/to/data.csv --target target_column
    python main.py --train train.csv --test test.csv --target target_column
    python main.py --dataset data.csv --target target --output report.json
"""

import argparse
import sys
import os
from pathlib import Path

from auditor import AutonomousDatasetAuditor


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Autonomous Dataset Auditor - An Agentic AI System for Pre-Modeling Data Risk Assessment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Audit a single dataset
  python main.py --dataset data/titanic.csv --target Survived
  
  # Audit with train/test split
  python main.py --train data/train.csv --test data/test.csv --target target
  
  # Save report to custom location
  python main.py --dataset data.csv --target label --output reports/my_audit.json
  
  # Quiet mode (minimal output)
  python main.py --dataset data.csv --target label --quiet
        """
    )
    
    # Dataset input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset CSV file'
    )
    input_group.add_argument(
        '--train',
        type=str,
        help='Path to training dataset CSV (use with --test)'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        help='Path to test dataset CSV (use with --train)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help='Name of the target/label column'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='reports/audit_report.json',
        help='Path to save audit report JSON (default: reports/audit_report.json)'
    )
    
    parser.add_argument(
        '--export-csv',
        type=str,
        help='Export findings to CSV file'
    )
    
    # Verbosity options
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages (only show final report)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed progress information (default)'
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate input arguments."""
    
    # Check if using train/test split
    if args.train and not args.test:
        print("❌ Error: --train requires --test")
        sys.exit(1)
    
    if args.test and not args.train:
        print("❌ Error: --test requires --train")
        sys.exit(1)
    
    # Check if files exist
    if args.dataset:
        if not os.path.exists(args.dataset):
            print(f"❌ Error: Dataset file not found: {args.dataset}")
            sys.exit(1)
    
    if args.train:
        if not os.path.exists(args.train):
            print(f"❌ Error: Training file not found: {args.train}")
            sys.exit(1)
        if not os.path.exists(args.test):
            print(f"❌ Error: Test file not found: {args.test}")
            sys.exit(1)
    
    # Ensure output directory exists
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return True


def print_banner():
    """Print application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║         AUTONOMOUS DATASET AUDITOR                           ║
║                                                                   ║
║     An Agentic AI System for Pre-Modeling Data Risk Assessment   ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_summary(report):
    """Print concise summary of audit results."""
    
    verdict = report['verdict']
    score = report['readiness_score']
    summary = report['summary']
    
    # Choose emoji based on verdict
    if verdict == "READY":
        emoji = "🟢"
    elif verdict == "NEEDS ATTENTION":
        emoji = "🟡"
    else:
        emoji = "🔴"
    
    print("\n" + "="*70)
    print("AUDIT SUMMARY")
    print("="*70)
    print(f"\n{emoji} VERDICT: {verdict}")
    print(f" READINESS SCORE: {score}/100\n")
    
    print(" Findings:")
    print(f"   • Critical Issues: {summary['critical_count']}")
    print(f"   • Warnings: {summary['warning_count']}")
    print(f"   • Informational: {summary['info_count']}")
    print(f"   • Total: {summary['total_findings']}")
    
    # Show top critical issues
    if report['critical_blockers']:
        print(f"\n Top Critical Blockers:")
        for i, blocker in enumerate(report['critical_blockers'][:3], 1):
            tool = blocker.get('tool', 'unknown')
            msg = blocker.get('message', blocker.get('type', 'Issue detected'))
            print(f"   {i}. [{tool}] {msg}")
        
        if len(report['critical_blockers']) > 3:
            remaining = len(report['critical_blockers']) - 3
            print(f"   ... and {remaining} more critical issue(s)")
    
    # Show recommendations
    if report['recommendations']:
        print(f"\n Top Recommendations:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
        
        if len(report['recommendations']) > 3:
            remaining = len(report['recommendations']) - 3
            print(f"   ... and {remaining} more recommendation(s)")
    
    print("\n" + "="*70)


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate inputs
    validate_inputs(args)
    
    # Determine verbosity
    verbose = not args.quiet
    
    if verbose:
        print_banner()
    
    try:
        # Initialize auditor
        auditor = AutonomousDatasetAuditor(verbose=verbose)
        
        # Load dataset
        if args.dataset:
            if verbose:
                print(f"\n Loading dataset: {args.dataset}")
            auditor.load_dataset(args.dataset, args.target)
        else:
            if verbose:
                print(f"\n Loading train/test split:")
                print(f"   Train: {args.train}")
                print(f"   Test: {args.test}")
            auditor.load_train_test_split(args.train, args.test, args.target)
        
        # Run audit
        if verbose:
            print("\n Starting autonomous audit...\n")
        
        report = auditor.run_audit()
        
        # Save report
        if verbose:
            print(f"\n Saving report to: {args.output}")
        
        auditor.save_report(args.output)
        
        # Export findings to CSV if requested
        if args.export_csv:
            if verbose:
                print(f"Exporting findings to: {args.export_csv}")
            auditor.export_findings_csv(args.export_csv)
        
        # Print summary (even in quiet mode)
        print_summary(report)
        
        # Exit with appropriate code
        if report['summary']['critical_count'] > 0:
            sys.exit(1)  # Exit with error if critical issues found
        else:
            sys.exit(0)  # Exit successfully
        
    except FileNotFoundError as e:
        print(f"\nError: File not found - {e}")
        sys.exit(1)
    
    except ValueError as e:
        print(f"\n Error: {e}")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n  Audit interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()