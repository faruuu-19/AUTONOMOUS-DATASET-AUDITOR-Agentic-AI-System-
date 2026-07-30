from agent.memory1 import AuditMemory

# Create memory instance
memory = AuditMemory()

print("="*60)
print("TESTING MEMORY MODULE")
print("="*60)

# Initialize audit
print("\n1. Initializing audit...")
memory.initialize_audit(dataset_shape=(1000, 20), target_column='target')
print(f"   ✓ Metadata: {memory.metadata}")

# Simulate adding findings from tools
print("\n2. Adding findings from Leakage Detector...")
leakage_findings = [
    {
        'type': 'perfect_correlation',
        'severity': 'critical',
        'feature': 'leaked_feature',
        'message': 'Feature has perfect correlation with target'
    },
    {
        'type': 'suspicious_name',
        'severity': 'warning',
        'feature': 'final_outcome',
        'message': 'Suspicious feature name'
    }
]
memory.add_audit_step('leakage_detector', 'fail', leakage_findings, execution_time=1.2)
print(f"   ✓ Added {len(leakage_findings)} findings")

print("\n3. Adding findings from Bias Detector...")
bias_findings = [
    {
        'type': 'severe_imbalance',
        'severity': 'critical',
        'message': 'Severe class imbalance detected'
    }
]
memory.add_audit_step('bias_detector', 'fail', bias_findings, execution_time=0.8)
print(f"   ✓ Added {len(bias_findings)} findings")

print("\n4. Adding findings from Feature Utility...")
utility_findings = [
    {
        'type': 'constant_feature',
        'severity': 'warning',
        'feature': 'constant_col',
        'message': 'Feature has constant value'
    }
]
memory.add_audit_step('feature_utility', 'warning', utility_findings, execution_time=0.5)
print(f"   ✓ Added {len(utility_findings)} findings")

# Test memory queries
print("\n" + "="*60)
print("MEMORY QUERIES")
print("="*60)

print(f"\n✓ Has executed leakage_detector? {memory.has_executed('leakage_detector')}")
print(f"✓ Has executed spurious_detector? {memory.has_executed('spurious_detector')}")

print(f"\n✓ Execution order: {memory.get_execution_order()}")

print(f"\n✓ Has critical blockers? {memory.has_critical_blockers()}")

critical = memory.get_critical_findings()
print(f"\n✓ Critical findings ({len(critical)}):")
for finding in critical:
    print(f"   - [{finding['tool']}] {finding['message']}")

print("\n✓ Summary stats:")
stats = memory.get_summary_stats()
for key, value in stats.items():
    print(f"   {key}: {value}")

print("\n✓ Tool summaries:")
for tool in memory.get_execution_order():
    summary = memory.get_tool_summary(tool)
    print(f"   {tool}: {summary}")

# Finalize and export
print("\n" + "="*60)
print("EXPORTING MEMORY")
print("="*60)

memory.finalize_audit()
print(f"\n Memory representation: {memory}")

# Save to JSON
memory.to_json('reports/audit_memory.json')
print("Saved to: reports/audit_memory.json")

print("\n Memory module test complete!")