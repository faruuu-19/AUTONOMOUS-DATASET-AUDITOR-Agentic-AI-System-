from agent.memory1 import AuditMemory
from agent.critic import AuditCritic

print("="*60)
print("TESTING CRITIC MODULE")
print("="*60)

# Create memory and critic
memory = AuditMemory()
memory.initialize_audit(dataset_shape=(1000, 20), target_column='target')

critic = AuditCritic(memory)

print(f"\n✓ Initial critic: {critic}")

# Simulate findings from different tools
print("\n" + "="*60)
print("SCENARIO 1: Leakage Detector with Perfect Correlation")
print("="*60)

leakage_findings = [
    {
        'type': 'perfect_correlation',
        'severity': 'critical',
        'feature': 'leaked_col',
        'evidence': {'correlation': 0.999}
    },
    {
        'type': 'identical_to_target',
        'severity': 'critical',
        'feature': 'exact_copy'
    }
]
memory.add_audit_step('leakage_detector', 'fail', leakage_findings, 1.2)

critique = critic.evaluate_tool_results('leakage_detector')
print(f"\n✓ Confidence: {critique['confidence']:.2f}")
print(f"✓ Concerns ({len(critique['concerns'])}):")
for concern in critique['concerns']:
    print(f"   - {concern}")
print(f"✓ Recommendations ({len(critique['recommendations'])}):")
for rec in critique['recommendations']:
    print(f"   - {rec}")

# Scenario 2: Bias detector
print("\n" + "="*60)
print("SCENARIO 2: Bias Detector with Small Class")
print("="*60)

bias_findings = [
    {
        'type': 'severe_imbalance',
        'severity': 'critical',
        'message': 'Severe imbalance'
    },
    {
        'type': 'small_class_size',
        'severity': 'critical',
        'evidence': {'min_class_size': 5}
    }
]
memory.add_audit_step('bias_detector', 'fail', bias_findings, 0.8)

critique = critic.evaluate_tool_results('bias_detector')
print(f"\n✓ Confidence: {critique['confidence']:.2f}")
print(f"✓ Concerns ({len(critique['concerns'])}):")
for concern in critique['concerns']:
    print(f"   - {concern}")

# Scenario 3: Spurious correlations (ambiguous)
print("\n" + "="*60)
print("SCENARIO 3: Spurious Detector (Needs Recheck)")
print("="*60)

spurious_findings = [
    {
        'type': 'single_feature_dominance',
        'severity': 'critical',
        'feature': 'mystery_feature',
        'evidence': {'accuracy': 0.92}
    }
]
memory.add_audit_step('spurious_detector', 'fail', spurious_findings, 2.5)

critique = critic.evaluate_tool_results('spurious_detector')
print(f"\n Confidence: {critique['confidence']:.2f}")
print(f" Needs recheck: {critique['needs_recheck']}")
print(f" Recommendations ({len(critique['recommendations'])}):")
for rec in critique['recommendations']:
    print(f"   - {rec}")

# Overall assessment
print("\n" + "="*60)
print("OVERALL ASSESSMENT")
print("="*60)

assessment = critic.get_overall_assessment()
print(f"\n Overall confidence: {assessment['overall_confidence']:.2f}")
print(f" Reliability: {assessment['reliability']}")
print(f" Tools needing recheck: {assessment['tools_needing_recheck']}")
print(f"\n Major concerns ({len(assessment['major_concerns'])}):")
for i, concern in enumerate(assessment['major_concerns'][:3], 1):  # Show first 3
    print(f"   {i}. {concern}")

print(f"\n Actionable recommendations ({len(assessment['actionable_recommendations'])}):")
for i, rec in enumerate(assessment['actionable_recommendations'][:3], 1):  # Show first 3
    print(f"   {i}. {rec}")

print(f"\n Final critic state: {critic}")

print("\n Critic module test complete!")