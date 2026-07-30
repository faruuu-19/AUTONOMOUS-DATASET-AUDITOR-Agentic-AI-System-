from agent.memory1 import AuditMemory
from agent.planner import AuditPlanner

print("="*60)
print("TESTING PLANNER MODULE")
print("="*60)

# Create memory and planner
memory = AuditMemory()
memory.initialize_audit(dataset_shape=(1000, 20), target_column='target')

planner = AuditPlanner(memory)

print("\n1. Initial State")
print(f"   Planner: {planner}")

print("\n2. Execution Plan (all tools):")
plan = planner.get_execution_plan()
for i, tool in enumerate(plan, 1):
    print(f"   {i}. {tool}")
    print(f"      → {planner.get_reason_for_tool(tool)[:80]}...")

print("\n3. Progress:")
progress = planner.get_progress()
for key, value in progress.items():
    print(f"   {key}: {value}")

print("\n" + "="*60)
print("SIMULATING AUDIT EXECUTION")
print("="*60)

# Simulate running tools one by one
step = 1
while planner.should_continue():
    next_tool = planner.get_next_tool()
    
    print(f"\nStep {step}: Executing {next_tool}")
    print(f"   Reason: {planner.get_reason_for_tool(next_tool)[:100]}...")
    
    # Simulate findings
    if next_tool == 'leakage_detector':
        findings = [
            {'type': 'perfect_correlation', 'severity': 'critical', 'message': 'Leakage found'}
        ]
        memory.add_audit_step(next_tool, 'fail', findings, 1.2)
        print(f"   Status: FAIL (found {len(findings)} critical issues)")
        
    elif next_tool == 'bias_detector':
        findings = [
            {'type': 'mild_imbalance', 'severity': 'info', 'message': 'Mild imbalance'}
        ]
        memory.add_audit_step(next_tool, 'info', findings, 0.8)
        print(f"    Status: INFO (found {len(findings)} issues)")
        
    else:
        findings = []
        memory.add_audit_step(next_tool, 'pass', findings, 0.5)
        print(f"    Status: PASS (no issues)")
    
    # Show progress
    progress = planner.get_progress()
    print(f"   Progress: {progress['completed']}/{progress['total_tools']} tools ({progress['progress_percentage']:.0f}%)")
    
    step += 1

print("\n" + "="*60)
print("AUDIT COMPLETE")
print("="*60)

print(f"\n Final planner state: {planner}")

print("\n Adaptive Strategy:")
strategy = planner.get_adaptive_strategy()
print(f"   Status: {strategy['current_status']}")
print(f"   Priority areas: {strategy['priority_areas']}")
print(f"   Recommendations:")
for rec in strategy['recommendations']:
    print(f"   - {rec}")

print("\n Execution order explanation:")
explanations = planner.explain_execution_order()
for exp in explanations:
    status = " DONE" if exp['executed'] else "○ PENDING"
    print(f"   [{status}] Priority {exp['priority']}: {exp['tool']}")

print("\n Planner module test complete!")