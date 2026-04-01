import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class AuditMemory:
    """
    Memory module for the autonomous dataset auditor.
    Stores audit history, findings, and metadata to enable
    reasoning over cumulative evidence and prevent redundant analysis.
    """
    
    def __init__(self):
        """Initialize empty memory."""
        self.audit_steps = []  # List of completed audit steps
        self.findings = {}  # Findings organized by tool name
        self.metadata = {
            'start_time': None,
            'end_time': None,
            'dataset_shape': None,
            'target_column': None
        }
        self.tool_execution_order = []  # Track order of tool execution
        
    def initialize_audit(self, dataset_shape: tuple, target_column: str):
        """
        Initialize a new audit session.
        
        Args:
            dataset_shape: (rows, columns) of the dataset
            target_column: name of the target column
        """
        self.metadata['start_time'] = datetime.now().isoformat()
        self.metadata['dataset_shape'] = dataset_shape
        self.metadata['target_column'] = target_column
        
    def add_audit_step(self, tool_name: str, status: str, findings: List[Dict], 
                       execution_time: float = 0.0):
        """
        Record completion of an audit step.
        
        Args:
            tool_name: Name of the tool that was executed
            status: 'pass', 'warning', or 'fail'
            findings: List of finding dictionaries from the tool
            execution_time: Time taken to execute (seconds)
        """
        step = {
            'tool_name': tool_name,
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'findings_count': len(findings),
            'execution_time': execution_time
        }
        
        self.audit_steps.append(step)
        self.tool_execution_order.append(tool_name)
        
        # Store findings by tool
        self.findings[tool_name] = findings
        
    def has_executed(self, tool_name: str) -> bool:
        """
        Check if a tool has already been executed.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if tool has been executed, False otherwise
        """
        return tool_name in self.tool_execution_order
    
    def get_findings_by_tool(self, tool_name: str) -> List[Dict]:
        """
        Retrieve findings from a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            List of findings, or empty list if tool hasn't been executed
        """
        return self.findings.get(tool_name, [])
    
    def get_all_findings(self) -> List[Dict]:
        """
        Get all findings from all tools.
        
        Returns:
            Flat list of all findings with tool name added
        """
        all_findings = []
        for tool_name, findings in self.findings.items():
            for finding in findings:
                finding_copy = finding.copy()
                finding_copy['tool'] = tool_name
                all_findings.append(finding_copy)
        return all_findings
    
    def get_critical_findings(self) -> List[Dict]:
        """
        Get only critical severity findings.
        
        Returns:
            List of critical findings
        """
        all_findings = self.get_all_findings()
        return [f for f in all_findings if f.get('severity') == 'critical']
    
    def get_findings_by_severity(self, severity: str) -> List[Dict]:
        """
        Get findings filtered by severity level.
        
        Args:
            severity: 'critical', 'warning', or 'info'
            
        Returns:
            List of findings with specified severity
        """
        all_findings = self.get_all_findings()
        return [f for f in all_findings if f.get('severity') == severity]
    
    def get_last_tool_status(self) -> Optional[str]:
        """
        Get the status of the most recently executed tool.
        
        Returns:
            Status string ('pass', 'warning', 'fail') or None if no tools executed
        """
        if not self.audit_steps:
            return None
        return self.audit_steps[-1]['status']
    
    def has_critical_blockers(self) -> bool:
        """
        Check if any critical findings have been found.
        
        Returns:
            True if critical findings exist, False otherwise
        """
        return len(self.get_critical_findings()) > 0
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the audit so far.
        
        Returns:
            Dictionary with counts and statistics
        """
        all_findings = self.get_all_findings()
        
        critical_count = len([f for f in all_findings if f.get('severity') == 'critical'])
        warning_count = len([f for f in all_findings if f.get('severity') == 'warning'])
        info_count = len([f for f in all_findings if f.get('severity') == 'info'])
        
        return {
            'tools_executed': len(self.audit_steps),
            'total_findings': len(all_findings),
            'critical_count': critical_count,
            'warning_count': warning_count,
            'info_count': info_count,
            'has_critical_blockers': critical_count > 0
        }
    
    def finalize_audit(self):
        """Mark the audit as complete."""
        self.metadata['end_time'] = datetime.now().isoformat()
        
    def get_execution_order(self) -> List[str]:
        """
        Get the order in which tools were executed.
        
        Returns:
            List of tool names in execution order
        """
        return self.tool_execution_order.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export memory to dictionary format.
        
        Returns:
            Dictionary representation of memory
        """
        return {
            'metadata': self.metadata,
            'audit_steps': self.audit_steps,
            'findings': self.findings,
            'tool_execution_order': self.tool_execution_order,
            'summary': self.get_summary_stats()
        }
    
    def to_json(self, filepath: str):
        """
        Save memory to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __repr__(self) -> str:
        """String representation of memory."""
        stats = self.get_summary_stats()
        return (f"AuditMemory(tools_executed={stats['tools_executed']}, "
                f"findings={stats['total_findings']}, "
                f"critical={stats['critical_count']}, "
                f"warnings={stats['warning_count']})")
    
    def get_tool_summary(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get summary for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary with tool execution info, or None if not executed
        """
        for step in self.audit_steps:
            if step['tool_name'] == tool_name:
                findings = self.get_findings_by_tool(tool_name)
                critical = len([f for f in findings if f.get('severity') == 'critical'])
                warnings = len([f for f in findings if f.get('severity') == 'warning'])
                
                return {
                    'status': step['status'],
                    'findings_count': step['findings_count'],
                    'critical_count': critical,
                    'warning_count': warnings,
                    'execution_time': step['execution_time']
                }
        return None