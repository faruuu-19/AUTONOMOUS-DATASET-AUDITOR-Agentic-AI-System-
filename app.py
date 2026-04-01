"""
Autonomous Dataset Auditor - Streamlit UI
Interactive web interface for dataset auditing with real-time visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import io

from auditor import AutonomousDatasetAuditor


# Page configuration
st.set_page_config(
    page_title="Autonomous Dataset Auditor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css(file_name):
    """Load CSS file for custom styling"""
    try:
        with open(file_name, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styles.")

load_css('styles.css')


def initialize_session_state():
    """Initialize session state variables"""
    if 'audit_complete' not in st.session_state:
        st.session_state.audit_complete = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'report' not in st.session_state:
        st.session_state.report = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None


def display_header():
    """Display application header with tech illustration"""
    st.markdown('''
    <div style="text-align: center;">
        <svg width="100" height="100" viewBox="0 0 100 100" style="margin: 1rem auto; display: block;">
            <!-- Robot head -->
            <rect x="20" y="30" width="60" height="50" rx="10" fill="none" stroke="#5B4E8F" stroke-width="3"/>
            <!-- Antenna -->
            <line x1="50" y1="20" x2="50" y2="30" stroke="#5B4E8F" stroke-width="3"/>
            <circle cx="50" cy="18" r="4" fill="#F4A460"/>
            <!-- Eyes -->
            <circle cx="38" cy="50" r="6" fill="#5B4E8F"/>
            <circle cx="62" cy="50" r="6" fill="#5B4E8F"/>
            <!-- Mouth -->
            <line x1="35" y1="65" x2="65" y2="65" stroke="#5B4E8F" stroke-width="3" stroke-linecap="round"/>
        </svg>
        <h1 style="font-family: 'Caveat', cursive; font-size: 4rem; color: #5B4E8F; margin: 0;">
            Autonomous Dataset Auditor
        </h1>
        <p style="font-family: 'Patrick Hand', cursive; font-size: 1.3rem; color: #7B6BA8;">
            An Agentic AI System for Pre-Modeling Data Risk Assessment
        </p>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown("---")


def upload_dataset():
    """Handle dataset upload"""
    st.sidebar.markdown('''
    <div style="text-align: center; padding: 1rem 0;">
        <svg width="60" height="60" viewBox="0 0 60 60" style="display: block; margin: 0 auto;">
            <rect x="15" y="15" width="30" height="35" rx="3" fill="none" stroke="#5B4E8F" stroke-width="2"/>
            <path d="M 20 10 L 30 15 L 40 10" fill="none" stroke="#5B4E8F" stroke-width="2"/>
            <line x1="22" y1="25" x2="38" y2="25" stroke="#5B4E8F" stroke-width="1.5"/>
            <line x1="22" y1="32" x2="35" y2="32" stroke="#5B4E8F" stroke-width="1.5"/>
            <line x1="22" y1="39" x2="38" y2="39" stroke="#5B4E8F" stroke-width="1.5"/>
        </svg>
        <h2 style="font-family: 'Dancing Script', cursive; color: #5B4E8F; margin-top: 0.5rem;">
            Dataset Upload
        </h2>
    </div>
    ''', unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your dataset for autonomous auditing"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Target column selection
            target_column = st.sidebar.selectbox(
                "Select Target Column",
                options=df.columns.tolist(),
                help="Choose the label/target column for prediction"
            )
            
            # Show data preview
            with st.sidebar.expander("Data Preview"):
                st.dataframe(df.head(5), use_container_width=True)
            
            return df, target_column, uploaded_file.name
            
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            return None, None, None
    
    return None, None, None


def create_progress_visualization(progress_data):
    """Create progress bar visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = progress_data['completed'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Audit Progress", 'font': {'size': 24}},
        delta = {'reference': progress_data['total']},
        gauge = {
            'axis': {'range': [None, progress_data['total']], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, progress_data['total']/3], 'color': '#ffebee'},
                {'range': [progress_data['total']/3, 2*progress_data['total']/3], 'color': '#fff3e0'},
                {'range': [2*progress_data['total']/3, progress_data['total']], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': progress_data['total']
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def display_verdict_card(report):
    """Display verdict with styled card"""
    verdict = report['verdict']
    score = report['readiness_score']
    
    if verdict == "READY":
        color = "#4caf50"
        icon = '''<svg width="40" height="40" viewBox="0 0 40 40" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem;">
            <circle cx="20" cy="20" r="18" fill="none" stroke="#4caf50" stroke-width="3"/>
            <path d="M 12 20 L 18 26 L 28 14" fill="none" stroke="#4caf50" stroke-width="3" stroke-linecap="round"/>
        </svg>'''
        box_class = "success-box"
    elif verdict == "NEEDS ATTENTION":
        color = "#ff9800"
        icon = '''<svg width="40" height="40" viewBox="0 0 40 40" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem;">
            <path d="M 20 5 L 35 35 L 5 35 Z" fill="none" stroke="#ff9800" stroke-width="3"/>
            <line x1="20" y1="15" x2="20" y2="24" stroke="#ff9800" stroke-width="3" stroke-linecap="round"/>
            <circle cx="20" cy="29" r="1.5" fill="#ff9800"/>
        </svg>'''
        box_class = "warning-box"
    else:
        color = "#f44336"
        icon = '''<svg width="40" height="40" viewBox="0 0 40 40" style="display: inline-block; vertical-align: middle; margin-right: 0.5rem;">
            <circle cx="20" cy="20" r="18" fill="none" stroke="#f44336" stroke-width="3"/>
            <line x1="14" y1="14" x2="26" y2="26" stroke="#f44336" stroke-width="3" stroke-linecap="round"/>
            <line x1="26" y1="14" x2="14" y2="26" stroke="#f44336" stroke-width="3" stroke-linecap="round"/>
        </svg>'''
        box_class = "critical-box"
    
    st.markdown(f"""
    <div class="{box_class}">
        <h2 style="color: {color}; margin: 0;">{icon} VERDICT: {verdict}</h2>
        <h1 style="color: {color}; margin: 0.5rem 0;">Readiness Score: {score}/100</h1>
    </div>
    """, unsafe_allow_html=True)


def create_findings_chart(report):
    """Create findings distribution chart"""
    summary = report['summary']
    
    categories = ['Critical', 'Warnings', 'Info']
    values = [
        summary['critical_count'],
        summary['warning_count'],
        summary['info_count']
    ]
    colors = ['#f44336', '#ff9800', '#2196f3']
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Findings Distribution",
        xaxis_title="Severity",
        yaxis_title="Count",
        height=300,
        showlegend=False
    )
    
    return fig


def create_tool_timeline(report):
    """Create timeline of tool execution"""
    timeline_data = report['execution_timeline']
    
    df = pd.DataFrame(timeline_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Map status to colors
    color_map = {'pass': 'green', 'warning': 'orange', 'fail': 'red'}
    df['color'] = df['status'].map(color_map)
    
    fig = px.timeline(
        df,
        x_start='timestamp',
        x_end='timestamp',
        y='tool_name',
        color='status',
        color_discrete_map=color_map,
        title="Audit Tool Execution Timeline",
        labels={'tool_name': 'Tool', 'timestamp': 'Time'}
    )
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=300)
    
    return fig


def display_findings_detail(report):
    """Display detailed findings"""
    all_findings = report['all_findings']
    
    if not all_findings:
        st.success("No issues found! Dataset appears clean.")
        return
    
    # Group by severity
    critical = [f for f in all_findings if f.get('severity') == 'critical']
    warnings = [f for f in all_findings if f.get('severity') == 'warning']
    info = [f for f in all_findings if f.get('severity') == 'info']
    
    # Display critical findings
    if critical:
        st.markdown("### Critical Issues")
        for i, finding in enumerate(critical, 1):
            tool = finding.get('tool', 'unknown')
            msg = finding.get('message', finding.get('type', 'Issue detected'))
            feature = finding.get('feature', '')
            
            with st.expander(f"Critical #{i}: [{tool}] {feature if feature else msg[:50]}", expanded=False):
                st.markdown(f"**Tool:** `{tool}`")
                if feature:
                    st.markdown(f"**Feature:** `{feature}`")
                st.markdown(f"**Message:** {msg}")
                st.markdown(f"**Type:** `{finding.get('type', 'N/A')}`")
                if finding.get('evidence'):
                    st.json(finding['evidence'])
    
    # Display warnings
    if warnings:
        st.markdown("### Warnings")
        for i, finding in enumerate(warnings, 1):
            tool = finding.get('tool', 'unknown')
            msg = finding.get('message', finding.get('type', 'Issue detected'))
            feature = finding.get('feature', '')
            
            with st.expander(f"Warning #{i}: [{tool}] {feature if feature else msg[:50]}", expanded=False):
                st.markdown(f"**Tool:** `{tool}`")
                if feature:
                    st.markdown(f"**Feature:** `{feature}`")
                st.markdown(f"**Message:** {msg}")
                if finding.get('evidence'):
                    st.json(finding['evidence'])
    
    # Display info
    if info:
        st.markdown("### Informational")
        for i, finding in enumerate(info, 1):
            tool = finding.get('tool', 'unknown')
            msg = finding.get('message', finding.get('type', 'Issue detected'))
            
            with st.expander(f"Info #{i}: [{tool}] {msg[:50]}", expanded=False):
                st.markdown(f"**Tool:** `{tool}`")
                st.markdown(f"**Message:** {msg}")


def display_recommendations(report):
    """Display actionable recommendations"""
    recommendations = report.get('recommendations', [])
    
    if recommendations:
        st.markdown("### Recommendations")
        for i, rec in enumerate(recommendations, 1):
            if "PRIORITY" in rec or "CRITICAL" in rec:
                st.markdown(f'<div class="critical-box">{i}. {rec}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'{i}. {rec}')


def run_audit_with_visualization(df, target_column, filename):
    """Run audit with real-time visualization"""
    
    # Create placeholders for dynamic updates
    st.markdown("### Audit in Progress...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    details_container = st.container()
    
    # Show dataset size info
    is_large = len(df) > 10000
    if is_large:
        st.info(f"Large dataset detected ({len(df):,} rows). Running optimized analysis...")
    
    # Save dataset temporarily
    temp_file = 'temp_audit_dataset.csv'
    df.to_csv(temp_file, index=False)
    
    # Create auditor
    auditor = AutonomousDatasetAuditor(verbose=False)
    auditor.load_dataset(temp_file, target_column)
    
    # Manual audit execution with progress updates
    auditor.memory.initialize_audit(df.shape, target_column)
    
    total_tools = 5
    completed = 0
    
    tool_names = ['leakage_detector', 'contamination_detector', 'bias_detector', 
                  'spurious_detector', 'feature_utility']
    
    # Tool descriptions for better UX
    tool_descriptions = {
        'leakage_detector': 'Checking for data leakage (features that shouldn\'t exist at prediction time)',
        'contamination_detector': 'Detecting duplicate samples between train/test sets',
        'bias_detector': 'Analyzing class balance and feature distributions',
        'spurious_detector': 'Identifying spurious correlations and shortcut learning',
        'feature_utility': 'Evaluating feature quality and utility'
    }
    
    for tool_name in tool_names:
        # Update main status
        tool_display = tool_name.replace('_', ' ').title()
        status_text.markdown(f"### Running: **{tool_display}**")
        
        # Show detailed progress in container
        with details_container:
            with st.expander(f"{tool_display} - Details", expanded=True):
                detail_status = st.empty()
                
                tool_progress_bar = st.progress(0)
                tool_progress_text = st.empty()
                
                detail_status.write(f"{tool_descriptions.get(tool_name, 'Running analysis...')}")
                
                # Show what we're checking with progress updates
                if tool_name == 'leakage_detector':
                    tool_progress_bar.progress(0.2)
                    tool_progress_text.text("Step 1/5: Analyzing correlations...")
                    detail_status.write("Analyzing feature correlations with target...")
                    detail_status.write(f"   - Checking {len(df.columns)-1} features for perfect correlations")
                    tool_progress_bar.progress(0.4)
                    tool_progress_text.text("Step 2/5: Scanning feature names...")
                    detail_status.write(f"   - Scanning suspicious feature names")
                    tool_progress_bar.progress(0.6)
                    tool_progress_text.text("Step 3/5: Detecting derived features...")
                    detail_status.write(f"   - Detecting identical/derived features")
                    tool_progress_bar.progress(0.8)
                    tool_progress_text.text("Step 4/5: Finalizing checks...")
                    
                elif tool_name == 'spurious_detector' and is_large:
                    tool_progress_bar.progress(0.15)
                    tool_progress_text.text("Step 1/6: Applying intelligent sampling...")
                    detail_status.write(f"Large dataset detected - using adaptive sampling strategy")
                    detail_status.write(f"   - Priority 1: Stratified sampling (preserves class balance)")
                    detail_status.write(f"   - Priority 2: Quantile sampling (preserves distributions)")
                    detail_status.write(f"   - Priority 3: Cluster sampling (preserves structure)")
                    tool_progress_bar.progress(0.3)
                    tool_progress_text.text("Step 2/6: Testing single features...")
                    detail_status.write(f"   - Analyzing 10K representative samples from {len(df):,} rows")
                    tool_progress_bar.progress(0.5)
                    tool_progress_text.text("Step 3/6: Running cross-validation...")
                    detail_status.write(f"   - Running cross-validation on sampled data")
                    tool_progress_bar.progress(0.7)
                    tool_progress_text.text("Step 4/6: Analyzing importance...")
                    detail_status.write(f"   - Computing feature importance scores")
                    tool_progress_bar.progress(0.85)
                    tool_progress_text.text("Step 5/6: Checking thresholds...")
                    
                elif tool_name == 'bias_detector':
                    tool_progress_bar.progress(0.25)
                    tool_progress_text.text("Step 1/4: Checking class balance...")
                    detail_status.write("Checking class distribution...")
                    detail_status.write(f"   - Analyzing {df[target_column].nunique()} target classes")
                    tool_progress_bar.progress(0.5)
                    tool_progress_text.text("Step 2/4: Analyzing distributions...")
                    tool_progress_bar.progress(0.75)
                    tool_progress_text.text("Step 3/4: Detecting missing patterns...")
                    detail_status.write(f"   - Detecting missing value patterns")
                    
                elif tool_name == 'contamination_detector':
                    tool_progress_bar.progress(0.3)
                    tool_progress_text.text("Step 1/3: Searching duplicates...")
                    detail_status.write("Searching for duplicate samples...")
                    tool_progress_bar.progress(0.6)
                    tool_progress_text.text("Step 2/3: Hashing rows...")
                    detail_status.write(f"   - Hashing {len(df):,} rows")
                    tool_progress_bar.progress(0.9)
                    tool_progress_text.text("Step 3/3: Comparing samples...")
                    
                elif tool_name == 'feature_utility':
                    tool_progress_bar.progress(0.2)
                    tool_progress_text.text("Step 1/5: Checking variance...")
                    detail_status.write("Evaluating feature quality...")
                    detail_status.write(f"   - Checking variance across {len(df.columns)} features")
                    tool_progress_bar.progress(0.4)
                    tool_progress_text.text("Step 2/5: Finding constants...")
                    tool_progress_bar.progress(0.6)
                    tool_progress_text.text("Step 3/5: Detecting redundancy...")
                    tool_progress_bar.progress(0.8)
                    tool_progress_text.text("Step 4/5: Measuring information...")
        
        progress_bar.progress(completed / total_tools)
        
        # Check if should skip
        should_skip, skip_reason = auditor.planner._should_skip_tool(tool_name)
        if should_skip:
            auditor.planner.skipped_tools.append(tool_name)
            auditor.planner.skip_reasons[tool_name] = skip_reason
            with details_container:
                tool_progress_bar.progress(1.0)
                tool_progress_text.text("Skipped")
                st.warning(f"Skipped: {skip_reason}")
            completed += 1
            continue
        
        # Execute tool
        tool_progress_bar.progress(0.95)
        tool_progress_text.text("Analyzing...")
        start_time = time.time()
        findings = auditor._execute_tool(tool_name)
        execution_time = time.time() - start_time
        
        # Complete tool progress
        tool_progress_bar.progress(1.0)
        tool_progress_text.text("Complete!")
        
        # Show results immediately
        with details_container:
            if findings:
                critical = [f for f in findings if f.get('severity') == 'critical']
                warnings = [f for f in findings if f.get('severity') == 'warning']
                
                if critical:
                    st.error(f"Found {len(critical)} critical issue(s) in {execution_time:.2f}s")
                elif warnings:
                    st.warning(f"Found {len(warnings)} warning(s) in {execution_time:.2f}s")
                else:
                    st.info(f"Found {len(findings)} info item(s) in {execution_time:.2f}s")
            else:
                st.success(f"No issues found ({execution_time:.2f}s)")
        
        # Determine status
        if findings:
            critical = [f for f in findings if f.get('severity') == 'critical']
            status = 'fail' if critical else 'warning'
        else:
            status = 'pass'
        
        # Store in memory
        auditor.memory.add_audit_step(tool_name, status, findings, execution_time)
        
        # Critic evaluation
        critique = auditor.critic.evaluate_tool_results(tool_name)
        
        # Show critic assessment
        with details_container:
            if critique.get('concerns'):
                st.write(f"**Critic Confidence:** {critique.get('confidence', 1.0):.0%}")
                if critique.get('needs_recheck'):
                    st.write("Critic recommends deeper analysis...")
        
        # Adaptive re-check
        if critique.get('needs_recheck') and critique.get('confidence', 1.0) < 0.75:
            with details_container:
                with st.spinner("Running adaptive re-check..."):
                    st.write("   → Cross-validating with different parameters")
                    st.write("   → Testing feature stability")
                    recheck_findings = auditor._adaptive_recheck(tool_name, findings)
                    if recheck_findings:
                        findings.extend(recheck_findings)
                        auditor.memory.findings[tool_name] = findings
                        st.success(f"   ✓ Re-check complete: found {len(recheck_findings)} additional issues")
        
        completed += 1
        progress_bar.progress(completed / total_tools)
        
        # Small delay for visual feedback
        time.sleep(0.3)
    
    # Finalize
    auditor.memory.finalize_audit()
    status_text.markdown("### Generating final report...")
    
    with details_container:
        st.write("Aggregating findings across all tools...")
        st.write("Running critic assessment...")
        st.write("Calculating readiness score...")
    
    # Generate report
    report = auditor._generate_report()
    auditor._last_report = report
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.markdown("### Audit Complete!")
    time.sleep(0.5)
    
    return report


def main():
    """Main application"""
    initialize_session_state()
    display_header()
    
    # Sidebar
    df, target_column, filename = upload_dataset()
    
    if df is not None:
        # Audit button
        if st.sidebar.button("Start Audit", type="primary", use_container_width=True):
            st.session_state.show_results = False
            with st.spinner("Initializing autonomous audit..."):
                report = run_audit_with_visualization(df, target_column, filename)
                st.session_state.report = report
                st.session_state.audit_complete = True
            st.rerun()
    
    # Show "View Results" button after audit completes
    if st.session_state.audit_complete and not st.session_state.show_results:
        st.markdown("---")
        
        # Create a centered, prominent results button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <h2 style="color: #48bb78; margin-bottom: 1rem;">Audit Complete!</h2>
                <p style="color: #4a5568; font-size: 1.1rem; margin-bottom: 1.5rem;">
                    Your dataset has been analyzed. Click below to view the detailed results.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View Audit Results", type="primary", use_container_width=True, key="view_results_btn"):
                st.session_state.show_results = True
                st.rerun()
        
        # Show quick summary preview
        if st.session_state.report:
            report = st.session_state.report
            st.markdown("### Quick Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Readiness Score", f"{report['readiness_score']}/100")
            with col2:
                st.metric("Critical Issues", report['summary']['critical_count'])
            with col3:
                st.metric("Warnings", report['summary']['warning_count'])
            with col4:
                verdict = report['verdict']
                if verdict == "READY":
                    st.metric("Status", "Ready")
                elif verdict == "NEEDS ATTENTION":
                    st.metric("Status", "Attention")
                else:
                    st.metric("Status", "Not Ready")
    
    # Display full results if button clicked
    if st.session_state.show_results and st.session_state.report:
        report = st.session_state.report
        
        # Verdict card
        display_verdict_card(report)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Findings", report['summary']['total_findings'])
        with col2:
            st.metric("Critical", report['summary']['critical_count'], delta_color="inverse")
        with col3:
            st.metric("Warnings", report['summary']['warning_count'], delta_color="inverse")
        with col4:
            st.metric("Confidence", f"{report['critic_assessment']['overall_confidence']:.2f}")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_findings_chart(report), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_tool_timeline(report), use_container_width=True)
        
        st.markdown("---")
        
        # Tabs for detailed view
        tab1, tab2, tab3 = st.tabs(["Findings", "Recommendations", "Raw Data"])
        
        with tab1:
            display_findings_detail(report)
        
        with tab2:
            display_recommendations(report)
            
            # Critic assessment
            st.markdown("### Critic Assessment")
            assessment = report['critic_assessment']
            st.markdown(f"**Overall Confidence:** {assessment['overall_confidence']:.2f}")
            st.markdown(f"**Reliability:** {assessment['reliability'].upper()}")
            
            if assessment.get('tools_needing_recheck'):
                st.warning(f"Tools flagged for recheck: {', '.join(assessment['tools_needing_recheck'])}")
        
        with tab3:
            st.json(report)
        
        # Download button
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Download JSON report
            import json
            report_json = json.dumps(report, indent=2, default=str)
            st.download_button(
                label="Download JSON Report",
                data=report_json,
                file_name=f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Download findings CSV
            if report['all_findings']:
                findings_df = pd.DataFrame(report['all_findings'])
                csv = findings_df.to_csv(index=False)
                st.download_button(
                    label="Download Findings CSV",
                    data=csv,
                    file_name=f"audit_findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    else:
        # Welcome screen with tech illustrations
        st.info("Upload a dataset from the sidebar to begin autonomous auditing")
        
        # Central illustration
        st.markdown('''
        <div style="text-align: center; padding: 2rem 0;">
            <svg width="200" height="200" viewBox="0 0 200 200">
                <!-- Data chart -->
                <rect x="20" y="20" width="160" height="120" rx="8" fill="none" stroke="#5B4E8F" stroke-width="3"/>
                <!-- Bar chart inside -->
                <rect x="40" y="90" width="20" height="40" fill="#5B4E8F"/>
                <rect x="70" y="70" width="20" height="60" fill="#7B6BA8"/>
                <rect x="100" y="50" width="20" height="80" fill="#5B4E8F"/>
                <rect x="130" y="60" width="20" height="70" fill="#7B6BA8"/>
                <!-- AI brain -->
                <circle cx="100" cy="170" r="20" fill="none" stroke="#F4A460" stroke-width="3"/>
                <path d="M 90 165 Q 100 160 110 165" fill="none" stroke="#F4A460" stroke-width="2"/>
                <path d="M 90 175 Q 100 170 110 175" fill="none" stroke="#F4A460" stroke-width="2"/>
            </svg>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("### How It Works")
        
        # Process steps with icons
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown('''
            <svg width="60" height="60" viewBox="0 0 60 60">
                <rect x="15" y="20" width="30" height="25" rx="3" fill="none" stroke="#5B4E8F" stroke-width="2"/>
                <path d="M 20 15 L 30 20 L 40 15" fill="none" stroke="#5B4E8F" stroke-width="2"/>
            </svg>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown("**Upload** your dataset (CSV format)")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown('''
            <svg width="60" height="60" viewBox="0 0 60 60">
                <circle cx="30" cy="30" r="18" fill="none" stroke="#5B4E8F" stroke-width="2"/>
                <circle cx="30" cy="30" r="4" fill="#F4A460"/>
            </svg>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown("**Select** the target column")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown('''
            <svg width="60" height="60" viewBox="0 0 60 60">
                <polygon points="30,15 45,40 15,40" fill="none" stroke="#5B4E8F" stroke-width="2"/>
                <path d="M 25 30 L 30 35 L 40 22" fill="none" stroke="#F4A460" stroke-width="2"/>
            </svg>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown("**Click** 'Start Audit' and watch the AI agent work")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown('''
            <svg width="60" height="60" viewBox="0 0 60 60">
                <rect x="15" y="15" width="30" height="35" rx="2" fill="none" stroke="#5B4E8F" stroke-width="2"/>
                <line x1="20" y1="25" x2="40" y2="25" stroke="#5B4E8F" stroke-width="2"/>
                <line x1="20" y1="32" x2="35" y2="32" stroke="#5B4E8F" stroke-width="2"/>
                <line x1="20" y1="39" x2="38" y2="39" stroke="#5B4E8F" stroke-width="2"/>
            </svg>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown("**Review** findings, recommendations, and verdict")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown('''
            <svg width="60" height="60" viewBox="0 0 60 60">
                <path d="M 30 15 L 30 35" stroke="#5B4E8F" stroke-width="2"/>
                <path d="M 22 27 L 30 35 L 38 27" fill="none" stroke="#5B4E8F" stroke-width="2"/>
                <line x1="20" y1="40" x2="40" y2="40" stroke="#5B4E8F" stroke-width="2"/>
            </svg>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown("**Download** detailed reports")
        
        st.markdown("")
        st.markdown("The system autonomously:")
        
        capabilities = [
            ("Plans the audit sequence", '''
            <svg width="50" height="50" viewBox="0 0 50 50">
                <circle cx="25" cy="15" r="6" fill="#5B4E8F"/>
                <circle cx="15" cy="35" r="6" fill="#7B6BA8"/>
                <circle cx="35" cy="35" r="6" fill="#7B6BA8"/>
                <line x1="25" y1="21" x2="18" y2="30" stroke="#5B4E8F" stroke-width="2"/>
                <line x1="25" y1="21" x2="32" y2="30" stroke="#5B4E8F" stroke-width="2"/>
            </svg>
            '''),
            ("Executes specialized detection tools", '''
            <svg width="50" height="50" viewBox="0 0 50 50">
                <rect x="15" y="15" width="20" height="20" fill="none" stroke="#5B4E8F" stroke-width="2"/>
                <circle cx="32" cy="32" r="10" fill="none" stroke="#F4A460" stroke-width="2"/>
                <line x1="39" y1="39" x2="45" y2="45" stroke="#F4A460" stroke-width="2"/>
            </svg>
            '''),
            ("Evaluates confidence in findings", '''
            <svg width="50" height="50" viewBox="0 0 50 50">
                <path d="M 15 35 L 25 15 L 35 35 Z" fill="none" stroke="#5B4E8F" stroke-width="2"/>
                <circle cx="25" cy="30" r="3" fill="#F4A460"/>
            </svg>
            '''),
            ("Triggers deeper analysis when needed", '''
            <svg width="50" height="50" viewBox="0 0 50 50">
                <circle cx="25" cy="25" r="12" fill="none" stroke="#5B4E8F" stroke-width="2"/>
                <circle cx="25" cy="25" r="7" fill="none" stroke="#7B6BA8" stroke-width="2"/>
                <circle cx="25" cy="25" r="2" fill="#F4A460"/>
            </svg>
            '''),
            ("Generates actionable recommendations", '''
            <svg width="50" height="50" viewBox="0 0 50 50">
                <circle cx="25" cy="20" r="8" fill="none" stroke="#F4A460" stroke-width="2"/>
                <line x1="25" y1="28" x2="25" y2="42" stroke="#F4A460" stroke-width="2"/>
                <path d="M 18 38 L 25 42 L 32 38" fill="none" stroke="#F4A460" stroke-width="2"/>
            </svg>
            '''),
        ]
        
        for text, svg in capabilities:
            col1, col2 = st.columns([1, 8])
            with col1:
                st.markdown(svg, unsafe_allow_html=True)
            with col2:
                st.markdown(f"- {text}")
        
        st.markdown("### What We Detect")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('''
            <div style="text-align: center;">
                <svg width="80" height="80" viewBox="0 0 80 80">
                    <circle cx="40" cy="40" r="25" fill="none" stroke="#D84315" stroke-width="3"/>
                    <line x1="55" y1="25" x2="25" y2="55" stroke="#D84315" stroke-width="3"/>
                </svg>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown("""
            **Data Leakage**
            - Target leakage
            - Feature correlation
            - Temporal issues
            """)
        
        with col2:
            st.markdown('''
            <div style="text-align: center;">
                <svg width="80" height="80" viewBox="0 0 80 80">
                    <rect x="20" y="30" width="15" height="30" fill="#F57C00"/>
                    <rect x="45" y="15" width="15" height="45" fill="#F57C00"/>
                    <line x1="20" y1="60" x2="60" y2="60" stroke="#5B4E8F" stroke-width="2"/>
                </svg>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown("""
            **Bias & Imbalance**
            - Class distribution
            - Feature skewness
            - Missing patterns
            """)
        
        with col3:
            st.markdown('''
            <div style="text-align: center;">
                <svg width="80" height="80" viewBox="0 0 80 80">
                    <path d="M 25 50 L 35 30 L 45 40 L 55 20" fill="none" stroke="#4caf50" stroke-width="3"/>
                    <circle cx="25" cy="50" r="4" fill="#4caf50"/>
                    <circle cx="35" cy="30" r="4" fill="#4caf50"/>
                    <circle cx="45" cy="40" r="4" fill="#4caf50"/>
                    <circle cx="55" cy="20" r="4" fill="#4caf50"/>
                </svg>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown("""
            **Quality Issues**
            - Spurious correlations
            - Low-utility features
            - Train-test contamination
            """)


if __name__ == '__main__':
    main()