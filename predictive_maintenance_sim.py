import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from datetime import datetime, timedelta
import time

# Set page configuration
st.set_page_config(
    page_title="Predictive Maintenance Simulation",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class PredictiveMaintenanceSimulator:
    def __init__(self):
        self.machines_data = {}
        self.simulation_results = {}
        
    def generate_failure_time(self, failure_rate_per_1000_hours):
        """Generate random failure time based on exponential distribution"""
        # Convert failure rate to lambda for exponential distribution
        lambda_param = failure_rate_per_1000_hours / 1000
        if lambda_param <= 0:
            return float('inf')
        return np.random.exponential(1 / lambda_param)
    
    def generate_maintenance_time(self, avg_maintenance_time, variability=0.2):
        """Generate maintenance time with some variability"""
        min_time = avg_maintenance_time * (1 - variability)
        max_time = avg_maintenance_time * (1 + variability)
        return np.random.uniform(min_time, max_time)
    
    def run_simulation(self, machines_config, simulation_days, maintenance_strategy):
        """Run the predictive maintenance simulation"""
        results = {
            'machine_type': [],
            'machine_id': [],
            'total_failures': [],
            'total_downtime_hours': [],
            'total_maintenance_cost': [],
            'total_downtime_cost': [],
            'total_cost': [],
            'availability': []
        }
        
        detailed_events = []
        
        for machine_type, config in machines_config.items():
            for machine_id in range(1, config['num_machines'] + 1):
                # Initialize machine state
                current_time = 0
                total_operational_hours = simulation_days * config['daily_hours']
                failures = 0
                total_downtime = 0
                total_maint_cost = 0
                
                events = []
                
                while current_time < total_operational_hours:
                    # Generate next failure time
                    time_to_failure = self.generate_failure_time(config['failure_rate'])
                    failure_time = current_time + time_to_failure
                    
                    if failure_time >= total_operational_hours:
                        break
                    
                    # Apply maintenance strategy
                    if maintenance_strategy == "Reactive":
                        # Wait for failure, then repair
                        maintenance_time = self.generate_maintenance_time(config['maintenance_time'])
                        downtime = maintenance_time
                        
                    elif maintenance_strategy == "Preventive":
                        # Schedule maintenance before failure (80% of expected failure time)
                        scheduled_maintenance_time = failure_time * 0.8
                        maintenance_time = self.generate_maintenance_time(config['maintenance_time'] * 0.7)  # Preventive is faster
                        downtime = maintenance_time
                        failure_time = scheduled_maintenance_time
                        
                    else:  # Predictive
                        # More sophisticated scheduling (85% of expected failure time)
                        scheduled_maintenance_time = failure_time * 0.85
                        maintenance_time = self.generate_maintenance_time(config['maintenance_time'] * 0.6)  # Predictive is fastest
                        downtime = maintenance_time
                        failure_time = scheduled_maintenance_time
                    
                    # Record the event
                    failures += 1
                    total_downtime += downtime
                    total_maint_cost += config['maintenance_cost']
                    
                    events.append({
                        'machine_type': machine_type,
                        'machine_id': f"{machine_type}_{machine_id}",
                        'event_time': failure_time,
                        'maintenance_time': maintenance_time,
                        'strategy': maintenance_strategy
                    })
                    
                    # Move to next time period
                    current_time = failure_time + maintenance_time
                
                # Calculate costs and metrics
                downtime_cost = total_downtime * config['downtime_cost']
                total_cost = total_maint_cost + downtime_cost
                availability = ((total_operational_hours - total_downtime) / total_operational_hours) * 100
                
                # Store results
                results['machine_type'].append(machine_type)
                results['machine_id'].append(f"{machine_type}_{machine_id}")
                results['total_failures'].append(failures)
                results['total_downtime_hours'].append(total_downtime)
                results['total_maintenance_cost'].append(total_maint_cost)
                results['total_downtime_cost'].append(downtime_cost)
                results['total_cost'].append(total_cost)
                results['availability'].append(availability)
                
                detailed_events.extend(events)
        
        return pd.DataFrame(results), pd.DataFrame(detailed_events)

def main():
    st.markdown('<h1 class="main-header">⚙️ Predictive Maintenance Simulation for Smart Manufacturing</h1>', unsafe_allow_html=True)
    st.markdown("**Course:** DA380 (Data Modeling and Simulation) | **Project:** Summer 2025")
    
    # Initialize simulator
    if 'simulator' not in st.session_state:
        st.session_state.simulator = PredictiveMaintenanceSimulator()
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Simulation Configuration")
    
    # Machine types configuration
    st.sidebar.subheader("Machine Types")
    
    machine_types = {}
    
    # CNC Machines
    with st.sidebar.expander("🔩 CNC Machines", expanded=True):
        machine_types['CNC'] = {
            'num_machines': st.number_input("Number of CNC Machines", min_value=1, max_value=20, value=5, key="cnc_num"),
            'failure_rate': st.number_input("Failure Rate (per 1000 hrs)", min_value=0.1, max_value=50.0, value=2.5, step=0.1, key="cnc_failure"),
            'maintenance_time': st.number_input("Avg Maintenance Time (hrs)", min_value=0.5, max_value=24.0, value=4.0, step=0.5, key="cnc_maint_time"),
            'daily_hours': st.number_input("Daily Operational Hours", min_value=8, max_value=24, value=16, key="cnc_daily"),
            'maintenance_cost': st.number_input("Maintenance Cost ($)", min_value=100, max_value=10000, value=1500, step=100, key="cnc_maint_cost"),
            'downtime_cost': st.number_input("Downtime Cost ($/hr)", min_value=10, max_value=1000, value=200, step=10, key="cnc_down_cost")
        }
    
    # Conveyor Belts
    with st.sidebar.expander("🔗 Conveyor Belts"):
        machine_types['Conveyor'] = {
            'num_machines': st.number_input("Number of Conveyor Belts", min_value=1, max_value=20, value=8, key="conv_num"),
            'failure_rate': st.number_input("Failure Rate (per 1000 hrs)", min_value=0.1, max_value=50.0, value=1.8, step=0.1, key="conv_failure"),
            'maintenance_time': st.number_input("Avg Maintenance Time (hrs)", min_value=0.5, max_value=24.0, value=2.5, step=0.5, key="conv_maint_time"),
            'daily_hours': st.number_input("Daily Operational Hours", min_value=8, max_value=24, value=20, key="conv_daily"),
            'maintenance_cost': st.number_input("Maintenance Cost ($)", min_value=100, max_value=10000, value=800, step=100, key="conv_maint_cost"),
            'downtime_cost': st.number_input("Downtime Cost ($/hr)", min_value=10, max_value=1000, value=150, step=10, key="conv_down_cost")
        }
    
    # Robotic Arms
    with st.sidebar.expander("🤖 Robotic Arms"):
        machine_types['Robot'] = {
            'num_machines': st.number_input("Number of Robotic Arms", min_value=1, max_value=20, value=3, key="robot_num"),
            'failure_rate': st.number_input("Failure Rate (per 1000 hrs)", min_value=0.1, max_value=50.0, value=3.2, step=0.1, key="robot_failure"),
            'maintenance_time': st.number_input("Avg Maintenance Time (hrs)", min_value=0.5, max_value=24.0, value=6.0, step=0.5, key="robot_maint_time"),
            'daily_hours': st.number_input("Daily Operational Hours", min_value=8, max_value=24, value=18, key="robot_daily"),
            'maintenance_cost': st.number_input("Maintenance Cost ($)", min_value=100, max_value=10000, value=2500, step=100, key="robot_maint_cost"),
            'downtime_cost': st.number_input("Downtime Cost ($/hr)", min_value=10, max_value=1000, value=300, step=10, key="robot_down_cost")
        }
    
    # Simulation parameters
    st.sidebar.subheader("📊 Simulation Parameters")
    simulation_days = st.sidebar.number_input("Simulation Period (days)", min_value=30, max_value=365, value=90)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📈 Scenario Analysis</h2>', unsafe_allow_html=True)
        
        # Run simulations for different strategies
        if st.button("🚀 Run All Scenarios", type="primary", use_container_width=True):
            strategies = ["Reactive", "Preventive", "Predictive"]
            scenario_results = {}
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, strategy in enumerate(strategies):
                status_text.text(f"Running {strategy} Maintenance Strategy...")
                
                # Set random seed for reproducible results
                np.random.seed(42 + i)
                random.seed(42 + i)
                
                results_df, events_df = st.session_state.simulator.run_simulation(
                    machine_types, simulation_days, strategy
                )
                
                scenario_results[strategy] = {
                    'results': results_df,
                    'events': events_df,
                    'summary': {
                        'total_failures': results_df['total_failures'].sum(),
                        'total_downtime': results_df['total_downtime_hours'].sum(),
                        'total_maintenance_cost': results_df['total_maintenance_cost'].sum(),
                        'total_downtime_cost': results_df['total_downtime_cost'].sum(),
                        'total_cost': results_df['total_cost'].sum(),
                        'avg_availability': results_df['availability'].mean()
                    }
                }
                
                progress_bar.progress((i + 1) / len(strategies))
            
            status_text.text("✅ All scenarios completed!")
            st.session_state.scenario_results = scenario_results
            
            # Clear progress indicators after a short delay
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
    
    with col2:
        st.markdown('<h2 class="section-header">📋 Configuration Summary</h2>', unsafe_allow_html=True)
        
        # Display machine summary
        total_machines = sum(config['num_machines'] for config in machine_types.values())
        st.metric("Total Machines", total_machines)
        st.metric("Simulation Period", f"{simulation_days} days")
        
        for machine_type, config in machine_types.items():
            st.markdown(f"**{machine_type}:** {config['num_machines']} units")
    
    # Display results if available
    if hasattr(st.session_state, 'scenario_results'):
        st.markdown('<h2 class="section-header">📊 Simulation Results</h2>', unsafe_allow_html=True)
        
        # Summary comparison
        col1, col2, col3 = st.columns(3)
        
        strategies = ["Reactive", "Preventive", "Predictive"]
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        with col1:
            st.subheader("🔧 Reactive Maintenance")
            reactive_summary = st.session_state.scenario_results['Reactive']['summary']
            st.metric("Total Cost", f"${reactive_summary['total_cost']:,.0f}")
            st.metric("Total Failures", f"{reactive_summary['total_failures']}")
            st.metric("Availability", f"{reactive_summary['avg_availability']:.1f}%")
        
        with col2:
            st.subheader("📅 Preventive Maintenance")
            preventive_summary = st.session_state.scenario_results['Preventive']['summary']
            st.metric("Total Cost", f"${preventive_summary['total_cost']:,.0f}")
            st.metric("Total Failures", f"{preventive_summary['total_failures']}")
            st.metric("Availability", f"{preventive_summary['avg_availability']:.1f}%")
        
        with col3:
            st.subheader("🎯 Predictive Maintenance")
            predictive_summary = st.session_state.scenario_results['Predictive']['summary']
            st.metric("Total Cost", f"${predictive_summary['total_cost']:,.0f}")
            st.metric("Total Failures", f"{predictive_summary['total_failures']}")
            st.metric("Availability", f"{predictive_summary['avg_availability']:.1f}%")
        
        # Comparative charts
        st.markdown('<h3 class="section-header">📈 Comparative Analysis</h3>', unsafe_allow_html=True)
        
        # Prepare data for comparison charts
        comparison_data = []
        for strategy in strategies:
            summary = st.session_state.scenario_results[strategy]['summary']
            comparison_data.append({
                'Strategy': strategy,
                'Total Cost': summary['total_cost'],
                'Total Failures': summary['total_failures'],
                'Total Downtime (hrs)': summary['total_downtime'],
                'Availability (%)': summary['avg_availability'],
                'Maintenance Cost': summary['total_maintenance_cost'],
                'Downtime Cost': summary['total_downtime_cost']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Cost Comparison', 'Availability Comparison', 
                           'Failure Count Comparison', 'Cost Breakdown'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "domain"}]]
        )
        
        # Total Cost Chart
        fig.add_trace(
            go.Bar(x=comparison_df['Strategy'], y=comparison_df['Total Cost'],
                   name='Total Cost', marker_color=colors),
            row=1, col=1
        )
        
        # Availability Chart
        fig.add_trace(
            go.Bar(x=comparison_df['Strategy'], y=comparison_df['Availability (%)'],
                   name='Availability', marker_color=colors),
            row=1, col=2
        )
        
        # Failure Count Chart
        fig.add_trace(
            go.Bar(x=comparison_df['Strategy'], y=comparison_df['Total Failures'],
                   name='Failures', marker_color=colors),
            row=2, col=1
        )
        
        # Cost Breakdown Pie Chart (for Predictive strategy as example)
        predictive_maint_cost = st.session_state.scenario_results['Predictive']['summary']['total_maintenance_cost']
        predictive_down_cost = st.session_state.scenario_results['Predictive']['summary']['total_downtime_cost']
        
        fig.add_trace(
            go.Pie(labels=['Maintenance Cost', 'Downtime Cost'],
                   values=[predictive_maint_cost, predictive_down_cost],
                   name="Cost Breakdown (Predictive)"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Maintenance Strategy Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed machine-level analysis
        st.markdown('<h3 class="section-header">🔍 Detailed Machine Analysis</h3>', unsafe_allow_html=True)
        
        selected_strategy = st.selectbox("Select Strategy for Detailed View", strategies)
        
        if selected_strategy:
            results_df = st.session_state.scenario_results[selected_strategy]['results']
            
            # Machine performance table
            st.subheader(f"{selected_strategy} Maintenance - Machine Performance")
            
            # Format the dataframe for display
            display_df = results_df.copy()
            display_df['total_maintenance_cost'] = display_df['total_maintenance_cost'].apply(lambda x: f"${x:,.0f}")
            display_df['total_downtime_cost'] = display_df['total_downtime_cost'].apply(lambda x: f"${x:,.0f}")
            display_df['total_cost'] = display_df['total_cost'].apply(lambda x: f"${x:,.0f}")
            display_df['availability'] = display_df['availability'].apply(lambda x: f"{x:.1f}%")
            display_df['total_downtime_hours'] = display_df['total_downtime_hours'].apply(lambda x: f"{x:.1f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Machine type comparison
            machine_summary = results_df.groupby('machine_type').agg({
                'total_failures': 'sum',
                'total_downtime_hours': 'sum',
                'total_cost': 'sum',
                'availability': 'mean'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_failures = px.bar(machine_summary, x='machine_type', y='total_failures',
                                    title=f'Total Failures by Machine Type ({selected_strategy})')
                st.plotly_chart(fig_failures, use_container_width=True)
            
            with col2:
                fig_cost = px.bar(machine_summary, x='machine_type', y='total_cost',
                                title=f'Total Cost by Machine Type ({selected_strategy})')
                st.plotly_chart(fig_cost, use_container_width=True)
        
        # Key insights and recommendations
        st.markdown('<h3 class="section-header">💡 Key Insights and Recommendations</h3>', unsafe_allow_html=True)
        
        # Calculate best strategy
        best_cost_strategy = min(strategies, key=lambda s: st.session_state.scenario_results[s]['summary']['total_cost'])
        best_availability_strategy = max(strategies, key=lambda s: st.session_state.scenario_results[s]['summary']['avg_availability'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"🏆 **Most Cost-Effective:** {best_cost_strategy} Maintenance")
            cost_savings = (st.session_state.scenario_results['Reactive']['summary']['total_cost'] - 
                           st.session_state.scenario_results[best_cost_strategy]['summary']['total_cost'])
            if cost_savings > 0:
                st.info(f"💰 Potential savings: ${cost_savings:,.0f} compared to Reactive maintenance")
        
        with col2:
            st.success(f"📈 **Highest Availability:** {best_availability_strategy} Maintenance")
            availability_improvement = (st.session_state.scenario_results[best_availability_strategy]['summary']['avg_availability'] - 
                                      st.session_state.scenario_results['Reactive']['summary']['avg_availability'])
            if availability_improvement > 0:
                st.info(f"⬆️ Availability improvement: {availability_improvement:.1f}% compared to Reactive maintenance")
        
        # Recommendations
        st.markdown("""
        ### 🎯 Optimization Recommendations:
        
        1. **Implement Predictive Maintenance** for high-value, critical machines (CNC machines, Robotic Arms)
        2. **Use Preventive Maintenance** for simpler, lower-cost machines (Conveyor Belts)
        3. **Focus on reducing downtime costs** through faster repair processes and spare parts availability
        4. **Invest in IoT sensors** and machine learning algorithms to improve failure prediction accuracy
        5. **Regular review and adjustment** of maintenance schedules based on actual performance data
        """)
        
        # Export functionality
        st.markdown('<h3 class="section-header">📁 Export Results</h3>', unsafe_allow_html=True)
        
        if st.button("📊 Generate Comprehensive Report", use_container_width=True):
            # Create a comprehensive summary
            report_data = {
                'Scenario_Comparison': comparison_df,
                'Detailed_Results_Reactive': st.session_state.scenario_results['Reactive']['results'],
                'Detailed_Results_Preventive': st.session_state.scenario_results['Preventive']['results'],
                'Detailed_Results_Predictive': st.session_state.scenario_results['Predictive']['results']
            }
            
            st.success("📋 Report generated! You can copy the data from the tables above for your final report.")
            
            # Show summary for final report
            st.markdown("### 📝 Summary for Final Report:")
            st.json({
                'simulation_period_days': simulation_days,
                'total_machines': total_machines,
                'best_strategy_cost': best_cost_strategy,
                'best_strategy_availability': best_availability_strategy,
                'cost_comparison': {strategy: f"${st.session_state.scenario_results[strategy]['summary']['total_cost']:,.0f}" 
                                  for strategy in strategies},
                'availability_comparison': {strategy: f"{st.session_state.scenario_results[strategy]['summary']['avg_availability']:.1f}%" 
                                          for strategy in strategies}
            })

if __name__ == "__main__":
    main()