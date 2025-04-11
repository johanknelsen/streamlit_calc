import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from tqdm import tqdm

def simulate_compound_interest(initial_investment, years, avg_annual_return, min_rate, max_rate, 
                              withdrawal_steps=None, simulations=10000):
    """
    Simulates compound interest growth with variable annual returns and multiple withdrawal steps.
    
    Parameters:
    - initial_investment: Starting investment amount
    - years: Length of investment in years
    - avg_annual_return: Expected average annual return (as decimal, e.g. 0.07 for 7%)
    - min_rate: Minimum possible annual return (as decimal)
    - max_rate: Maximum possible annual return (as decimal)
    - withdrawal_steps: List of tuples (start_year, end_year, amount) for each withdrawal step
    - simulations: Number of simulations to run
    
    Returns:
    - results: Array of final values for each simulation
    - paths: Array of growth paths (for visualization)
    - depleted_count: Number of simulations where the investment was depleted
    - rates_by_path: Array of annual rates for each path (for median path analysis)
    """
    # Initialize withdrawal steps if None
    if withdrawal_steps is None:
        withdrawal_steps = []
    
    # Sort withdrawal steps by start year
    withdrawal_steps.sort(key=lambda x: x[0])
    
    # Create array to store final values
    results = np.zeros(simulations)
    depleted_count = 0
    
    # Create array to store a subset of paths for visualization
    viz_count = min(100, simulations)  # Store only up to 100 paths for visualization
    paths = np.zeros((viz_count, years + 1))
    paths[:, 0] = initial_investment
    
    # Store annual rates for each path (for the visualization subset)
    rates_by_path = np.zeros((viz_count, years))
    
    # Create a progress bar for Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run simulations
    for i in range(simulations):
        # Update progress
        if i % 100 == 0 or i == simulations - 1:
            progress_bar.progress(min((i + 1) / simulations, 1.0))
            status_text.text(f"Running simulation {i+1}/{simulations}")
            
        value = initial_investment
        depleted = False
        
        # For each year, apply a random return within the specified range
        yearly_values = [value]
        annual_rates = []  # Store the annual rates for this path
        
        for year in range(years):
            # Generate random return between min_rate and max_rate
            # Using a normal distribution centered around avg_annual_return
            std_dev = (max_rate - min_rate) / 4  # This ensures about 95% of values fall within range
            annual_return = np.random.normal(avg_annual_return, std_dev)
            
            # Clip to ensure we stay within min and max rates
            annual_return = max(min_rate, min(max_rate, annual_return))
            annual_rates.append(annual_return)
            
            # Apply compound interest for this year
            prev_value = value
            value *= (1 + annual_return)
            
            # Apply withdrawals if applicable
            for start_year, end_year, withdrawal_amount in withdrawal_steps:
                # Check if this withdrawal step is active for this year
                current_year = year + 1  # Convert from 0-indexed to 1-indexed
                
                # Skip if we're before the start year or after the end year
                if current_year < start_year or (end_year is not None and current_year > end_year):
                    continue
                    
                # Handle first partial year if applicable
                if year < start_year and current_year > start_year:
                    # Calculate the fraction of the year to apply the withdrawal
                    fraction = current_year - start_year
                    value -= withdrawal_amount * fraction
                else:
                    # Apply full withdrawal for complete years
                    value -= withdrawal_amount
            
            # Check if portfolio is depleted
            if value <= 0:
                value = 0
                depleted = True
                depleted_count += 1
                break
            
            yearly_values.append(value)
        
        # If simulation ended early due to depletion, pad with zeros
        if depleted and len(yearly_values) < years + 1:
            yearly_values.extend([0] * (years + 1 - len(yearly_values)))
            annual_rates.extend([0] * (years - len(annual_rates)))
        
        # Store the final value
        results[i] = value
        
        # Store path and rates for visualization (only for a subset)
        if i < viz_count:
            if len(yearly_values) == years + 1:
                paths[i, :] = yearly_values
            else:
                # This shouldn't happen with the padding above, but just in case
                paths[i, :len(yearly_values)] = yearly_values
                paths[i, len(yearly_values):] = 0
                
            rates_by_path[i, :len(annual_rates)] = annual_rates
            if len(annual_rates) < years:
                rates_by_path[i, len(annual_rates):] = 0
    
    # Clear the progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return results, paths, depleted_count, rates_by_path

def main():
    # Set page config
    st.set_page_config(
        page_title="Monte Carlo Compound Interest Simulator",
        page_icon="📈",
        layout="wide"
    )
    
    # App title and description
    st.title("Compound Interest Monte Carlo Simulator")
    st.markdown("""
    This simulator helps you visualize how your investments might grow over time, 
    accounting for market volatility and planned withdrawals. The simulation runs 
    thousands of scenarios using Monte Carlo methods.
    """)
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("Investment Parameters")
        
        initial_investment = st.number_input(
            "Initial Investment ($)",
            min_value=100,
            max_value=10000000,
            value=10000,
            step=1000
        )
        
        years = st.number_input(
            "Investment Period (years)",
            min_value=1,
            max_value=100,
            value=30,
            step=1
        )
        
        avg_annual_return = st.number_input(
            "Expected Annual Return (%)",
            min_value=-10.0,
            max_value=30.0,
            value=7.0,
            step=0.5
        ) / 100
        
        min_rate = st.number_input(
            "Minimum Annual Return (%)",
            min_value=-50.0,
            max_value=20.0,
            value=-5.0,
            step=1.0
        ) / 100
        
        max_rate = st.number_input(
            "Maximum Annual Return (%)",
            min_value=0.0,
            max_value=50.0,
            value=20.0,
            step=1.0
        ) / 100
        
        simulations = st.number_input(
            "Number of Simulations",
            min_value=100,
            max_value=100000,
            value=10000,
            step=1000
        )
        
    # Main content area for withdrawal plan
    st.header("Withdrawal Plan")
    st.markdown("Add withdrawal steps to model regular withdrawals from your investment.")
    
    # Create a state for withdrawal steps
    if "withdrawal_steps" not in st.session_state:
        # Initialize with one step
        st.session_state.withdrawal_steps = [
            {"start_year": 10, "end_year": None, "amount": 500}
        ]
    
    # Display withdrawal steps as a dataframe for editing
    withdrawal_df = pd.DataFrame(st.session_state.withdrawal_steps)
    
    # Create columns for the table and buttons
    col1, col2 = st.columns([3, 1])
    
    with col1:
        edited_df = st.data_editor(
            withdrawal_df,
            column_config={
                "start_year": st.column_config.NumberColumn(
                    "Start Year",
                    min_value=0,
                    max_value=years,
                    step=1,
                    help="When to start this withdrawal"
                ),
                "end_year": st.column_config.NumberColumn(
                    "End Year (optional)",
                    min_value=0,
                    max_value=years,
                    step=1,
                    help="When to end this withdrawal (leave blank for until end)"
                ),
                "amount": st.column_config.NumberColumn(
                    "Annual Amount ($)",
                    min_value=0,
                    step=100,
                    help="Amount to withdraw each year"
                )
            },
            hide_index=True,
            num_rows="dynamic",
            key="withdrawal_editor"
        )
    
    with col2:
        if st.button("Update Withdrawal Plan"):
            # Update the session state with edited values
            st.session_state.withdrawal_steps = edited_df.to_dict('records')
            st.success("Withdrawal plan updated!")
    
    # Prepare withdrawal steps for simulation
    withdrawal_steps = []
    for step in st.session_state.withdrawal_steps:
        start_year = step['start_year']
        end_year = step['end_year'] if pd.notna(step['end_year']) else None
        amount = step['amount']
        if pd.notna(start_year) and pd.notna(amount) and amount > 0:
            withdrawal_steps.append((start_year, end_year, amount))
    
    # Run simulation button
    if st.button("Run Simulation", type="primary"):
        # Verify inputs
        if min_rate >= max_rate:
            st.error("Minimum return rate must be less than maximum return rate")
        else:
            # Run the simulation
            with st.spinner("Running simulation..."):
                results, paths, depleted_count, rates_by_path = simulate_compound_interest(
                    initial_investment, years, avg_annual_return, min_rate, max_rate, 
                    withdrawal_steps, simulations
                )
            
            # Calculate statistics for non-depleted portfolios
            non_zero_results = results[results > 0]
            if len(non_zero_results) > 0:
                min_result = np.min(non_zero_results)
                max_result = np.max(results)
                median_result = np.median(non_zero_results)
                
                if len(non_zero_results) > len(results) * 0.1:
                    percentile_10 = np.percentile(non_zero_results, 10)
                else:
                    percentile_10 = 0
                    
                if len(non_zero_results) > len(results) * 0.9:
                    percentile_90 = np.percentile(non_zero_results, 90)
                else:
                    percentile_90 = np.percentile(results, 90)
            else:
                min_result = max_result = median_result = percentile_10 = percentile_90 = 0
            
            # Success rate (non-depleted portfolios)
            success_rate = 100 * (1 - depleted_count / len(results))
            
            # Calculate total withdrawals
            total_withdrawn = 0
            for start_year, end_year, amount in withdrawal_steps:
                # Calculate years of this withdrawal step
                end_year = end_year if end_year is not None else years
                years_active = min(end_year, years) - start_year
                if years_active > 0:
                    total_withdrawn += amount * years_active
            
            # Display results in multiple columns
            st.header("Simulation Results")
            
            # Create three columns for main stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="Median Final Balance", 
                          value=f"${median_result:,.2f}", 
                          delta=f"{(median_result / initial_investment - 1):.1%} growth")
                
                st.metric(label="Success Rate", 
                          value=f"{success_rate:.1f}%",
                          delta=f"{depleted_count} depleted scenarios" if depleted_count > 0 else "No depleted scenarios",
                          delta_color="inverse")
            
            with col2:
                st.metric(label="10th Percentile", 
                          value=f"${percentile_10:,.2f}", 
                          delta=f"{(percentile_10 / initial_investment - 1):.1%} growth")
                
                st.metric(label="90th Percentile", 
                          value=f"${percentile_90:,.2f}", 
                          delta=f"{(percentile_90 / initial_investment - 1):.1%} growth")
                
            with col3:
                st.metric(label="Minimum Final Balance", 
                          value=f"${min_result:,.2f}", 
                          delta=f"{(min_result / initial_investment - 1):.1%} growth")
                
                st.metric(label="Maximum Final Balance", 
                          value=f"${max_result:,.2f}", 
                          delta=f"{(max_result / initial_investment - 1):.1%} growth")
            
            if withdrawal_steps:
                st.subheader("Withdrawal Analysis")
                st.metric(label="Total Withdrawn", value=f"${total_withdrawn:,.2f}")
                
                if success_rate < 95:
                    st.warning("Current withdrawal plan may not be sustainable")
                elif success_rate > 99:
                    st.success("Current withdrawal plan appears very sustainable")
                else:
                    st.info("Current withdrawal plan appears reasonably sustainable")
            
            # Create visualization in tabs
            tab1, tab2, tab3 = st.tabs(["Growth Paths", "Final Balance Distribution", "Yearly Detail Table"])
            
            with tab1:
                # Set up the figure for growth paths using Plotly
                import plotly.graph_objects as go

                fig = go.Figure()

                # Plot all paths
                for path in paths:
                    fig.add_trace(go.Scatter(
                        x=list(range(years + 1)),
                        y=path,
                        mode='lines',
                        line=dict(color='lightblue', width=1),
                        opacity=0.1,
                        showlegend=False
                    ))

                # Plot min, median, max paths from non-depleted paths
                non_depleted_indices = [i for i, path in enumerate(paths) if path[-1] > 0]
                if non_depleted_indices:
                    non_depleted_paths = paths[non_depleted_indices]
                    if len(non_depleted_paths) > 0:
                        min_idx = np.argmin(non_depleted_paths[:, -1])
                        med_idx = len(non_depleted_paths) // 2  # Approximate median
                        max_idx = np.argmax(non_depleted_paths[:, -1])

                        fig.add_trace(go.Scatter(
                            x=list(range(years + 1)),
                            y=non_depleted_paths[min_idx],
                            mode='lines',
                            line=dict(color='red', width=2),
                            name='Minimum Surviving Path'
                        ))
                        fig.add_trace(go.Scatter(
                            x=list(range(years + 1)),
                            y=non_depleted_paths[med_idx],
                            mode='lines',
                            line=dict(color='green', width=2),
                            name='Median Path'
                        ))
                        fig.add_trace(go.Scatter(
                            x=list(range(years + 1)),
                            y=non_depleted_paths[max_idx],
                            mode='lines',
                            line=dict(color='blue', width=2),
                            name='Maximum Path'
                        ))

                # Add a depleted path if there were any
                depleted_indices = [i for i, path in enumerate(paths) if path[-1] == 0]
                if depleted_indices and withdrawal_steps:
                    # Find the path that depleted fastest
                    depletion_points = []
                    for idx in depleted_indices:
                        # Find where the path first hits zero
                        zero_points = np.where(paths[idx] == 0)[0]
                        if len(zero_points) > 0:
                            depletion_points.append((idx, zero_points[0]))

                    if depletion_points:
                        # Sort by the year of depletion
                        depletion_points.sort(key=lambda x: x[1])
                        fastest_depletion_idx = depletion_points[0][0]
                        fig.add_trace(go.Scatter(
                            x=list(range(years + 1)),
                            y=paths[fastest_depletion_idx],
                            mode='lines',
                            line=dict(color='red', dash='dash', width=2),
                            name='Example Depleted Path'
                        ))

                # Add layout details
                fig.update_layout(
                    title='Investment Growth Scenarios',
                    xaxis_title='Years',
                    yaxis_title='Investment Value ($)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template='plotly_white'
                )

                st.plotly_chart(fig)

                # Show explanation
                st.markdown("""
                **Chart Explanation:**
                - **Light Blue Lines**: Individual simulation paths (showing 100 of the total simulations)
                - **Red Line**: The minimum surviving path (portfolio didn't deplete)
                - **Green Line**: The median path
                - **Blue Line**: The maximum path
                - **Red Dashed Line**: Example of a path where the portfolio depleted
                """)

            with tab2:
                # Set up the figure for histogram using Plotly
                fig = go.Figure()

                # If there were many depleted paths, show them separately
                if depleted_count > 0:
                    # First histogram for non-depleted results
                    fig.add_trace(go.Histogram(
                        x=non_zero_results,
                        nbinsx=50,
                        marker=dict(color='skyblue', line=dict(color='black', width=1)),
                        opacity=0.7,
                        name=f'Surviving Portfolios ({success_rate:.1f}%)'
                    ))

                    # Add a vertical line for depleted cases
                    fig.add_trace(go.Scatter(
                        x=[0, 0],
                        y=[0, len(non_zero_results)],
                        mode='lines',
                        line=dict(color='red', width=2),
                        name=f'Depleted Portfolios ({100-success_rate:.1f}%)'
                    ))
                else:
                    # Regular histogram
                    fig.add_trace(go.Histogram(
                        x=results,
                        nbinsx=50,
                        marker=dict(color='skyblue', line=dict(color='black', width=1)),
                        opacity=0.7
                    ))

                # Add lines for statistics
                if min_result > 0:
                    fig.add_trace(go.Scatter(
                        x=[min_result, min_result],
                        y=[0, len(non_zero_results)],
                        mode='lines',
                        line=dict(color='red', dash='dash', width=2),
                        name=f'Min: ${min_result:,.2f}'
                    ))
                if median_result > 0:
                    fig.add_trace(go.Scatter(
                        x=[median_result, median_result],
                        y=[0, len(non_zero_results)],
                        mode='lines',
                        line=dict(color='green', dash='dash', width=2),
                        name=f'Median: ${median_result:,.2f}'
                    ))
                fig.add_trace(go.Scatter(
                    x=[max_result, max_result],
                    y=[0, len(non_zero_results)],
                    mode='lines',
                    line=dict(color='blue', dash='dash', width=2),
                    name=f'Max: ${max_result:,.2f}'
                ))

                # Add layout details
                fig.update_layout(
                    title='Distribution of Final Investment Values',
                    xaxis_title='Final Value ($)',
                    yaxis_title='Frequency',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template='plotly_white'
                )

                st.plotly_chart(fig)

                # Show explanation
                st.markdown("""
                **Chart Explanation:**
                - This histogram shows the distribution of final investment values across all simulations
                - The vertical lines show the minimum, median, and maximum final values
                - If any portfolios depleted to zero, they are shown as a red vertical line at $0
                """)
            
            with tab3:
                st.subheader("Yearly Detail for Median Path")
                
                # Find the median path
                non_depleted_indices = [i for i, path in enumerate(paths) if path[-1] > 0]
                if non_depleted_indices:
                    non_depleted_paths = paths[non_depleted_indices]
                    if len(non_depleted_paths) > 0:
                        # Find the index of the median path
                        med_idx = len(non_depleted_indices) // 2
                        med_path_idx = non_depleted_indices[med_idx]
                        
                        # Get the median path values and rates
                        median_path = paths[med_path_idx]
                        median_rates = rates_by_path[med_path_idx]
                        
                        # Create the yearly detail table
                        yearly_data = []
                        accumulated_interest = 0
                        total_withdrawals = 0
                        
                        for year in range(years):
                            start_value = median_path[year]
                            end_value = median_path[year + 1]
                            interest_rate = median_rates[year]
                            
                            # Calculate interest accrued before withdrawals
                            interest_accrued = start_value * interest_rate
                            
                            # Calculate withdrawals for this year
                            year_withdrawals = 0
                            for start_yr, end_yr, amount in withdrawal_steps:
                                current_year = year + 1  # 1-indexed for withdrawals
                                if current_year >= start_yr and (end_yr is None or current_year <= end_yr):
                                    year_withdrawals += amount
                            
                            # Accumulated interest
                            accumulated_interest += interest_accrued
                            total_withdrawals += year_withdrawals
                            
                            yearly_data.append({
                                "Year": year + 1,
                                "Beginning Balance": f"${start_value:,.2f}",
                                "Interest Rate": f"{interest_rate:.2%}",
                                "Interest Accrued": f"${interest_accrued:,.2f}",
                                "Withdrawals": f"${year_withdrawals:,.2f}",
                                "Accumulated Interest": f"${accumulated_interest:,.2f}",
                                "Ending Balance": f"${end_value:,.2f}"
                            })
                        
                        # Display the table
                        st.table(pd.DataFrame(yearly_data))
                        
                        # Show summary statistics
                        st.metric(label="Total Interest Accrued", value=f"${accumulated_interest:,.2f}")
                        if total_withdrawals > 0:
                            st.metric(label="Total Withdrawals", value=f"${total_withdrawals:,.2f}")
                        
                        st.markdown("""
                        **Table Explanation:**
                        - This table shows yearly details for the median simulation path
                        - Beginning Balance: Account value at start of year
                        - Interest Rate: The random rate generated for this year
                        - Interest Accrued: Interest earned during the year
                        - Withdrawals: Amount withdrawn during the year
                        - Accumulated Interest: Total interest earned since start
                        - Ending Balance: Account value at end of year after interest and withdrawals
                        """)
                else:
                    st.warning("No surviving paths to analyze. Try adjusting your withdrawal plan or investment parameters.")
            
            with tab3:
                st.subheader("Yearly Detail for 10th Percentile Path")
                
                # Find the 10th percentile path
                non_depleted_indices = [i for i, path in enumerate(paths) if path[-1] > 0]
                if non_depleted_indices:
                    non_depleted_paths = paths[non_depleted_indices]
                    if len(non_depleted_paths) > 0:
                        # Calculate the 10th percentile value
                        percentile_10_value = np.percentile([path[-1] for path in non_depleted_paths], 10)
                        
                        # Find the index of the path closest to the 10th percentile value
                        closest_idx = np.argmin(np.abs([path[-1] for path in non_depleted_paths] - percentile_10_value))
                        perc10_path_idx = non_depleted_indices[closest_idx]
                        
                        # Get the 10th percentile path values and rates
                        perc10_path = paths[perc10_path_idx]
                        perc10_rates = rates_by_path[perc10_path_idx]
                        
                        # Create the yearly detail table
                        yearly_data = []
                        accumulated_interest = 0
                        total_withdrawals = 0
                        
                        for year in range(years):
                            start_value = perc10_path[year]
                            end_value = perc10_path[year + 1]
                            interest_rate = perc10_rates[year]
                            
                            # Calculate interest accrued before withdrawals
                            interest_accrued = start_value * interest_rate
                            
                            # Calculate withdrawals for this year
                            year_withdrawals = 0
                            for start_yr, end_yr, amount in withdrawal_steps:
                                current_year = year + 1  # 1-indexed for withdrawals
                                if current_year >= start_yr and (end_yr is None or current_year <= end_yr):
                                    year_withdrawals += amount
                            
                            # Accumulated interest
                            accumulated_interest += interest_accrued
                            total_withdrawals += year_withdrawals
                            
                            yearly_data.append({
                                "Year": year + 1,
                                "Beginning Balance": f"${start_value:,.2f}",
                                "Interest Rate": f"{interest_rate:.2%}",
                                "Interest Accrued": f"${interest_accrued:,.2f}",
                                "Withdrawals": f"${year_withdrawals:,.2f}",
                                "Accumulated Interest": f"${accumulated_interest:,.2f}",
                                "Ending Balance": f"${end_value:,.2f}"
                            })
                        
                        # Display the table
                        st.table(pd.DataFrame(yearly_data))
                        
                        # Show summary statistics
                        st.metric(label="Total Interest Accrued", value=f"${accumulated_interest:,.2f}")
                        if total_withdrawals > 0:
                            st.metric(label="Total Withdrawals", value=f"${total_withdrawals:,.2f}")
                        
                        st.markdown("""
                        **Table Explanation:**
                        - This table shows yearly details for the 10th percentile simulation path
                        - Beginning Balance: Account value at start of year
                        - Interest Rate: The random rate generated for this year
                        - Interest Accrued: Interest earned during the year
                        - Withdrawals: Amount withdrawn during the year
                        - Accumulated Interest: Total interest earned since start
                        - Ending Balance: Account value at end of year after interest and withdrawals
                        """)
                else:
                    st.warning("No surviving paths to analyze. Try adjusting your withdrawal plan or investment parameters.")
            
            with tab3:
                st.subheader("Yearly Detail for Depleted Portfolio Path")
                
                # Find the first depleted path
                depleted_indices = [i for i, path in enumerate(paths) if path[-1] == 0]
                if depleted_indices:
                    # Use the first depleted path for analysis
                    depleted_path_idx = depleted_indices[0]
                    depleted_path = paths[depleted_path_idx]
                    depleted_rates = rates_by_path[depleted_path_idx]
                    
                    # Create the yearly detail table
                    yearly_data = []
                    accumulated_interest = 0
                    total_withdrawals = 0
                    
                    for year in range(years):
                        start_value = depleted_path[year]
                        end_value = depleted_path[year + 1]
                        interest_rate = depleted_rates[year]
                        
                        # Calculate interest accrued before withdrawals
                        interest_accrued = start_value * interest_rate
                        
                        # Calculate withdrawals for this year
                        year_withdrawals = 0
                        for start_yr, end_yr, amount in withdrawal_steps:
                            current_year = year + 1  # 1-indexed for withdrawals
                            if current_year >= start_yr and (end_yr is None or current_year <= end_yr):
                                year_withdrawals += amount
                        
                        # Accumulated interest
                        accumulated_interest += interest_accrued
                        total_withdrawals += year_withdrawals
                        
                        yearly_data.append({
                            "Year": year + 1,
                            "Beginning Balance": f"${start_value:,.2f}",
                            "Interest Rate": f"{interest_rate:.2%}",
                            "Interest Accrued": f"${interest_accrued:,.2f}",
                            "Withdrawals": f"${year_withdrawals:,.2f}",
                            "Accumulated Interest": f"${accumulated_interest:,.2f}",
                            "Ending Balance": f"${end_value:,.2f}"
                        })
                        
                        # Stop if the portfolio is depleted
                        if end_value == 0:
                            break
                    
                    # Display the table
                    st.table(pd.DataFrame(yearly_data))
                    
                    # Show summary statistics
                    st.metric(label="Total Interest Accrued", value=f"${accumulated_interest:,.2f}")
                    if total_withdrawals > 0:
                        st.metric(label="Total Withdrawals", value=f"${total_withdrawals:,.2f}")
                    
                    st.markdown("""
                    **Table Explanation:**
                    - This table shows yearly details for the first depleted simulation path
                    - Beginning Balance: Account value at start of year
                    - Interest Rate: The random rate generated for this year
                    - Interest Accrued: Interest earned during the year
                    - Withdrawals: Amount withdrawn during the year
                    - Accumulated Interest: Total interest earned since start
                    - Ending Balance: Account value at end of year after interest and withdrawals
                    """)
                else:
                    st.warning("No depleted paths found. Try adjusting your withdrawal plan or investment parameters.")
            
            with tab3:
                st.subheader("Yearly Detail for Worst Depleted Scenario")
                
                # Find the worst depleted path (fastest depletion)
                depleted_indices = [i for i, path in enumerate(paths) if path[-1] == 0]
                if depleted_indices:
                    # Identify the path that depleted the fastest
                    depletion_years = [
                        next((year for year, value in enumerate(paths[idx]) if value == 0), years)
                        for idx in depleted_indices
                    ]
                    worst_depletion_idx = depleted_indices[np.argmin(depletion_years)]
                    worst_depletion_path = paths[worst_depletion_idx]
                    worst_depletion_rates = rates_by_path[worst_depletion_idx]
                    
                    # Create the yearly detail table
                    yearly_data = []
                    accumulated_interest = 0
                    total_withdrawals = 0
                    
                    for year in range(years):
                        start_value = worst_depletion_path[year]
                        end_value = worst_depletion_path[year + 1]
                        interest_rate = worst_depletion_rates[year]
                        
                        # Calculate interest accrued before withdrawals
                        interest_accrued = start_value * interest_rate
                        
                        # Calculate withdrawals for this year
                        year_withdrawals = 0
                        for start_yr, end_yr, amount in withdrawal_steps:
                            current_year = year + 1  # 1-indexed for withdrawals
                            if current_year >= start_yr and (end_yr is None or current_year <= end_yr):
                                year_withdrawals += amount
                        
                        # Accumulated interest
                        accumulated_interest += interest_accrued
                        total_withdrawals += year_withdrawals
                        
                        yearly_data.append({
                            "Year": year + 1,
                            "Beginning Balance": f"${start_value:,.2f}",
                            "Interest Rate": f"{interest_rate:.2%}",
                            "Interest Accrued": f"${interest_accrued:,.2f}",
                            "Withdrawals": f"${year_withdrawals:,.2f}",
                            "Accumulated Interest": f"${accumulated_interest:,.2f}",
                            "Ending Balance": f"${end_value:,.2f}"
                        })
                        
                        # Stop if the portfolio is depleted
                        if end_value == 0:
                            break
                    
                    # Display the table
                    st.table(pd.DataFrame(yearly_data))
                    
                    # Show summary statistics
                    st.metric(label="Total Interest Accrued", value=f"${accumulated_interest:,.2f}")
                    if total_withdrawals > 0:
                        st.metric(label="Total Withdrawals", value=f"${total_withdrawals:,.2f}")
                    
                    st.markdown("""
                    **Table Explanation:**
                    - This table shows yearly details for the worst depleted simulation path
                    - Beginning Balance: Account value at start of year
                    - Interest Rate: The random rate generated for this year
                    - Interest Accrued: Interest earned during the year
                    - Withdrawals: Amount withdrawn during the year
                    - Accumulated Interest: Total interest earned since start
                    - Ending Balance: Account value at end of year after interest and withdrawals
                    """)
                else:
                    st.warning("No depleted paths found. Try adjusting your withdrawal plan or investment parameters.")
            
            # Detailed withdrawal plan
            if withdrawal_steps:
                st.subheader("Detailed Withdrawal Plan")
                
                # Create a dataframe for the withdrawal plan
                plan_data = []
                for i, (start_year, end_year, amount) in enumerate(sorted(withdrawal_steps, key=lambda x: x[0])):
                    end_year_display = end_year if end_year is not None else "Until end"
                    years_active = (end_year if end_year is not None else years) - start_year
                    total = amount * years_active
                    
                    plan_data.append({
                        "Step": i+1,
                        "Start Year": start_year,
                        "End Year": end_year_display,
                        "Annual Amount": f"${amount:,.2f}",
                        "Years Active": years_active,
                        "Total Withdrawn": f"${total:,.2f}"
                    })
                
                st.table(pd.DataFrame(plan_data))
    
    # Add some tips for users at the bottom
    with st.expander("Tips for Using This Simulator"):
        st.markdown("""
        - **Investment Parameters**: Set these in the sidebar. These define your initial investment, time horizon, and expected market behavior.
        - **Withdrawal Plan**: Use the table to specify when you'll withdraw money and how much. You can add multiple withdrawal steps.
        - **Start Year**: The year when the withdrawal begins (year 1 is the first year).
        - **End Year**: Optional - when the withdrawal ends. Leave blank to continue until the end of the simulation.
        - **Simulation Analysis**: After running the simulation, examine the success rate and distribution of outcomes to evaluate your plan.
        - **Sustainability**: A success rate of 95% or higher generally indicates a sustainable withdrawal plan.
        - **Yearly Detail Table**: The new table shows year-by-year performance for the median path, helping you understand how your investment might grow over time.
        """)
    
    # Footer
    st.markdown("---")
    st.caption("Monte Carlo Compound Interest Simulator - For educational purposes only. Not financial advice.")

if __name__ == "__main__":
    main()