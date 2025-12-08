import streamlit as st
import pandas as pd
import plotly.express as px
import os
import warnings
warnings.filterwarnings("ignore")

os.chdir(r'D:\OneDrive - Northeastern University\Jupyter Notebook\Data-Science-Mini-Projects\2025_10_09_Monthly_Expenses_Analysis')
# Set page configuration
st.set_page_config(
    page_title="Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    file_path = r'.\data\Expenses_2025_analysis_data.xlsx'
    df = pd.read_excel(file_path, sheet_name='spending25')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  
    df['Month'] = df['Date'].dt.to_period('M')
    df['YearMonth'] = df['Date'].dt.strftime('%b %Y')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['Categories'] = df['Categories'].str.lower().str.capitalize()
    return df

# Initialize data
df = load_data()

# Sidebar navigation
st.sidebar.title("Expense Tracker")
page = st.sidebar.radio(
    "Select Page",
    ["All Categories Pivot", "Rent/Utilities & Misc", "Car Expenses", "Gudiya Expenses", "Gym Expenses"]
)
columns_order = ['Rent', 'Wifi/phone', 'Electricity', 'Gas',  'Insurance', 'Groceries', 'Restaurant', 'Travel', 'Petrol',  'Gym', 'Misc', 'Other_misc']

# PAGE 1: ALL CATEGORIES PIVOT
if page == "All Categories Pivot":
    st.title("📈 All Categories Pivot - Month by Month")
    st.write("A breif Overview of your spending across all expense categories for each month")
    
    # Create pivot table for all months
    pivot_data = df.groupby(['YearMonth', 'Categories'])[['My Net Cost', 'Original Cost']].sum().reset_index()
    pivot_table_net = pivot_data.pivot(index='YearMonth', columns='Categories', values='My Net Cost').fillna(0)
    pivot_table_net = pivot_table_net.reindex(columns = columns_order).fillna(0)
    pivot_table_original = pivot_data.pivot(index='YearMonth', columns='Categories', values='Original Cost').fillna(0)
    pivot_table_original = pivot_table_original.reindex(columns = columns_order).fillna(0)
    
    # Sort by date
    pivot_table_net = pivot_table_net.sort_index()
    pivot_table_original = pivot_table_original.sort_index()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Spending (Net)", f"${df['My Net Cost'].sum():.2f}")
    with col2:
        st.metric("Total Spending (Original)", f"${df['Original Cost'].sum():.2f}")
    with col3:
        st.metric("Number of Categories", f"{len(pivot_table_net.columns)}")
    
    # Create bar chart showing My Net Cost vs Original Cost
    st.subheader("📊 Monthly Comparison Chart")
    monthly_totals = df.groupby('YearMonth')[['My Net Cost', 'Original Cost']].sum().round(2).reset_index()
    monthly_totals = monthly_totals.sort_values('YearMonth')
    
    # Create comparison chart
    col1, col2 = st.columns(2)
    with col1:
        fig_net = px.bar(
            monthly_totals,
            x='YearMonth',
            y='My Net Cost',
            title='Total My Net Cost by Month',
            labels={'My Net Cost': 'Amount ($)', 'YearMonth': 'Month'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_net.update_layout(xaxis_tickangle=-45,  showlegend=False)  # hovermode='x unified'
        st.plotly_chart(fig_net, use_container_width=True)
    
    with col2:
        fig_original = px.bar(
            monthly_totals,
            x='YearMonth',
            y='Original Cost',
            title='Total Original Cost by Month',
            labels={'Original Cost': 'Amount ($)', 'YearMonth': 'Month'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig_original.update_layout(xaxis_tickangle=-45, showlegend=False) # hovermode='x unified'
        st.plotly_chart(fig_original, use_container_width=True)
    
    # Month filter for pivot table
    st.subheader("📋 Pivot Table by Month")
    available_months = monthly_totals['YearMonth'].unique().tolist()
    month_options = ["All"] + available_months
    
    selected_pivot_month = st.selectbox(
        "Select Month",
        options=month_options,
        key="pivot_month_filter"
    )
    
    # Filter pivot table based on selected month
    if selected_pivot_month == "All":
        display_pivot_net = pivot_table_net
        display_pivot_original = pivot_table_original
    else:
        display_pivot_net = pivot_table_net.loc[[selected_pivot_month]] if selected_pivot_month in pivot_table_net.index else pd.DataFrame()
        display_pivot_original = pivot_table_original.loc[[selected_pivot_month]] if selected_pivot_month in pivot_table_original.index else pd.DataFrame()
    
    # Display pivot tables
    st.write("**My Net Cost by Category**")
    st.dataframe(display_pivot_net, use_container_width=True)
    
    st.write("**Original Cost by Category**")
    st.dataframe(display_pivot_original, use_container_width=True)
    
    # Download button
    csv_net = pivot_table_net.to_csv()
    st.download_button(
        label="Download My Net Cost Pivot Table as CSV",
        data=csv_net,
        file_name="expense_pivot_net.csv",
        mime="text/csv"
    )
    
    csv_original = pivot_table_original.to_csv()
    st.download_button(
        label="Download Original Cost Pivot Table as CSV",
        data=csv_original,
        file_name="expense_pivot_original.csv",
        mime="text/csv"
    )

# PAGE 2: RENT/UTILITIES & MISC EXPENSES
elif page == "Rent/Utilities & Misc":
    st.title("🏠 Rent/Utilities & Miscellaneous Expenses")
    st.write("Track your essential expenses and miscellaneous spending month by month")
    
    # Define categories for Rent + Utilities
    rent_utilities_keywords = ['rent', 'electricity', 'gas', 'insurance', 'wifi', 'phone', 'internet', 'utility', 'utilities']
    
    # Define categories for Misc
    misc_keywords = ['groceries', 'misc', 'petrol', 'restaurant', 'travel', 'dining', 'food', 'shopping']
    
    # Filter data for Rent + Utilities
    rent_utilities_data = df[
        df['Expense Title'].str.lower().str.contains('|'.join(rent_utilities_keywords), na=False, regex=True) |
        df['Description'].str.lower().str.contains('|'.join(rent_utilities_keywords), na=False, regex=True)
    ].copy()
    
    # Filter data for Misc
    misc_data = df[
        df['Expense Title'].str.lower().str.contains('|'.join(misc_keywords), na=False, regex=True) |
        df['Description'].str.lower().str.contains('|'.join(misc_keywords), na=False, regex=True)
    ].copy()
    
    # Monthly aggregation for Rent + Utilities
    rent_utilities_monthly = rent_utilities_data.groupby('YearMonth')['My Net Cost'].sum().sort_index().reset_index()
    rent_utilities_monthly = rent_utilities_monthly.sort_values('YearMonth')
    
    # Monthly aggregation for Misc
    misc_monthly = misc_data.groupby('YearMonth')['My Net Cost'].sum().sort_index().reset_index()
    misc_monthly = misc_monthly.sort_values('YearMonth')
    
    # Rent + Utilities Chart
    st.subheader("🏠 Rent + Utilities Expenses")
    if len(rent_utilities_monthly) > 0:
        # Create comparison with Original Cost
        rent_utilities_original = rent_utilities_data.groupby('YearMonth')['Original Cost'].sum().sort_index().reset_index()
        rent_util_comparison = pd.DataFrame({
            'YearMonth': rent_utilities_monthly['YearMonth'],
            'My Net Cost': rent_utilities_monthly['My Net Cost'],
            'Original Cost': rent_utilities_original['Original Cost']
        })
        rent_util_melted = rent_util_comparison.melt(id_vars=['YearMonth'], var_name='Cost Type', value_name='Amount')
        
        fig_rent_util = px.bar(
            rent_util_melted,
            x='YearMonth',
            y='Amount',
            color='Cost Type',
            title='Rent + Utilities (Electricity, Gas, Insurance, Rent, Wifi/Phone): My Net Cost vs Original Cost by Month',
            labels={'Amount': 'Amount Spent ($)', 'YearMonth': 'Month', 'Cost Type': 'Cost Type'},
            barmode='group',
            color_discrete_map={'My Net Cost': '#ff7f0e', 'Original Cost': '#17becf'}
        )
        fig_rent_util.update_layout(xaxis_tickangle=-45, hovermode='x unified')
        st.plotly_chart(fig_rent_util, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rent + Utilities (Net)", f"${rent_utilities_monthly['My Net Cost'].sum():.2f}")
        with col2:
            st.metric("Total Rent + Utilities (Original)", f"${rent_utilities_original['Original Cost'].sum():.2f}")
        
        # Show breakdown by category
        with st.expander("View Rent + Utilities Breakdown"):
            rent_util_breakdown = rent_utilities_data.groupby('Expense Title')[['My Net Cost', 'Original Cost']].sum().sort_values('My Net Cost', ascending=False)
            st.bar_chart(rent_util_breakdown)
    else:
        st.info("No rent/utilities expenses found")
    
    # Misc Expenses Chart
    st.subheader("🛒 Miscellaneous Expenses")
    if len(misc_monthly) > 0:
        # Create comparison with Original Cost
        misc_original = misc_data.groupby('YearMonth')['Original Cost'].sum().sort_index().reset_index()
        misc_comparison = pd.DataFrame({
            'YearMonth': misc_monthly['YearMonth'],
            'My Net Cost': misc_monthly['My Net Cost'],
            'Original Cost': misc_original['Original Cost']
        })
        misc_melted = misc_comparison.melt(id_vars=['YearMonth'], var_name='Cost Type', value_name='Amount')
        
        fig_misc = px.bar(
            misc_melted,
            x='YearMonth',
            y='Amount',
            color='Cost Type',
            title='Miscellaneous (Groceries, Petrol, Restaurant, Travel, etc.): My Net Cost vs Original Cost by Month',
            labels={'Amount': 'Amount Spent ($)', 'YearMonth': 'Month', 'Cost Type': 'Cost Type'},
            barmode='group',
            color_discrete_map={'My Net Cost': '#9467bd', 'Original Cost': '#bcbd22'}
        )
        fig_misc.update_layout(xaxis_tickangle=-45, hovermode='x unified')
        st.plotly_chart(fig_misc, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Misc Expenses (Net)", f"${misc_monthly['My Net Cost'].sum():.2f}")
        with col2:
            st.metric("Total Misc Expenses (Original)", f"${misc_original['Original Cost'].sum():.2f}")
        
        # Show breakdown by category
        with st.expander("View Misc Expenses Breakdown"):
            misc_breakdown = misc_data.groupby('Expense Title')[['My Net Cost', 'Original Cost']].sum().sort_values('My Net Cost', ascending=False)
            st.bar_chart(misc_breakdown)
    else:
        st.info("No miscellaneous expenses found")
    
    # Combined comparison
    st.subheader("📊 Comparison: Rent + Utilities vs Misc")
    
    # Merge the two series for comparison
    comparison_df = pd.DataFrame({
        'Rent + Utilities': rent_utilities_monthly.set_index('YearMonth')['My Net Cost'],
        'Miscellaneous': misc_monthly.set_index('YearMonth')['My Net Cost']
    }).fillna(0).reset_index()
    
    fig_comparison = px.bar(
        comparison_df,
        x='YearMonth',
        y=['Rent + Utilities', 'Miscellaneous'],
        title='Rent + Utilities vs Miscellaneous Expenses by Month',
        labels={'value': 'Amount Spent ($)', 'YearMonth': 'Month', 'variable': 'Category'},
        barmode='group'
    )
    fig_comparison.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig_comparison, use_container_width=True)

# PAGE 3: CAR EXPENSES 
elif page == "Car Expenses":
    st.title("🚗 Car Expenses")
    
    car_data = df[df['Description'].str.lower().str.contains('car', na=False)].copy()
    
    # Get all monthly data for charts (no filter applied to charts)
    car_all_monthly = car_data.groupby('YearMonth')[['My Net Cost', 'Original Cost']].sum().sort_index().reset_index()
    car_all_monthly = car_all_monthly.sort_values('YearMonth')
    
    if len(car_data) > 0:
        # Display metrics for all data
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Paid by Me (All Months)", f"${car_data['My Net Cost'].sum():.2f}")
        with col2:
            st.metric("Total Original Cost (All Months)", f"${car_data['Original Cost'].sum():.2f}")
        
        # Two separate bar charts (showing all months - no filter)
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for Paid by me
            if len(car_all_monthly) > 0:
                fig_car_paid = px.bar(
                    car_all_monthly,
                    x='YearMonth',
                    y='My Net Cost',
                    title='Paid by Me (My Net Cost)',
                    labels={'My Net Cost': 'Amount ($)', 'YearMonth': 'Month'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig_car_paid.update_layout(xaxis_tickangle=-45, hovermode='x unified', showlegend=False)
                st.plotly_chart(fig_car_paid, use_container_width=True)
        
        with col2:
            # Bar chart for Original cost
            if len(car_all_monthly) > 0:
                fig_car_original = px.bar(
                    car_all_monthly,
                    x='YearMonth',
                    y='Original Cost',
                    title='Original Cost',
                    labels={'Original Cost': 'Amount ($)', 'YearMonth': 'Month'},
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_car_original.update_layout(xaxis_tickangle=-45, hovermode='x unified', showlegend=False)
                st.plotly_chart(fig_car_original, use_container_width=True)
        
        # Month filter for detailed table
        st.subheader("📋 Detailed Car Expenses")
        available_months = sorted(car_data['YearMonth'].unique().tolist())
        month_options = ["All"] + available_months
        
        selected_month = st.selectbox(
            "Select Month",
            options=month_options,
            key="car_month_filter"
        )
        
        # Filter data based on selected month for table only
        if selected_month == "All":
            filtered_car_data = car_data.copy()
        else:
            filtered_car_data = car_data[car_data['YearMonth'] == selected_month].copy()
        
        display_columns = ['Date', 'Expense Title', 'Description', 'My Net Cost', 'Original Cost']
        table_data = filtered_car_data[display_columns].sort_values('Date', ascending=True)
        st.dataframe(table_data, use_container_width=True, hide_index=True)
        
        # Download button
        csv = table_data.to_csv(index=False)
        st.download_button(
            label="Download Car Expenses as CSV",
            data=csv,
            file_name="car_expenses.csv",
            mime="text/csv"
        )
    else:
        st.info("No car expenses found")

# PAGE 4: GUDIYA EXPENSES
elif page == "Gudiya Expenses":
    st.title("👧 Gudiya Expenses")
    
    gudiya_data = df[df['Description'].str.lower().str.contains('gudiya', na=False)].copy()
    
    # Get all monthly data for charts (no filter applied to charts)
    gudiya_all_monthly = gudiya_data.groupby('YearMonth')[['My Net Cost', 'Original Cost']].sum().sort_index().reset_index()
    gudiya_all_monthly = gudiya_all_monthly.sort_values('YearMonth')
    
    if len(gudiya_data) > 0:
        # Display metrics for all data
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Paid by Me (All Months)", f"${gudiya_data['My Net Cost'].sum():.2f}")
        with col2:
            st.metric("Total Original Cost (All Months)", f"${gudiya_data['Original Cost'].sum():.2f}")
        
        # Two separate bar charts (showing all months - no filter)
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for Paid by me
            if len(gudiya_all_monthly) > 0:
                fig_gudiya_paid = px.bar(
                    gudiya_all_monthly,
                    x='YearMonth',
                    y='My Net Cost',
                    title='Paid by Me (My Net Cost)',
                    labels={'My Net Cost': 'Amount ($)', 'YearMonth': 'Month'},
                    color_discrete_sequence=['#9467bd']
                )
                fig_gudiya_paid.update_layout(xaxis_tickangle=-45, hovermode='x unified', showlegend=False)
                st.plotly_chart(fig_gudiya_paid, use_container_width=True)
        
        with col2:
            # Bar chart for Original cost
            if len(gudiya_all_monthly) > 0:
                fig_gudiya_original = px.bar(
                    gudiya_all_monthly,
                    x='YearMonth',
                    y='Original Cost',
                    title='Original Cost',
                    labels={'Original Cost': 'Amount ($)', 'YearMonth': 'Month'},
                    color_discrete_sequence=['#8c564b']
                )
                fig_gudiya_original.update_layout(xaxis_tickangle=-45, hovermode='x unified', showlegend=False)
                st.plotly_chart(fig_gudiya_original, use_container_width=True)
        
        # Month filter for detailed table
        st.subheader("📋 Detailed Gudiya Expenses")
        available_months = sorted(gudiya_data['YearMonth'].unique().tolist())
        month_options = ["All"] + available_months
        
        selected_month = st.selectbox(
            "Select Month",
            options=month_options,
            key="gudiya_month_filter"
        )
        
        # Filter data based on selected month for table only
        if selected_month == "All":
            filtered_gudiya_data = gudiya_data.copy()
        else:
            filtered_gudiya_data = gudiya_data[gudiya_data['YearMonth'] == selected_month].copy()
        
        display_columns = ['Date', 'Expense Title', 'Description', 'My Net Cost', 'Original Cost']
        table_data = filtered_gudiya_data[display_columns].sort_values('Date', ascending=True)
        st.dataframe(table_data, use_container_width=True, hide_index=True)
        
        # Download button
        csv = table_data.to_csv(index=False)
        st.download_button(
            label="Download Gudiya Expenses as CSV",
            data=csv,
            file_name="gudiya_expenses.csv",
            mime="text/csv"
        )
    else:
        st.info("No Gudiya expenses found")
    
# PAGE 5: GYM EXPENSES
elif page == "Gym Expenses":
    st.title("💪 Planet Fitness (Gym) Expenses")
    
    gym_data = df[(df['Description'].str.lower().str.contains('gym', na=False)) | 
                  (df['Expense Title'].str.lower().str.contains('gym|planet fitness', na=False, regex=True))].copy()
    
    # Get all monthly data for charts (no filter applied to charts)
    gym_all_monthly = gym_data.groupby('YearMonth')[['My Net Cost', 'Original Cost']].sum().sort_index().reset_index()
    gym_all_monthly = gym_all_monthly.sort_values('YearMonth')
    
    if len(gym_data) > 0:
        # Display metrics for all data
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Paid by Me (All Months)", f"${gym_data['My Net Cost'].sum():.2f}")
        with col2:
            st.metric("Total Original Cost (All Months)", f"${gym_data['Original Cost'].sum():.2f}")
        
        # Two separate bar charts (showing all months - no filter)
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for Paid by me
            if len(gym_all_monthly) > 0:
                fig_gym_paid = px.line(
                    gym_all_monthly,
                    x='YearMonth',
                    y='My Net Cost',
                    title='Paid by Me (My Net Cost)',
                    labels={'My Net Cost': 'Amount ($)', 'YearMonth': 'Month'},
                    color_discrete_sequence=['#2ca02c']
                )
                fig_gym_paid.update_layout(xaxis_tickangle=-45, hovermode='x unified', showlegend=False)
                st.plotly_chart(fig_gym_paid, use_container_width=True)
        
        with col2:
            # Bar chart for Original cost
            if len(gym_all_monthly) > 0:
                fig_gym_original = px.line(
                    gym_all_monthly,
                    x='YearMonth',
                    y='Original Cost',
                    title='Original Cost',
                    labels={'Original Cost': 'Amount ($)', 'YearMonth': 'Month'},
                    color_discrete_sequence=['#d62728']
                )
                fig_gym_original.update_layout(xaxis_tickangle=-45, hovermode='x unified', showlegend=False)
                st.plotly_chart(fig_gym_original, use_container_width=True)
        
        # Month filter for detailed table
        st.subheader("📋 Detailed Gym Expenses")
        available_months = sorted(gym_data['YearMonth'].unique().tolist())
        month_options = ["All"] + available_months
        
        selected_month = st.selectbox(
            "Select Month",
            options=month_options,
            key="gym_month_filter"
        )
        
        # Filter data based on selected month for table only
        if selected_month == "All":
            filtered_gym_data = gym_data.copy()
        else:
            filtered_gym_data = gym_data[gym_data['YearMonth'] == selected_month].copy()
        
        display_columns = ['Date', 'Expense Title', 'Description', 'My Net Cost', 'Original Cost']
        table_data = filtered_gym_data[display_columns].sort_values('Date', ascending=True)
        st.dataframe(table_data, use_container_width=True, hide_index=True)
        
        # Download button
        csv = table_data.to_csv(index=False)
        st.download_button(
            label="Download Gym Expenses as CSV",
            data=csv,
            file_name="gym_expenses.csv",
            mime="text/csv"
        )
    else:
        st.info("No gym expenses found")

