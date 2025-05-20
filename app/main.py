import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Solar Data Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Load the datasets
@st.cache_data
def load_data():
    df_ben = pd.read_csv("data/benin-malanville-clean.csv")
    df_sier = pd.read_csv("data/sierraleone-bumbuna-clean.csv")
    df_tog = pd.read_csv("data/togo-dapaong-clean.csv")
    
    # Add country column to each dataframe
    df_ben['Country'] = 'Benin'
    df_sier['Country'] = 'Sierra Leone'
    df_tog['Country'] = 'Togo'
    
    # Convert timestamp columns to datetime
    for df in [df_ben, df_sier, df_tog]:
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Date'] = df['Timestamp'].dt.date
            df['Hour'] = df['Timestamp'].dt.hour
            df['Month'] = df['Timestamp'].dt.month
    
    return df_ben, df_sier, df_tog

# Load data
df_ben, df_sier, df_tog = load_data()

# Combine all data
df_all = pd.concat([df_ben, df_sier, df_tog], ignore_index=True)

# Title and description
st.title("🌞 Solar Data Analysis Dashboard")
st.markdown("""
    This dashboard provides interactive visualizations and analysis of solar data from three locations:
    Benin (Malanville), Sierra Leone (Bumbuna), and Togo (Dapaong).
""")

# Sidebar
st.sidebar.header("📊 Dashboard Controls")

# Country selection
selected_countries = st.sidebar.multiselect(
    "Select countries to analyze",
    ["Benin", "Sierra Leone", "Togo"],
    default=["Benin", "Sierra Leone", "Togo"]
)

# Date range selection
if 'Timestamp' in df_all.columns:
    min_date = df_all['Date'].min()
    max_date = df_all['Date'].max()
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

# Metric selection
metric = st.sidebar.selectbox(
    "Select metric to analyze",
    ["GHI", "DNI", "DHI", "Tamb", "RH", "WS"],
    index=0
)

# Filter data based on selection
df_filtered = df_all[df_all['Country'].isin(selected_countries)]
if len(date_range) == 2:
    df_filtered = df_filtered[
        (df_filtered['Date'] >= date_range[0]) &
        (df_filtered['Date'] <= date_range[1])
    ]

# Main content
# Create tabs for different analysis sections
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview", 
    "🌡️ Temperature Analysis", 
    "💨 Wind Analysis",
    "📊 Statistical Analysis"
])

with tab1:
    # Overview section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{metric} Distribution by Country")
        # Interactive boxplot using plotly
        fig = px.box(
            df_filtered,
            x='Country',
            y=metric,
            color='Country',
            title=f'Distribution of {metric} by Country'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Regions")
        # Calculate statistics
        top_regions = df_filtered.groupby('Country')[metric].agg(['mean', 'std']).round(2)
        top_regions.columns = ['Average', 'Standard Deviation']
        top_regions = top_regions.sort_values('Average', ascending=False)
        
        # Display as a styled table
        st.dataframe(
            top_regions.style.background_gradient(cmap='RdYlGn'),
            use_container_width=True
        )
    
    # Time series analysis
    st.subheader("Time Series Analysis")
    # Aggregate data by date and country
    daily_avg = df_filtered.groupby(['Date', 'Country'])[metric].mean().reset_index()
    
    # Create interactive time series plot
    fig = px.line(
        daily_avg,
        x='Date',
        y=metric,
        color='Country',
        title=f'Daily Average {metric} Over Time'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Temperature Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature vs GHI")
        fig = px.scatter(
            df_filtered,
            x='GHI',
            y='Tamb',
            color='Country',
            title='Temperature vs GHI by Country',
            labels={'GHI': 'GHI (W/m²)', 'Tamb': 'Ambient Temperature (°C)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Temperature Distribution")
        fig = px.histogram(
            df_filtered,
            x='Tamb',
            color='Country',
            title='Temperature Distribution by Country',
            labels={'Tamb': 'Ambient Temperature (°C)'},
            marginal='box'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Wind Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Wind Speed vs GHI")
        fig = px.scatter(
            df_filtered,
            x='WS',
            y='GHI',
            color='Country',
            title='Wind Speed vs GHI by Country',
            labels={'WS': 'Wind Speed (m/s)', 'GHI': 'GHI (W/m²)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Wind Direction Analysis")
        # Create wind rose plot
        fig = go.Figure()
        for country in selected_countries:
            df_country = df_filtered[df_filtered['Country'] == country]
            fig.add_trace(go.Histogram(
                x=df_country['WD'],
                name=country,
                opacity=0.7
            ))
        fig.update_layout(
            title='Wind Direction Distribution by Country',
            xaxis_title='Wind Direction (degrees)',
            yaxis_title='Frequency',
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Statistical Analysis
    st.subheader("Detailed Statistics")
    
    # Calculate comprehensive statistics
    stats = df_filtered.groupby('Country').agg({
        'GHI': ['mean', 'std', 'min', 'max', 'median'],
        'Tamb': ['mean', 'std', 'min', 'max'],
        'RH': ['mean', 'std', 'min', 'max'],
        'WS': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    # Display statistics
    st.dataframe(stats, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    corr_matrix = df_filtered[['GHI', 'Tamb', 'RH', 'WS']].corr()
    fig = px.imshow(
        corr_matrix,
        title='Correlation Matrix',
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Solar Data Analysis Dashboard | Created with Streamlit</p>
    </div>
""", unsafe_allow_html=True)