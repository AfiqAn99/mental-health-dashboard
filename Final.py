import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
df = pd.read_csv('cleaned_mental_health_tech_survey.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sidebar Header
st.sidebar.title("🔧 Dashboard Filters")

# Collapsible filter sections
with st.sidebar.expander("📍 Country Filter", expanded=True):
    selected_countries = st.multiselect(
        "Select Country (Searchable)", options=df['country'].unique(),
        help="Filter by country. Select multiple options using the search box."
    )
    if not selected_countries:
        filtered_countries = df['country']
    else:
        filtered_countries = selected_countries

with st.sidebar.expander("👫 Gender Filter", expanded=False):
    selected_genders = st.multiselect(
        "Select Gender", options=df['gender'].unique(),
        help="Select genders for filtering. Use the dropdown to search."
    )
    if not selected_genders:
        filtered_genders = df['gender']
    else:
        filtered_genders = selected_genders

with st.sidebar.expander("📅 Age Filter", expanded=False):
    age_filter = st.slider(
        "Select Age Range", int(df['age'].min()), int(df['age'].max()), (20, 50),
        help="Drag the slider to filter data by the age range of participants."
    )

# Apply filters without hiding options
filtered_data = df[
    (df['country'].isin(filtered_countries)) &
    (df['gender'].isin(filtered_genders)) &
    (df['age'].between(*age_filter))
]

# Sidebar for Feature Selection
st.sidebar.header("📊 Advanced Feature Selection")
selected_kpi = st.sidebar.selectbox("Select KPI Feature", options=['age', 'treatment', 'family_history'], help="Select the KPI to display.")
selected_chart_feature = st.sidebar.selectbox("Select Feature for Visualization", options=['age', 'treatment', 'family_history', 'gender'], help="Select a feature to visualize.")

# Dashboard Title
st.markdown("<h1 style='color: #2C3E50;'>📊 Tech Mental Health Survey</h1>", unsafe_allow_html=True)

# Create Tabs
tabs = st.tabs(["Overview", "KPIs", "Visualizations", "Trend Prediction", "Data Table"])

# ------------------ Tab 1: Overview ------------------
with tabs[0]:
    st.markdown("<h2 style='color: #2E86C1;'>🌍 Quick Overview</h2>", unsafe_allow_html=True)
    st.write("Summary of key insights based on the filtered data.")

    # World map heatmap of entries by country
    st.subheader("🗺️ Heatmap of Entries by Country")
    country_counts = filtered_data['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'entries']

    fig_map = px.choropleth(
        country_counts, locations='country', locationmode='country names',
        color='entries', title='Number of Entries by Country',
        color_continuous_scale='Blues', 
        labels={'entries': 'Number of Entries'}
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Family history distribution by country
    st.subheader("🌏 Family History Distribution by Country (Top 10)")
    family_history_by_country = (
        filtered_data.groupby('country')['family_history']
        .value_counts()
        .unstack(fill_value=0)
        .sort_values(by='Yes', ascending=False)
        .head(10)
    )
    fig_country = px.bar(family_history_by_country, barmode='group', title='Family History Distribution by Country')
    st.plotly_chart(fig_country, use_container_width=True)

# ------------------ Tab 2: KPIs ------------------
with tabs[1]:
    st.markdown("<h2 style='color: #2E86C1;'>📈 Key Performance Indicators</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Entries", len(filtered_data))
    col2.metric(f"{selected_kpi.capitalize()} Average", round(filtered_data[selected_kpi].mean(), 2))
    col3.metric("Currently Receiving Treatment", f"{(filtered_data['treatment'] == 'Yes').mean() * 100:.1f}%")

# ------------------ Tab 3: Visualizations ------------------
with tabs[2]:
    st.markdown("<h2 style='color: #2E86C1;'>📊 Custom Visualizations</h2>", unsafe_allow_html=True)

    st.subheader(f"📅 Distribution of {selected_chart_feature.capitalize()}")
    fig = px.histogram(filtered_data, x=selected_chart_feature, color='treatment', title=f"Distribution of {selected_chart_feature.capitalize()}", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    # Optional scatter plot for feature comparison
    st.subheader("🔍 Compare Two Features")
    feature_x = st.selectbox("Select X-axis", options=df.columns, index=0)
    feature_y = st.selectbox("Select Y-axis", options=df.columns, index=1)
    scatter_fig = px.scatter(filtered_data, x=feature_x, y=feature_y, color='treatment', title=f"{feature_x} vs {feature_y}")
    st.plotly_chart(scatter_fig, use_container_width=True)

# ------------------ Tab 4: Trend Prediction ------------------
with tabs[3]:
    st.markdown("<h2 style='color: #2E86C1;'>📉 Trend Prediction: Treatment Rates</h2>", unsafe_allow_html=True)

    # Prepare data for linear regression
    time_data = filtered_data.set_index('timestamp').resample('M')['treatment'].count().reset_index()
    time_data['month_index'] = np.arange(len(time_data))

    # Linear regression
    X = time_data[['month_index']]
    y = time_data['treatment']
    model = LinearRegression().fit(X, y)
    time_data['predicted_treatment'] = model.predict(X)

    # Visualization of actual vs. predicted treatment
    fig_trend = px.line(time_data, x='timestamp', y=['treatment', 'predicted_treatment'], 
                        labels={'value': 'Number of Treatments', 'variable': 'Actual vs. Predicted'},
                        title='Actual vs. Predicted Monthly Treatment Rates')
    st.plotly_chart(fig_trend, use_container_width=True)

# ------------------ Tab 5: Data Table ------------------
with tabs[4]:
    st.markdown("<h2 style='color: #2E86C1;'>🔍 Explore Filtered Data</h2>", unsafe_allow_html=True)
    st.write("View and explore the data interactively.")
    st.dataframe(filtered_data, height=600)

# --- CSS for CoreUI Theme ---
st.markdown("""
    <style>
        body {
            background-color: #F3F4F6;
            color: #2C3E50;
            font-family: "Arial, sans-serif";
        }
        .stTabs [role="tablist"] { justify-content: space-between; }
        .stTabs [role="tab"] { font-size: 16px; padding: 10px 20px; border-radius: 10px; background-color: #E8EAF6; }
        .stTabs [role="tab"][aria-selected="true"] { background-color: #2C3E50; color: white; font-weight: bold; }
        .stMetric { background-color: #E3F2FD; padding: 15px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)