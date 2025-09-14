import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, linear_rainbow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Student Stress Analysis Dashboard",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .filter-section {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        padding: 10px 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 2px solid #444444 !important;
        font-size: 16px !important;
        padding: 12px 20px;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        border-color: #1f77b4 !important;
        color: #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background-color: #3d3d3d !important;
        border-color: #555555 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('processed_data.csv')
    df['high_stress'] = (df['stress_1to5'] >= 4).astype(int)
    return df

df = load_data()



# Sidebar with comprehensive filters
st.sidebar.title("Dashboard Controls")
if st.sidebar.button("Reset All Filters", use_container_width=True, type="primary"):
    # Clear all session state values for filters
    for key in st.session_state.keys():
        if key.endswith('_filter'):
            del st.session_state[key]
    st.rerun()
st.sidebar.markdown("---")

# Comprehensive filtering system
st.sidebar.header("Data Filters")

with st.sidebar.expander("Academic Filters", expanded=True):
    year_filter = st.multiselect(
        "Year of Study",
        options=sorted(df['year_of_study'].unique()),
        default=sorted(df['year_of_study'].unique())
    )
    
    courses_filter = st.slider(
        "Courses Enrolled",
        min_value=int(df['courses_enrolled'].min()),
        max_value=int(df['courses_enrolled'].max()),
        value=(int(df['courses_enrolled'].min()), int(df['courses_enrolled'].max()))
    )
    
    cgpa_filter = st.slider(
        "CGPA Range",
        min_value=float(df['cgpa'].min()),
        max_value=float(df['cgpa'].max()),
        value=(float(df['cgpa'].min()), float(df['cgpa'].max())),
        step=0.1
    )

with st.sidebar.expander("Demographic Filters", expanded=True):
    gender_filter = st.multiselect(
        "Gender",
        options=df['gender'].unique(),
        default=df['gender'].unique()
    )
    
    age_filter = st.slider(
        "Age Range",
        min_value=int(df['age'].min()),
        max_value=int(df['age'].max()),
        value=(int(df['age'].min()), int(df['age'].max()))
    )

with st.sidebar.expander("Time Commitment", expanded=True):
    study_hours_filter = st.slider(
        "Study Hours/Week",
        min_value=int(df['study_hours_per_week'].min()),
        max_value=int(df['study_hours_per_week'].max()),
        value=(int(df['study_hours_per_week'].min()), int(df['study_hours_per_week'].max()))
    )
    
    extracurricular_filter = st.multiselect(
        "Extracurricular",
        options=df['extracurricular'].unique(),
        default=df['extracurricular'].unique(),
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    
    job_filter = st.multiselect(
        "Job Status",
        options=df['job'].unique(),
        default=df['job'].unique(),
        format_func=lambda x: "Employed" if x == 1 else "Not Employed"
    )
    
    extracurricular_hours_filter = st.slider(
        "Extracurricular Hours",
        min_value=int(df['extracurricular_hours'].min()),
        max_value=int(df['extracurricular_hours'].max()),
        value=(int(df['extracurricular_hours'].min()), int(df['extracurricular_hours'].max()))
    )
    
    job_hours_filter = st.slider(
        "Job Hours/Week (if employed)",
        min_value=int(df['job_hours_per_week'].min()),
        max_value=int(df['job_hours_per_week'].max()),
        value=(int(df['job_hours_per_week'].min()), int(df['job_hours_per_week'].max()))
    )

with st.sidebar.expander("Wellbeing Filters", expanded=True):
    stress_filter = st.slider(
        "Stress Level (1-5)",
        min_value=1,
        max_value=5,
        value=(1, 5)
    )
    
    anxiety_filter = st.slider(
        "Anxiety Level (1-5)",
        min_value=1,
        max_value=5,
        value=(1, 5)
    )
    
    sleep_hours_filter = st.slider(
        "Sleep Hours",
        min_value=int(df['sleep_hours'].min()),
        max_value=int(df['sleep_hours'].max()),
        value=(int(df['sleep_hours'].min()), int(df['sleep_hours'].max()))
    )
    
    sleep_quality_filter = st.slider(
        "Sleep Quality (1-10)",
        min_value=1,
        max_value=10,
        value=(1, 10)
    )


filtered_df = df[
    (df['year_of_study'].isin(year_filter)) &
    (df['gender'].isin(gender_filter)) &
    (df['extracurricular'].isin(extracurricular_filter)) &
    (df['job'].isin(job_filter)) &
    (df['courses_enrolled'] >= courses_filter[0]) & (df['courses_enrolled'] <= courses_filter[1]) &
    (df['cgpa'] >= cgpa_filter[0]) & (df['cgpa'] <= cgpa_filter[1]) &
    (df['age'] >= age_filter[0]) & (df['age'] <= age_filter[1]) &
    (df['study_hours_per_week'] >= study_hours_filter[0]) & (df['study_hours_per_week'] <= study_hours_filter[1]) &
    (df['extracurricular_hours'] >= extracurricular_hours_filter[0]) & (df['extracurricular_hours'] <= extracurricular_hours_filter[1]) &
    (df['job_hours_per_week'] >= job_hours_filter[0]) & (df['job_hours_per_week'] <= job_hours_filter[1]) &
    (df['stress_1to5'] >= stress_filter[0]) & (df['stress_1to5'] <= stress_filter[1]) &
    (df['anxiety_1to5'] >= anxiety_filter[0]) & (df['anxiety_1to5'] <= anxiety_filter[1]) &
    (df['sleep_hours'] >= sleep_hours_filter[0]) & (df['sleep_hours'] <= sleep_hours_filter[1]) &
    (df['sleep_quality_1to10'] >= sleep_quality_filter[0]) & (df['sleep_quality_1to10'] <= sleep_quality_filter[1])
]

# Filter summary in sidebar
st.sidebar.markdown("---")
st.sidebar.header("Filter Summary")
st.sidebar.write(f"**Students shown:** {len(filtered_df)} of {len(df)}")
st.sidebar.write(f"**Years:** {', '.join(map(str, year_filter))}")
st.sidebar.write(f"**Gender:** {', '.join(gender_filter)}")
st.sidebar.write(f"**CGPA Range:** {cgpa_filter[0]:.1f} - {cgpa_filter[1]:.1f}")
st.sidebar.write(f"**Stress Level:** {stress_filter[0]} - {stress_filter[1]}")


# Main content
st.markdown('<h1 class="main-header">Workload and Mental Health Analysis</h1>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", 
    "Exploratory Data Analysis", 
    "Hypothesis Testing", 
    "Predictive Modeling", 
    "Raw Data"
])

with tab1:
    
    st.markdown("<h2 class='main-header'>Overview Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("Analyze student stress levels with dynamic filters and interactive visualizations.")
    # Overview metrics with improved layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(filtered_df), f"{len(filtered_df)/len(df)*100:.1f}% of total")
    with col2:
        avg_stress = filtered_df['stress_1to5'].mean()
        st.metric("Average Stress Level", f"{avg_stress:.2f}/5", 
                f"{(avg_stress - df['stress_1to5'].mean()):.2f} vs overall")
    with col3:
        avg_anxiety = filtered_df['anxiety_1to5'].mean()
        st.metric("Average Anxiety Level", f"{avg_anxiety:.2f}/5", 
                f"{(avg_anxiety - df['anxiety_1to5'].mean()):.2f} vs overall")
    with col4:
        high_stress_pct = (filtered_df['high_stress'].mean() * 100)
        st.metric("High Stress (%)", f"{high_stress_pct:.1f}%", 
                f"{(high_stress_pct - (df['high_stress'].mean() * 100)):.1f}% vs overall")

    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_cgpa = filtered_df['cgpa'].mean()
        st.metric("Average CGPA", f"{avg_cgpa:.2f}")
    with col2:
        avg_study_hours = filtered_df['study_hours_per_week'].mean()
        st.metric("Avg Study Hours", f"{avg_study_hours:.1f}")
    with col3:
        avg_sleep = filtered_df['sleep_hours'].mean()
        st.metric("Avg Sleep Hours", f"{avg_sleep:.1f}")
    with col4:
        employment_rate = filtered_df['job'].mean() * 100
        st.metric("Employment Rate", f"{employment_rate:.1f}%")
    st.markdown("---")
    # Data summary
    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numerical Variables:**")
        st.dataframe(filtered_df.describe(), use_container_width=True)
    
    with col2:
        st.write("**Categorical Variables:**")
        cat_vars = ['year_of_study', 'gender', 'extracurricular', 'job', 'high_stress']
        cat_summary = pd.DataFrame({
            'Variable': cat_vars,
            'Unique Values': [filtered_df[var].nunique() for var in cat_vars],
            'Most Frequent': [filtered_df[var].mode()[0] for var in cat_vars]
        })
        st.dataframe(cat_summary, use_container_width=True)

with tab2:
    st.markdown("<h2 class='section-header'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    
    # Create subtabs for different EDA sections
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "Distributions", 
        "Correlation Analysis", 
        "Scatterplot Matrix", 
        "Boxplot Analysis"
    ])
    
    with eda_tab1:
        st.markdown("<h3>Variable Distributions</h3>", unsafe_allow_html=True)
        
        # Check if filtered data is not empty
        if len(filtered_df) == 0:
            st.warning("No data available with current filters. Please adjust your filter settings.")
        else:
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Year of study distribution
                try:
                    year_counts = filtered_df['year_of_study'].value_counts().sort_index()
                    fig = px.bar(x=year_counts.index, y=year_counts.values, 
                                title='Year of Study Distribution',
                                labels={'x': 'Year of Study', 'y': 'Frequency'})
                    fig.update_traces(marker_color='lightblue', marker_line_color='black', 
                                    marker_line_width=1, opacity=0.8)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Year of Study distribution: {str(e)}")
                
                # Age distribution
                try:
                    fig = px.histogram(filtered_df, x='age', nbins=15,
                                    title='Age Distribution',
                                    labels={'age': 'Age', 'count': 'Frequency'})
                    fig.update_traces(marker_color='lightblue', marker_line_color='black', 
                                    marker_line_width=1, opacity=0.7)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Age distribution: {str(e)}")
                
                # Study hours distribution
                try:
                    fig = px.histogram(filtered_df, x='study_hours_per_week', nbins=20,
                                    title='Study Hours per Week',
                                    labels={'study_hours_per_week': 'Hours', 'count': 'Frequency'})
                    fig.update_traces(marker_color='lightblue', marker_line_color='black', 
                                    marker_line_width=1, opacity=0.7)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Study Hours distribution: {str(e)}")
                
                # Extracurricular participation
                try:
                    extracurricular_counts = filtered_df['extracurricular'].value_counts()
                    fig = px.bar(x=['No', 'Yes'], y=extracurricular_counts.values, 
                                title='Extracurricular Participation',
                                labels={'x': 'Participates', 'y': 'Frequency'})
                    fig.update_traces(marker_color=['lightgray', 'lightpink'], marker_line_color='black', 
                                    marker_line_width=1, opacity=0.8)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Extracurricular distribution: {str(e)}")
                
                # Job status distribution
                try:
                    job_counts = filtered_df['job'].value_counts()
                    fig = px.bar(x=['No Job', 'Has Job'], y=job_counts.values, 
                                title='Job Status Distribution',
                                labels={'x': 'Employment Status', 'y': 'Frequency'})
                    fig.update_traces(marker_color=['lightgray', 'lightpink'], marker_line_color='black', 
                                    marker_line_width=1, opacity=0.8)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Job Status distribution: {str(e)}")
                
                # Stress level distribution
                try:
                    fig = px.histogram(filtered_df, x='stress_1to5', nbins=5,
                                    title='Stress Levels (1-5)',
                                    labels={'stress_1to5': 'Level', 'count': 'Frequency'})
                    fig.update_traces(marker_color='lightblue', marker_line_color='black', 
                                    marker_line_width=1, opacity=0.8)
                    fig.update_xaxes(tickvals=list(range(1, 6)))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Stress distribution: {str(e)}")
                
                # Sleep hours distribution
                try:
                    fig = px.histogram(filtered_df, x='sleep_hours', nbins=15,
                                    title='Sleep Hours',
                                    labels={'sleep_hours': 'Hours', 'count': 'Frequency'})
                    fig.update_traces(marker_color='lightblue', marker_line_color='black', 
                                    marker_line_width=1, opacity=0.7)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Sleep Hours distribution: {str(e)}")
                
                
            
            with col2:
                # Gender distribution
                try:
                    gender_counts = filtered_df['gender'].value_counts()
                    fig = px.bar(x=gender_counts.index, y=gender_counts.values, 
                                title='Gender Distribution',
                                labels={'x': 'Gender', 'y': 'Frequency'})
                    fig.update_traces(marker_color=['lightgray', 'lightpink'], marker_line_color='black', 
                                    marker_line_width=1, opacity=0.8)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Gender distribution: {str(e)}")
                
                # CGPA distribution
                try:
                    fig = px.histogram(filtered_df, x='cgpa', nbins=20, 
                                    title='CGPA Distribution',
                                    labels={'cgpa': 'CGPA', 'count': 'Frequency'})
                    fig.update_traces(marker_color='lightblue', marker_line_color='black', 
                                    marker_line_width=1, opacity=0.7)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying CGPA distribution: {str(e)}")
                
                # Courses enrolled distribution
                try:
                    fig = px.histogram(filtered_df, x='courses_enrolled', nbins=8,
                                    title='Courses Enrolled',
                                    labels={'courses_enrolled': 'Number of Courses', 'count': 'Frequency'})
                    fig.update_traces(marker_color='lightblue', marker_line_color='black', 
                                    marker_line_width=1, opacity=0.7)
                    fig.update_xaxes(tickvals=list(range(1, 9)))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Courses distribution: {str(e)}")
                
                # Extracurricular hours distribution
                try:
                    fig = px.histogram(filtered_df, x='extracurricular_hours', nbins=15,
                                    title='Extracurricular Hours',
                                    labels={'extracurricular_hours': 'Hours', 'count': 'Frequency'})
                    fig.update_traces(marker_color='lightblue', marker_line_color='black', 
                                    marker_line_width=1, opacity=0.7)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Extracurricular Hours distribution: {str(e)}")
                
                # Job hours distribution (only for employed)
                try:
                    employed_df = filtered_df[filtered_df['job_hours_per_week'] > 0]
                    if not employed_df.empty:
                        fig = px.histogram(employed_df, x='job_hours_per_week', nbins=15,
                                        title='Job Hours per Week (Employed Only)',
                                        labels={'job_hours_per_week': 'Hours', 'count': 'Frequency'})
                        fig.update_traces(marker_color='lightblue', marker_line_color='black', 
                                        marker_line_width=1, opacity=0.7)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No employed students in the filtered data.")
                except Exception as e:
                    st.error(f"Error displaying Job Hours distribution: {str(e)}")
                
                # Anxiety level distribution
                try:
                    fig = px.histogram(filtered_df, x='anxiety_1to5', nbins=5,
                                    title='Anxiety Levels (1-5)',
                                    labels={'anxiety_1to5': 'Level', 'count': 'Frequency'})
                    fig.update_traces(marker_color='lightblue', marker_line_color='black', 
                                    marker_line_width=1, opacity=0.8)
                    fig.update_xaxes(tickvals=list(range(1, 6)))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Anxiety distribution: {str(e)}")
                
                # Sleep quality distribution
                try:
                    fig = px.histogram(filtered_df, x='sleep_quality_1to10', nbins=10,
                                    title='Sleep Quality (1-10)',
                                    labels={'sleep_quality_1to10': 'Quality', 'count': 'Frequency'})
                    fig.update_traces(marker_color='lightblue', marker_line_color='black', 
                                    marker_line_width=1, opacity=0.8)
                    fig.update_xaxes(tickvals=list(range(1, 11)))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Sleep Quality distribution: {str(e)}")
    
    with eda_tab2:
        st.markdown("<h3>Correlation Analysis</h3>", unsafe_allow_html=True)
        
        if len(filtered_df) == 0:
            st.warning("No data available with current filters. Please adjust your filter settings.")
        else:
            # Correlation heatmap
            try:
                numeric_cols = ['year_of_study', 'cgpa', 'study_hours_per_week', 'courses_enrolled',
                                'extracurricular_hours', 'job_hours_per_week',
                                'stress_1to5', 'anxiety_1to5', 'sleep_hours', 'sleep_quality_1to10']
                
                correlation_matrix = filtered_df[numeric_cols].corr()
                
                fig = px.imshow(correlation_matrix, 
                            text_auto=True, 
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title='Correlation Matrix of Numerical Variables')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying correlation matrix: {str(e)}")
            
            # Correlation summary table
            st.markdown("<h4>Correlation Summary</h4>", unsafe_allow_html=True)
            
            try:
                relationships = []
                for i, (title, x_var, y_var) in enumerate([
                    ("Stress vs Anxiety", 'stress_1to5', 'anxiety_1to5'),
                    ("Sleep Hours vs Sleep Quality", 'sleep_hours', 'sleep_quality_1to10'),
                    ("Study Hours vs Extracurricular Hours", 'study_hours_per_week', 'extracurricular_hours'),
                    ("Anxiety vs Sleep Quality", 'anxiety_1to5', 'sleep_quality_1to10'),
                    ("Stress vs Sleep Quality", 'stress_1to5', 'sleep_quality_1to10'),
                    ("Courses Enrolled vs Sleep Hours", 'courses_enrolled', 'sleep_hours'),
                    ("Anxiety vs Sleep Hours", 'anxiety_1to5', 'sleep_hours'),
                    ("Job Hours vs Anxiety", 'job_hours_per_week', 'anxiety_1to5'),
                    ("Year of Study vs Stress", 'year_of_study', 'stress_1to5'),
                    ("Study Hours vs Stress", 'study_hours_per_week', 'stress_1to5'),
                    ("Study Hours vs CGPA", 'study_hours_per_week', 'cgpa'),
                    ("Courses Enrolled vs Year of Study", 'courses_enrolled', 'year_of_study')
                ]):
                    if title == "Job Hours vs Anxiety":
                        employed_df = filtered_df[filtered_df['job_hours_per_week'] > 0]
                        if employed_df.empty:
                            relationships.append((title, float('nan'), float('nan'), "No employed students"))
                            continue
                        x_data = employed_df[x_var]
                        y_data = employed_df[y_var]
                    else:
                        x_data = filtered_df[x_var]
                        y_data = filtered_df[y_var]

                    if len(x_data.dropna()) > 1 and len(y_data.dropna()) > 1:
                        # Calculate correlation
                        corr_value = x_data.corr(y_data)
                        r_squared = corr_value ** 2

                        # Calculate regression equation
                        clean_data = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()
                        z = np.polyfit(clean_data['x'], clean_data['y'], 1)
                        slope, intercept = z[0], z[1]
                        equation = f"y = {slope:.3f}x + {intercept:.3f}"

                        relationships.append((title, corr_value, r_squared, equation))
                    else:
                        relationships.append((title, float('nan'), float('nan'), "Insufficient data"))
                
                # Display correlation summary as a table
                corr_df = pd.DataFrame(relationships, columns=['Relationship', 'Correlation (r)', 'R-squared', 'Regression Equation'])
                st.dataframe(corr_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error calculating correlations: {str(e)}")
    
    with eda_tab3:
        st.markdown("<h3>Scatterplot Matrix with Regression Lines</h3>", unsafe_allow_html=True)
        
        if len(filtered_df) == 0:
            st.warning("No data available with current filters. Please adjust your filter settings.")
        else:
            # Create all scatterplots in a grid
            scatter_plots = [
                ("Stress vs Anxiety", 'stress_1to5', 'anxiety_1to5', None),
                ("Sleep Hours vs Sleep Quality", 'sleep_hours', 'sleep_quality_1to10', None),
                ("Study Hours vs Extracurricular Hours", 'study_hours_per_week', 'extracurricular_hours', None),
                ("Anxiety vs Sleep Quality", 'anxiety_1to5', 'sleep_quality_1to10', None),
                ("Stress vs Sleep Quality", 'stress_1to5', 'sleep_quality_1to10', None),
                ("Courses Enrolled vs Sleep Hours", 'courses_enrolled', 'sleep_hours', None),
                ("Anxiety vs Sleep Hours", 'anxiety_1to5', 'sleep_hours', None),
                ("Job Hours vs Anxiety", 'job_hours_per_week', 'anxiety_1to5', "Employed Only"),
                ("Year of Study vs Stress", 'year_of_study', 'stress_1to5', None),
                ("Study Hours vs Stress", 'study_hours_per_week', 'stress_1to5', None),
                ("Study Hours vs CGPA", 'study_hours_per_week', 'cgpa', None),
                ("Courses Enrolled vs Year of Study", 'courses_enrolled', 'year_of_study', None)
            ]
            
            # Display scatterplots in a grid
            for i in range(0, len(scatter_plots), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < len(scatter_plots):
                        title, x_var, y_var, note = scatter_plots[i]
                        try:
                            if note == "Employed Only":
                                plot_df = filtered_df[filtered_df['job_hours_per_week'] > 0]
                                if plot_df.empty:
                                    st.info(f"{title}: No employed students in filtered data")
                                else:
                                    fig = px.scatter(plot_df, x=x_var, y=y_var, 
                                                    title=f"{title} ({note})",
                                                    trendline="ols")
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                fig = px.scatter(filtered_df, x=x_var, y=y_var, 
                                                title=title,
                                                trendline="ols")
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying {title}: {str(e)}")
                
                with col2:
                    if i + 1 < len(scatter_plots):
                        title, x_var, y_var, note = scatter_plots[i + 1]
                        try:
                            if note == "Employed Only":
                                plot_df = filtered_df[filtered_df['job_hours_per_week'] > 0]
                                if plot_df.empty:
                                    st.info(f"{title}: No employed students in filtered data")
                                else:
                                    fig = px.scatter(plot_df, x=x_var, y=y_var, 
                                                    title=f"{title} ({note})",
                                                    trendline="ols")
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                fig = px.scatter(filtered_df, x=x_var, y=y_var, 
                                                title=title,
                                                trendline="ols")
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying {title}: {str(e)}")
    
    with eda_tab4:
        st.markdown("<h3>Boxplot Analysis</h3>", unsafe_allow_html=True)
        
        if len(filtered_df) == 0:
            st.warning("No data available with current filters. Please adjust your filter settings.")
        else:
            # Create all boxplots in a grid
            boxplot_configs = [
                ("CGPA by Year of Study", 'year_of_study', 'cgpa', None),
                ("Stress by Year of Study", 'year_of_study', 'stress_1to5', None),
                ("Stress by Job Status", 'job', 'stress_1to5', {0: 'No Job', 1: 'Has Job'}),
                ("Anxiety by Job Status", 'job', 'anxiety_1to5', {0: 'No Job', 1: 'Has Job'}),
                ("Sleep Quality by Extracurricular", 'extracurricular', 'sleep_quality_1to10', {0: 'No', 1: 'Yes'}),
                ("Study Hours by Gender", 'gender', 'study_hours_per_week', None),
                ("CGPA by Gender", 'gender', 'cgpa', None),
                ("Sleep Hours by Year of Study", 'year_of_study', 'sleep_hours', None)
            ]
            
            # Display boxplots in a grid
            for i in range(0, len(boxplot_configs), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < len(boxplot_configs):
                        title, cat_var, num_var, mapping = boxplot_configs[i]
                        try:
                            plot_df = filtered_df.copy()
                            if mapping:
                                plot_df[cat_var] = plot_df[cat_var].map(mapping)
                            
                            fig = px.box(plot_df, x=cat_var, y=num_var, title=title)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying {title}: {str(e)}")
                
                with col2:
                    if i + 1 < len(boxplot_configs):
                        title, cat_var, num_var, mapping = boxplot_configs[i + 1]
                        try:
                            plot_df = filtered_df.copy()
                            if mapping:
                                plot_df[cat_var] = plot_df[cat_var].map(mapping)
                            
                            fig = px.box(plot_df, x=cat_var, y=num_var, title=title)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying {title}: {str(e)}")
            
            # Boxplot summary
            st.markdown("<h4>Boxplot Analysis Summary</h4>", unsafe_allow_html=True)
            
            try:
                summary_data = []
                for title, cat_var, num_var, mapping in boxplot_configs:
                    plot_df = filtered_df.copy()
                    if mapping:
                        plot_df[cat_var] = plot_df[cat_var].map(mapping)
                    
                    categories = sorted(plot_df[cat_var].unique())
                    data = [plot_df[plot_df[cat_var] == cat][num_var] for cat in categories]
                    
                    # Calculate means and sample sizes
                    means = [np.mean(group) for group in data]
                    sizes = [len(group) for group in data]
                    
                    for i, cat in enumerate(categories):
                        summary_data.append({
                            'Comparison': title,
                            'Category': cat,
                            'Mean': f"{means[i]:.2f}",
                            'Sample Size': sizes[i]
                        })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error calculating boxplot summary: {str(e)}")

with tab3:
    st.markdown("<h2 class='section-header'>Hypothesis Testing Results</h2>", unsafe_allow_html=True)
    
    # Check if filtered data is not empty
    if len(filtered_df) == 0:
        st.warning("No data available with current filters. Please adjust your filter settings.")
    else:
        # Set significance level
        alpha = 0.05
        
        # Create subtabs for different hypothesis tests
        ht_tab1, ht_tab2, ht_tab3, ht_tab4, ht_tab5, ht_tab6, ht_tab7 = st.tabs([
            "Summary", 
            "Stress by Extracurricular", 
            "Anxiety by Year", 
            "Study Hours vs Stress",
            "Sleep Hours vs Anxiety",
            "High Stress by Job",
            "Multiple Regression"
        ])
        
        with ht_tab1:
            st.markdown("<h3>Hypothesis Testing Summary</h3>", unsafe_allow_html=True)
            
            # Perform all tests for summary
            summary_data = []
            
            # 1. Stress by extracurricular participation
            try:
                stress_extra_yes = filtered_df[filtered_df['extracurricular'] == 1]['stress_1to5']
                stress_extra_no = filtered_df[filtered_df['extracurricular'] == 0]['stress_1to5']
                t_stat, p_value = stats.ttest_ind(stress_extra_yes, stress_extra_no)
                summary_data.append(["Stress by Extracurricular", "t-test", f"{p_value:.4f}", alpha])
            except:
                summary_data.append(["Stress by Extracurricular", "t-test", "N/A", alpha])
            
            # 2. Anxiety across year of study
            try:
                anxiety_by_year = [filtered_df[filtered_df['year_of_study'] == i]['anxiety_1to5'] for i in range(1, 5)]
                f_stat, p_value = stats.f_oneway(*anxiety_by_year)
                summary_data.append(["Anxiety by Year", "ANOVA", f"{p_value:.4f}", alpha])
            except:
                summary_data.append(["Anxiety by Year", "ANOVA", "N/A", alpha])
            
            # 3. Study hours vs. stress
            try:
                corr, p_value = stats.pearsonr(filtered_df['study_hours_per_week'], filtered_df['stress_1to5'])
                summary_data.append(["Study Hours vs Stress", "Correlation", f"{p_value:.4f}", alpha])
            except:
                summary_data.append(["Study Hours vs Stress", "Correlation", "N/A", alpha])
            
            # 4. Sleep hours vs. anxiety
            try:
                corr, p_value = stats.pearsonr(filtered_df['sleep_hours'], filtered_df['anxiety_1to5'])
                summary_data.append(["Sleep Hours vs Anxiety", "Correlation", f"{p_value:.4f}", alpha])
            except:
                summary_data.append(["Sleep Hours vs Anxiety", "Correlation", "N/A", alpha])
            
            # 5. High stress proportions by job status
            try:
                filtered_df['high_stress'] = (filtered_df['stress_1to5'] >= 4).astype(int)
                contingency_table = pd.crosstab(filtered_df['job'], filtered_df['high_stress'])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                summary_data.append(["High Stress by Job", "Chi-square", f"{p_value:.4f}", alpha])
            except:
                summary_data.append(["High Stress by Job", "Chi-square", "N/A", alpha])
            
            # 6. Multiple regression
            try:
                X_vars = ['study_hours_per_week', 'sleep_hours', 'sleep_quality_1to10',
                         'extracurricular', 'job', 'year_of_study']
                X = filtered_df[X_vars]
                y = filtered_df['stress_1to5']
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                p_value = model.f_pvalue
                summary_data.append(["Multiple Regression", "F-test", f"{p_value:.4f}", alpha])
            except:
                summary_data.append(["Multiple Regression", "F-test", "N/A", alpha])
            
            # Create summary table
            summary_df = pd.DataFrame(summary_data, columns=['Test', 'Method', 'p-value', 'α'])
            summary_df['Significant'] = summary_df['p-value'].apply(
                lambda x: "Yes" if isinstance(x, str) and x != "N/A" and float(x) < alpha else "No" if x != "N/A" else "N/A"
            )
            summary_df['Conclusion'] = summary_df.apply(
                lambda row: "Reject H₀" if row['Significant'] == "Yes" else "Fail to reject H₀" if row['Significant'] == "No" else "N/A", 
                axis=1
            )
            
            # Display summary table
            st.dataframe(summary_df, use_container_width=True)
            
            # Summary statistics
            sig_count = sum(1 for row in summary_data if row[2] != "N/A" and float(row[2]) < alpha)
            total_count = sum(1 for row in summary_data if row[2] != "N/A")
            
            st.metric("Significant Results", f"{sig_count}/{total_count}", f"{sig_count/total_count*100:.1f}%")
            
            # Interpretation
            st.markdown("""
            **Interpretation Guide:**
            - **p-value < α (0.05)**: Statistically significant result - reject the null hypothesis
            - **p-value ≥ α (0.05)**: Not statistically significant - fail to reject the null hypothesis
            """)
        
        with ht_tab2:
            st.markdown("<h3>Stress by Extracurricular Participation</h3>", unsafe_allow_html=True)
            
            try:
                stress_extra_yes = filtered_df[filtered_df['extracurricular'] == 1]['stress_1to5']
                stress_extra_no = filtered_df[filtered_df['extracurricular'] == 0]['stress_1to5']
                
                # Display hypothesis info
                st.markdown("""
                **Hypotheses:**
                - H₀: μ₁ = μ₂ (No difference in mean stress between groups)
                - H₁: μ₁ ≠ μ₂ (Difference in mean stress between groups)
                
                **Test:** Independent t-test
                **Significance level:** α = 0.05
                """)
                
                # Group statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Extracurricular (n)", len(stress_extra_yes), 
                             f"M = {stress_extra_yes.mean():.2f}, SD = {stress_extra_yes.std():.2f}")
                with col2:
                    st.metric("No Extracurricular (n)", len(stress_extra_no), 
                             f"M = {stress_extra_no.mean():.2f}, SD = {stress_extra_no.std():.2f}")
                
                # Perform test
                t_stat, p_value = stats.ttest_ind(stress_extra_yes, stress_extra_no)
                pooled_sd = np.sqrt(((len(stress_extra_yes)-1)*stress_extra_yes.var() +
                                    (len(stress_extra_no)-1)*stress_extra_no.var()) /
                                    (len(stress_extra_yes) + len(stress_extra_no) - 2))
                cohens_d = (stress_extra_yes.mean() - stress_extra_no.mean()) / pooled_sd
                
                # Display results
                st.markdown(f"""
                **Results:**
                - t({len(stress_extra_yes)+len(stress_extra_no)-2}) = {t_stat:.3f}
                - p = {p_value:.4f}
                - Effect size (Cohen's d) = {cohens_d:.3f}
                """)
                
                # Conclusion
                if p_value < alpha:
                    st.success("**CONCLUSION:** Reject H₀ - Significant difference found")
                    direction = "higher" if stress_extra_yes.mean() > stress_extra_no.mean() else "lower"
                    st.info(f"Students with extracurricular activities have {direction} stress levels")
                else:
                    st.error("**CONCLUSION:** Fail to reject H₀ - No significant difference")
                
                # Visualization
                fig = px.box(filtered_df, x='extracurricular', y='stress_1to5', 
                            title='Stress Levels by Extracurricular Participation',
                            labels={'extracurricular': 'Extracurricular Participation', 'stress_1to5': 'Stress Level'})
                fig.update_xaxes(tickvals=[0, 1], ticktext=['No', 'Yes'])
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error performing test: {str(e)}")
        
        with ht_tab3:
            st.markdown("<h3>Anxiety Across Year of Study</h3>", unsafe_allow_html=True)
            
            try:
                # Display hypothesis info
                st.markdown("""
                **Hypotheses:**
                - H₀: μ₁ = μ₂ = μ₃ = μ₄ (No difference in mean anxiety across years)
                - H₁: At least one μ differs (Difference in mean anxiety across years)
                
                **Test:** One-way ANOVA
                **Significance level:** α = 0.05
                """)
                
                # Group statistics
                years = sorted(filtered_df['year_of_study'].unique())
                anxiety_by_year = [filtered_df[filtered_df['year_of_study'] == i]['anxiety_1to5'] for i in years]
                
                cols = st.columns(len(years))
                for i, year in enumerate(years):
                    group_data = filtered_df[filtered_df['year_of_study'] == year]['anxiety_1to5']
                    with cols[i]:
                        st.metric(f"Year {year} (n)", len(group_data), 
                                 f"M = {group_data.mean():.2f}, SD = {group_data.std():.2f}")
                
                # Perform test
                f_stat, p_value = stats.f_oneway(*anxiety_by_year)
                ss_between = sum(len(group) * (group.mean() - filtered_df['anxiety_1to5'].mean())**2 for group in anxiety_by_year)
                ss_total = sum((x - filtered_df['anxiety_1to5'].mean())**2 for x in filtered_df['anxiety_1to5'])
                eta_squared = ss_between / ss_total
                
                # Display results
                st.markdown(f"""
                **Results:**
                - F({len(years)-1}, {len(filtered_df)-len(years)}) = {f_stat:.3f}
                - p = {p_value:.4f}
                - Effect size (Eta squared) = {eta_squared:.3f}
                """)
                
                # Conclusion
                if p_value < alpha:
                    st.success("**CONCLUSION:** Reject H₀ - Significant differences across years")
                else:
                    st.error("**CONCLUSION:** Fail to reject H₀ - No significant differences")
                
                # Visualization
                fig = px.box(filtered_df, x='year_of_study', y='anxiety_1to5', 
                            title='Anxiety Levels by Year of Study',
                            labels={'year_of_study': 'Year of Study', 'anxiety_1to5': 'Anxiety Level'})
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error performing test: {str(e)}")
        
        with ht_tab4:
            st.markdown("<h3>Study Hours vs. Stress</h3>", unsafe_allow_html=True)
            
            try:
                # Display hypothesis info
                st.markdown("""
                **Hypotheses:**
                - H₀: ρ = 0 (No correlation between study hours and stress)
                - H₁: ρ ≠ 0 (Correlation exists between study hours and stress)
                
                **Test:** Pearson correlation
                **Significance level:** α = 0.05
                """)
                
                # Descriptive statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Study Hours", f"M = {filtered_df['study_hours_per_week'].mean():.2f}", 
                             f"SD = {filtered_df['study_hours_per_week'].std():.2f}")
                with col2:
                    st.metric("Stress Level", f"M = {filtered_df['stress_1to5'].mean():.2f}", 
                             f"SD = {filtered_df['stress_1to5'].std():.2f}")
                
                # Perform test
                corr, p_value = stats.pearsonr(filtered_df['study_hours_per_week'], filtered_df['stress_1to5'])
                
                # Display results
                st.markdown(f"""
                **Results:**
                - r({len(filtered_df)-2}) = {corr:.3f}
                - p = {p_value:.4f}
                - Effect size (r) = {corr:.3f}
                """)
                
                # Conclusion
                if p_value < alpha:
                    st.success("**CONCLUSION:** Reject H₀ - Significant correlation found")
                    direction = "positive" if corr > 0 else "negative"
                    st.info(f"{direction.capitalize()} relationship: More study hours associated with {direction} stress")
                else:
                    st.error("**CONCLUSION:** Fail to reject H₀ - No significant correlation")
                
                # Visualization
                fig = px.scatter(filtered_df, x='study_hours_per_week', y='stress_1to5', 
                                title='Study Hours vs. Stress Level',
                                trendline='ols',
                                labels={'study_hours_per_week': 'Study Hours per Week', 'stress_1to5': 'Stress Level'})
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error performing test: {str(e)}")
        
        with ht_tab5:
            st.markdown("<h3>Sleep Hours vs. Anxiety</h3>", unsafe_allow_html=True)
            
            try:
                # Display hypothesis info
                st.markdown("""
                **Hypotheses:**
                - H₀: ρ = 0 (No correlation between sleep hours and anxiety)
                - H₁: ρ ≠ 0 (Correlation exists between sleep hours and anxiety)
                
                **Test:** Pearson correlation
                **Significance level:** α = 0.05
                """)
                
                # Descriptive statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sleep Hours", f"M = {filtered_df['sleep_hours'].mean():.2f}", 
                             f"SD = {filtered_df['sleep_hours'].std():.2f}")
                with col2:
                    st.metric("Anxiety Level", f"M = {filtered_df['anxiety_1to5'].mean():.2f}", 
                             f"SD = {filtered_df['anxiety_1to5'].std():.2f}")
                
                # Perform test
                corr, p_value = stats.pearsonr(filtered_df['sleep_hours'], filtered_df['anxiety_1to5'])
                
                # Display results
                st.markdown(f"""
                **Results:**
                - r({len(filtered_df)-2}) = {corr:.3f}
                - p = {p_value:.4f}
                - Effect size (r) = {corr:.3f}
                """)
                
                # Conclusion
                if p_value < alpha:
                    st.success("**CONCLUSION:** Reject H₀ - Significant correlation found")
                    direction = "positive" if corr > 0 else "negative"
                    st.info(f"{direction.capitalize()} relationship: More sleep hours associated with {direction} anxiety")
                else:
                    st.error("**CONCLUSION:** Fail to reject H₀ - No significant correlation")
                
                # Visualization
                fig = px.scatter(filtered_df, x='sleep_hours', y='anxiety_1to5', 
                                title='Sleep Hours vs. Anxiety Level',
                                trendline='ols',
                                labels={'sleep_hours': 'Sleep Hours', 'anxiety_1to5': 'Anxiety Level'})
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error performing test: {str(e)}")
        
        with ht_tab6:
            st.markdown("<h3>High Stress Proportions by Job Status</h3>", unsafe_allow_html=True)
            
            try:
                # Display hypothesis info
                st.markdown("""
                **Hypotheses:**
                - H₀: p₁ = p₂ (No difference in high stress proportion between job groups)
                - H₁: p₁ ≠ p₂ (Difference in high stress proportion between job groups)
                
                **Test:** Chi-square test of independence
                **Significance level:** α = 0.05
                """)
                
                # Create high stress variable
                filtered_df['high_stress'] = (filtered_df['stress_1to5'] >= 4).astype(int)
                
                # Contingency table
                contingency_table = pd.crosstab(filtered_df['job'], filtered_df['high_stress'])
                st.write("**Contingency Table:**")
                st.dataframe(contingency_table, use_container_width=True)
                
                # Proportions
                prop_no_job = contingency_table.loc[0, 1] / contingency_table.loc[0].sum() if contingency_table.loc[0].sum() > 0 else 0
                prop_has_job = contingency_table.loc[1, 1] / contingency_table.loc[1].sum() if contingency_table.loc[1].sum() > 0 else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("No Job", f"{prop_no_job:.3f}", 
                            f"{contingency_table.loc[0, 1]}/{contingency_table.loc[0].sum()}")
                with col2:
                    st.metric("Has Job", f"{prop_has_job:.3f}", 
                            f"{contingency_table.loc[1, 1]}/{contingency_table.loc[1].sum()}")
                
                # Perform test
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                n = len(filtered_df)
                cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                
                # Display results
                st.markdown(f"""
                **Results:**
                - χ²({dof}, N={n}) = {chi2:.3f}
                - p = {p_value:.4f}
                - Effect size (Cramer's V) = {cramers_v:.3f}
                """)
                
                # Conclusion
                if p_value < alpha:
                    st.success("**CONCLUSION:** Reject H₀ - Significant difference in proportions")
                    higher_group = "no job" if prop_no_job > prop_has_job else "has job"
                    st.info(f"Students with {higher_group} have higher proportion of high stress")
                else:
                    st.error("**CONCLUSION:** Fail to reject H₀ - No significant difference")
                
                # Visualization
                fig = px.bar(contingency_table, barmode='group', 
                            title='High Stress Proportions by Job Status',
                            labels={'value': 'Count', 'job': 'Job Status', 'high_stress': 'High Stress'})
                fig.update_xaxes(title_text='Job Status', tickvals=[0, 1], ticktext=['No Job', 'Has Job'])
                fig.update_layout(legend_title_text='High Stress', 
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error performing test: {str(e)}")


            st.markdown("<h3>Multiple Regression Predicting Stress</h3>", unsafe_allow_html=True)
            
            try:
                # Display hypothesis info
                st.markdown("""
                **Hypotheses:**
                - H₀: β = 0 (Predictor has no effect on stress controlling for other variables)
                - H₁: β ≠ 0 (Predictor has effect on stress controlling for other variables)
                
                **Test:** Multiple regression with t-tests for coefficients
                **Significance level:** α = 0.05
                """)
                
                # Prepare regression data
                X_vars = ['study_hours_per_week', 'sleep_hours', 'sleep_quality_1to10',
                        'extracurricular', 'job', 'year_of_study']
                X = filtered_df[X_vars]
                y = filtered_df['stress_1to5']
                
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                
                # Display model summary
                st.markdown(f"""
                **Overall Model Results:**
                - R² = {model.rsquared:.3f}
                - F = {model.fvalue:.3f}
                - p = {model.f_pvalue:.4f}
                """)
                
                # Display coefficients
                coef_data = []
                for i, var in enumerate(X_vars):
                    coef = model.params[i+1]  # +1 for constant
                    p_val = model.pvalues[i+1]
                    sig = "Yes" if p_val < alpha else "No"
                    coef_data.append([var, coef, p_val, sig])
                
                coef_df = pd.DataFrame(coef_data, columns=['Predictor', 'Coefficient', 'p-value', 'Significant'])
                st.write("**Coefficient Estimates:**")
                st.dataframe(coef_df, use_container_width=True)
                
                # Conclusion
                if model.f_pvalue < alpha:
                    st.success("**CONCLUSION:** Reject H₀ - Overall model is significant")
                    sig_predictors = [var for var, sig in zip(X_vars, coef_df['Significant']) if sig == "Yes"]
                    if sig_predictors:
                        st.info(f"Significant predictors: {', '.join(sig_predictors)}")
                    else:
                        st.info("No individual predictors are significant")
                else:
                    st.error("**CONCLUSION:** Fail to reject H₀ - Overall model is not significant")
                
                # Visualization - Coefficient plot
                # Create proper columns for error bars
                coef_plot_df = pd.DataFrame({
                    'Predictor': X_vars,
                    'Coefficient': model.params[1:],
                    'CI_lower': model.conf_int().iloc[1:, 0],
                    'CI_upper': model.conf_int().iloc[1:, 1]
                })
                
                # Calculate error values
                coef_plot_df['error_lower'] = coef_plot_df['Coefficient'] - coef_plot_df['CI_lower']
                coef_plot_df['error_upper'] = coef_plot_df['CI_upper'] - coef_plot_df['Coefficient']
                
                # Create the coefficient plot using Graph Objects for better control
                fig = go.Figure()
                
                # Add points for coefficients
                fig.add_trace(go.Scatter(
                    x=coef_plot_df['Coefficient'],
                    y=coef_plot_df['Predictor'],
                    mode='markers',
                    marker=dict(size=10, color='blue'),
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=coef_plot_df['error_upper'],
                        arrayminus=coef_plot_df['error_lower']
                    ),
                    name='Coefficients'
                ))
                
                # Add vertical line at zero
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                
                # Update layout
                fig.update_layout(
                    title='Regression Coefficients with 95% Confidence Intervals',
                    xaxis_title='Coefficient Value',
                    yaxis_title='Predictor',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error performing regression: {str(e)}")
         
        with ht_tab7:
            st.markdown("<h3>Multiple Regression: Sleep Quality ~ Multiple Predictors</h3>", unsafe_allow_html=True)
            
            try:
                # Display hypothesis info
                st.markdown("""
                **Hypotheses:**
                - H₀: β = 0 (Predictor has no effect on sleep quality controlling for other variables)
                - H₁: β ≠ 0 (Predictor has effect on sleep quality controlling for other variables)
                
                **Test:** Multiple regression with t-tests for coefficients
                **Significance level:** α = 0.05
                """)
                
                # Prepare regression data
                X = filtered_df[['sleep_hours', 'stress_1to5', 'anxiety_1to5']]
                y = filtered_df['sleep_quality_1to10']
                
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                
                # Display model summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R²", f"{model.rsquared:.3f}")
                with col2:
                    st.metric("Adjusted R²", f"{model.rsquared_adj:.3f}")
                with col3:
                    st.metric("Overall p-value", f"{model.f_pvalue:.4f}")
                
                # Display coefficients
                st.write("**Coefficient Estimates:**")
                coef_data = []
                predictors = ['Intercept'] + X.columns.tolist()[1:]  # Skip the 'const' column
                for i, pred in enumerate(predictors):
                    coef_data.append([
                        pred,
                        model.params[i],
                        model.bse[i],
                        model.tvalues[i],
                        model.pvalues[i]
                    ])
                
                coef_df = pd.DataFrame(coef_data, 
                                    columns=['Predictor', 'Coefficient', 'Std. Error', 't-value', 'p-value'])
                st.dataframe(coef_df, use_container_width=True)
                
                # Check multicollinearity
                st.write("**Multicollinearity Check (VIF):**")
                vif_data = pd.DataFrame()
                vif_data["Variable"] = X.columns[1:]  # Skip the 'const' column
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]  # Skip constant
                st.dataframe(vif_data, use_container_width=True)
                
                # Interpretation
                st.markdown("""
                **Interpretation:**
                - **Overall Model**: Statistically significant (p < 0.001) and explains 21.6% of variance
                - **Sleep Hours**: Highly significant (p < 0.001) with strong positive relationship to sleep quality
                - **Stress and Anxiety**: Not statistically significant in this multivariate model
                - **Multicollinearity**: Very high VIF values (>19) for stress and anxiety, indicating severe multicollinearity
                - **Practical Implications**: Sleep hours is the primary predictor of sleep quality
                - **Model Quality**: Good overall fit but multicollinearity issues limit interpretation of individual coefficients
                """)
                
                # Assumption checking
                st.markdown("**Assumption Checking:**")
                residuals = model.resid
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Normality
                    shapiro_stat, shapiro_p = stats.shapiro(residuals)
                    st.metric("Shapiro-Wilk", f"p = {shapiro_p:.4f}", 
                            "Violated" if shapiro_p < 0.05 else "OK")
                
                with col2:
                    # Independence
                    dw = sm.stats.stattools.durbin_watson(residuals)
                    st.metric("Durbin-Watson", f"{dw:.3f}", 
                            "OK (1.5-2.5)" if 1.5 <= dw <= 2.5 else "Potential issue")
                
                with col3:
                    # Homoscedasticity
                    try:
                        bp_test = het_breuschpagan(residuals, model.model.exog)
                        st.metric("Breusch-Pagan", f"p = {bp_test[1]:.4f}",
                                "Violated" if bp_test[1] < 0.05 else "OK")
                    except:
                        pass
                
                # Coefficient plot
                coef_plot_df = pd.DataFrame({
                    'Predictor': X.columns[1:],  # Skip the 'const' column
                    'Coefficient': model.params[1:],
                    'CI_lower': model.conf_int().iloc[1:, 0],
                    'CI_upper': model.conf_int().iloc[1:, 1]
                })
                
                # Calculate error values
                coef_plot_df['error_lower'] = coef_plot_df['Coefficient'] - coef_plot_df['CI_lower']
                coef_plot_df['error_upper'] = coef_plot_df['CI_upper'] - coef_plot_df['Coefficient']
                
                # Create the coefficient plot using Graph Objects
                fig = go.Figure()
                
                # Add points for coefficients
                fig.add_trace(go.Scatter(
                    x=coef_plot_df['Coefficient'],
                    y=coef_plot_df['Predictor'],
                    mode='markers',
                    marker=dict(size=10, color='blue'),
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=coef_plot_df['error_upper'],
                        arrayminus=coef_plot_df['error_lower']
                    ),
                    name='Coefficients'
                ))
                
                # Add vertical line at zero
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                
                # Update layout
                fig.update_layout(
                    title='Regression Coefficients with 95% Confidence Intervals',
                    xaxis_title='Coefficient Value',
                    yaxis_title='Predictor',
                    showlegend=False
                )
                
                # Add a unique key to the plotly_chart
                st.plotly_chart(fig, use_container_width=True, key="sleep_quality_coef_plot")
                
            except Exception as e:
                st.error(f"Error performing regression: {str(e)}")
with tab4:  
        st.markdown("<h2 class='section-header'>Predictive Modeling</h2>", unsafe_allow_html=True)
        
        # Check if filtered data is not empty
        if len(filtered_df) == 0:
            st.warning("No data available with current filters. Please adjust your filter settings.")
        else:
            # Create subtabs for different regression models
            reg_tab1, reg_tab2, reg_tab3, reg_tab4, reg_tab5 = st.tabs([
                "Stress ~ Study Hours", 
                "Stress ~ Year of Study", 
                "Anxiety ~ Sleep Quality",
                "Stress ~ Multiple Predictors",
                "Sleep Quality ~ Multiple Predictors"
            ])
            
            with reg_tab1:
                st.markdown("<h3>Simple Regression: Stress ~ Study Hours</h3>", unsafe_allow_html=True)
                
                try:
                    # Prepare data
                    X = filtered_df[['study_hours_per_week']]
                    y = filtered_df['stress_1to5']
                    
                    # Fit model
                    X_const = sm.add_constant(X)
                    model = sm.OLS(y, X_const).fit()
                    
                    # Display model summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R²", f"{model.rsquared:.3f}")
                    with col2:
                        st.metric("F-statistic", f"{model.fvalue:.3f}")
                    with col3:
                        st.metric("p-value", f"{model.f_pvalue:.4f}")
                    
                    # Display coefficients
                    st.write("**Coefficient Estimates:**")
                    coef_df = pd.DataFrame({
                        'Predictor': ['Intercept', 'Study Hours'],
                        'Coefficient': [model.params[0], model.params[1]],
                        'Std. Error': [model.bse[0], model.bse[1]],
                        't-value': [model.tvalues[0], model.tvalues[1]],
                        'p-value': [model.pvalues[0], model.pvalues[1]]
                    })
                    st.dataframe(coef_df, use_container_width=True)
                    
                    # Interpretation
                    st.markdown("""
                    **Interpretation:**
                    - **Study Hours Coefficient (0.018)**: For each additional hour of study per week, 
                    stress levels increase by 0.018 points on average (on a 1-5 scale)
                    - **Statistical Significance**: p = 0.060 (borderline significant at α = 0.05)
                    - **Model Fit**: R² = 0.031 indicates the model explains only 3.1% of variance in stress levels
                    - **Direction**: Positive relationship suggests more study hours are associated with higher stress
                    """)
                    
                    # Assumption checking
                    st.markdown("**Assumption Checking:**")
                    residuals = model.resid
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Normality
                        shapiro_stat, shapiro_p = stats.shapiro(residuals)
                        st.metric("Shapiro-Wilk (Normality)", f"p = {shapiro_p:.4f}", 
                                "Violated" if shapiro_p < 0.05 else "OK")
                    
                    with col2:
                        # Independence
                        dw = sm.stats.stattools.durbin_watson(residuals)
                        st.metric("Durbin-Watson (Independence)", f"{dw:.3f}", 
                                "OK (1.5-2.5)" if 1.5 <= dw <= 2.5 else "Potential issue")
                    
                    # Homoscedasticity
                    try:
                        bp_test = het_breuschpagan(residuals, model.model.exog)
                        st.metric("Breusch-Pagan (Homoscedasticity)", f"p = {bp_test[1]:.4f}",
                                "Violated" if bp_test[1] < 0.05 else "OK")
                    except:
                        pass
                    
                    # Visualization
                    fig = px.scatter(filtered_df, x='study_hours_per_week', y='stress_1to5', 
                                trendline='ols', title='Stress ~ Study Hours',
                                labels={'study_hours_per_week': 'Study Hours per Week', 'stress_1to5': 'Stress Level'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error performing regression: {str(e)}")
            
            with reg_tab2:
                st.markdown("<h3>Simple Regression: Stress ~ Year of Study</h3>", unsafe_allow_html=True)
                
                try:
                    # Prepare data
                    X = filtered_df[['year_of_study']]
                    y = filtered_df['stress_1to5']
                    
                    # Fit model
                    X_const = sm.add_constant(X)
                    model = sm.OLS(y, X_const).fit()
                    
                    # Display model summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R²", f"{model.rsquared:.3f}")
                    with col2:
                        st.metric("F-statistic", f"{model.fvalue:.3f}")
                    with col3:
                        st.metric("p-value", f"{model.f_pvalue:.4f}")
                    
                    # Display coefficients
                    st.write("**Coefficient Estimates:**")
                    coef_df = pd.DataFrame({
                        'Predictor': ['Intercept', 'Year of Study'],
                        'Coefficient': [model.params[0], model.params[1]],
                        'Std. Error': [model.bse[0], model.bse[1]],
                        't-value': [model.tvalues[0], model.tvalues[1]],
                        'p-value': [model.pvalues[0], model.pvalues[1]]
                    })
                    st.dataframe(coef_df, use_container_width=True)
                    
                    # Interpretation
                    st.markdown("""
                    **Interpretation:**
                    - **Year of Study Coefficient (0.173)**: For each additional year of study, 
                    stress levels increase by 0.173 points on average (on a 1-5 scale)
                    - **Statistical Significance**: p = 0.107 (not significant at α = 0.05)
                    - **Model Fit**: R² = 0.023 indicates the model explains only 2.3% of variance in stress levels
                    - **Direction**: Positive relationship suggests higher years of study are associated with higher stress
                    """)
                    
                    # Visualization
                    fig = px.box(filtered_df, x='year_of_study', y='stress_1to5', 
                            title='Stress by Year of Study',
                            labels={'year_of_study': 'Year of Study', 'stress_1to5': 'Stress Level'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error performing regression: {str(e)}")
            
            with reg_tab3:
                st.markdown("<h3>Simple Regression: Anxiety ~ Sleep Quality</h3>", unsafe_allow_html=True)
                
                try:
                    # Prepare data
                    X = filtered_df[['sleep_quality_1to10']]
                    y = filtered_df['anxiety_1to5']
                    
                    # Fit model
                    X_const = sm.add_constant(X)
                    model = sm.OLS(y, X_const).fit()
                    
                    # Display model summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R²", f"{model.rsquared:.3f}")
                    with col2:
                        st.metric("F-statistic", f"{model.fvalue:.3f}")
                    with col3:
                        st.metric("p-value", f"{model.f_pvalue:.4f}")
                    
                    # Display coefficients
                    st.write("**Coefficient Estimates:**")
                    coef_df = pd.DataFrame({
                        'Predictor': ['Intercept', 'Sleep Quality'],
                        'Coefficient': [model.params[0], model.params[1]],
                        'Std. Error': [model.bse[0], model.bse[1]],
                        't-value': [model.tvalues[0], model.tvalues[1]],
                        'p-value': [model.pvalues[0], model.pvalues[1]]
                    })
                    st.dataframe(coef_df, use_container_width=True)
                    
                    # Interpretation
                    st.markdown("""
                    **Interpretation:**
                    - **Sleep Quality Coefficient (-0.139)**: For each 1-point increase in sleep quality (1-10 scale), 
                    anxiety levels decrease by 0.139 points on average (on a 1-5 scale)
                    - **Statistical Significance**: p = 0.007 (significant at α = 0.05)
                    - **Model Fit**: R² = 0.063 indicates the model explains 6.3% of variance in anxiety levels
                    - **Direction**: Negative relationship suggests better sleep quality is associated with lower anxiety
                    - **Practical Significance**: While statistically significant, the effect size is relatively small
                    """)
                    
                    # Visualization
                    fig = px.scatter(filtered_df, x='sleep_quality_1to10', y='anxiety_1to5', 
                                trendline='ols', title='Anxiety ~ Sleep Quality',
                                labels={'sleep_quality_1to10': 'Sleep Quality (1-10)', 'anxiety_1to5': 'Anxiety Level'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error performing regression: {str(e)}")
            
            with reg_tab4:
                st.markdown("<h3>Multiple Regression: Stress ~ Multiple Predictors</h3>", unsafe_allow_html=True)
                
                try:
                    # Prepare data
                    X = filtered_df[['study_hours_per_week', 'courses_enrolled', 'extracurricular_hours', 
                                'job_hours_per_week', 'sleep_quality_1to10']]
                    y = filtered_df['stress_1to5']
                    
                    # Fit model
                    X_const = sm.add_constant(X)
                    model = sm.OLS(y, X_const).fit()
                    
                    # Display model summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R²", f"{model.rsquared:.3f}")
                    with col2:
                        st.metric("Adjusted R²", f"{model.rsquared_adj:.3f}")
                    with col3:
                        st.metric("Overall p-value", f"{model.f_pvalue:.4f}")
                    
                    # Display coefficients
                    st.write("**Coefficient Estimates:**")
                    coef_data = []
                    predictors = ['Intercept'] + X.columns.tolist()
                    for i, pred in enumerate(predictors):
                        coef_data.append([
                            pred,
                            model.params[i],
                            model.bse[i],
                            model.tvalues[i],
                            model.pvalues[i]
                        ])
                    
                    coef_df = pd.DataFrame(coef_data, 
                                        columns=['Predictor', 'Coefficient', 'Std. Error', 't-value', 'p-value'])
                    st.dataframe(coef_df, use_container_width=True)
                    
                    # Check multicollinearity
                    st.write("**Multicollinearity Check (VIF):**")
                    vif_data = pd.DataFrame()
                    vif_data["Variable"] = X.columns
                    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    st.dataframe(vif_data, use_container_width=True)
                    
                    # Interpretation
                    st.markdown("""
                    **Interpretation:**
                    - **Overall Model**: Not statistically significant (p = 0.162) and explains only 6.9% of variance
                    - **Study Hours**: Borderline significant (p = 0.079) with positive relationship to stress
                    - **Sleep Quality**: Borderline significant (p = 0.097) with negative relationship to stress
                    - **Multicollinearity**: Courses enrolled and sleep quality show high VIF values (>7.5), 
                    indicating potential multicollinearity issues
                    - **Practical Implications**: The model has limited predictive power for stress levels
                    """)
                    
                    # Assumption checking
                    st.markdown("**Assumption Checking:**")
                    residuals = model.resid
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        # Normality
                        shapiro_stat, shapiro_p = stats.shapiro(residuals)
                        st.metric("Shapiro-Wilk", f"p = {shapiro_p:.4f}", 
                                "Violated" if shapiro_p < 0.05 else "OK")
                    
                    with col2:
                        # Independence
                        dw = sm.stats.stattools.durbin_watson(residuals)
                        st.metric("Durbin-Watson", f"{dw:.3f}", 
                                "OK (1.5-2.5)" if 1.5 <= dw <= 2.5 else "Potential issue")
                    
                    with col3:
                        # Homoscedasticity
                        try:
                            bp_test = het_breuschpagan(residuals, model.model.exog)
                            st.metric("Breusch-Pagan", f"p = {bp_test[1]:.4f}",
                                    "Violated" if bp_test[1] < 0.05 else "OK")
                        except:
                            pass
                    
                    # Coefficient plot
                    coef_plot_df = pd.DataFrame({
                        'Predictor': X.columns,
                        'Coefficient': model.params[1:],
                        'CI_lower': model.conf_int().iloc[1:, 0],
                        'CI_upper': model.conf_int().iloc[1:, 1]
                    })
                    
                    coef_plot_df['error_lower'] = coef_plot_df['Coefficient'] - coef_plot_df['CI_lower']
                    coef_plot_df['error_upper'] = coef_plot_df['CI_upper'] - coef_plot_df['Coefficient']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=coef_plot_df['Coefficient'],
                        y=coef_plot_df['Predictor'],
                        mode='markers',
                        marker=dict(size=10, color='blue'),
                        error_x=dict(
                            type='data',
                            symmetric=False,
                            array=coef_plot_df['error_upper'],
                            arrayminus=coef_plot_df['error_lower']
                        ),
                        name='Coefficients'
                    ))
                    fig.add_vline(x=0, line_dash="dash", line_color="red")
                    fig.update_layout(
                        title='Regression Coefficients with 95% Confidence Intervals',
                        xaxis_title='Coefficient Value',
                        yaxis_title='Predictor'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error performing regression: {str(e)}")
            
            with reg_tab5:
                st.markdown("<h3>Multiple Regression: Sleep Quality ~ Multiple Predictors</h3>", unsafe_allow_html=True)
                
                try:
                    # Prepare data
                    X = filtered_df[['sleep_hours', 'stress_1to5', 'anxiety_1to5']]
                    y = filtered_df['sleep_quality_1to10']
                    
                    # Fit model
                    X_const = sm.add_constant(X)
                    model = sm.OLS(y, X_const).fit()
                    
                    # Display model summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R²", f"{model.rsquared:.3f}")
                    with col2:
                        st.metric("Adjusted R²", f"{model.rsquared_adj:.3f}")
                    with col3:
                        st.metric("Overall p-value", f"{model.f_pvalue:.4f}")
                    
                    # Display coefficients
                    st.write("**Coefficient Estimates:**")
                    coef_data = []
                    predictors = ['Intercept'] + X.columns.tolist()
                    for i, pred in enumerate(predictors):
                        coef_data.append([
                            pred,
                            model.params[i],
                            model.bse[i],
                            model.tvalues[i],
                            model.pvalues[i]
                        ])
                    
                    coef_df = pd.DataFrame(coef_data, 
                                        columns=['Predictor', 'Coefficient', 'Std. Error', 't-value', 'p-value'])
                    st.dataframe(coef_df, use_container_width=True)
                    
                    # Check multicollinearity
                    st.write("**Multicollinearity Check (VIF):**")
                    vif_data = pd.DataFrame()
                    vif_data["Variable"] = X.columns
                    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    st.dataframe(vif_data, use_container_width=True)
                    
                    # Interpretation
                    st.markdown("""
                    **Interpretation:**
                    - **Overall Model**: Statistically significant (p < 0.001) and explains 21.6% of variance
                    - **Sleep Hours**: Highly significant (p < 0.001) with strong positive relationship to sleep quality
                    - **Stress and Anxiety**: Not statistically significant in this multivariate model
                    - **Multicollinearity**: Very high VIF values (>19) for stress and anxiety, indicating severe multicollinearity
                    - **Practical Implications**: Sleep hours is the primary predictor of sleep quality
                    - **Model Quality**: Good overall fit but multicollinearity issues limit interpretation of individual coefficients
                    """)
                    
                    # Assumption checking
                    st.markdown("**Assumption Checking:**")
                    residuals = model.resid
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        # Normality
                        shapiro_stat, shapiro_p = stats.shapiro(residuals)
                        st.metric("Shapiro-Wilk", f"p = {shapiro_p:.4f}", 
                                "Violated" if shapiro_p < 0.05 else "OK")
                    
                    with col2:
                        # Independence
                        dw = sm.stats.stattools.durbin_watson(residuals)
                        st.metric("Durbin-Watson", f"{dw:.3f}", 
                                "OK (1.5-2.5)" if 1.5 <= dw <= 2.5 else "Potential issue")
                    
                    with col3:
                        # Homoscedasticity
                        try:
                            bp_test = het_breuschpagan(residuals, model.model.exog)
                            st.metric("Breusch-Pagan", f"p = {bp_test[1]:.4f}",
                                    "Violated" if bp_test[1] < 0.05 else "OK")
                        except:
                            pass
                    
                    # Coefficient plot
                    coef_plot_df = pd.DataFrame({
                        'Predictor': X.columns,
                        'Coefficient': model.params[1:],
                        'CI_lower': model.conf_int().iloc[1:, 0],
                        'CI_upper': model.conf_int().iloc[1:, 1]
                    })
                    
                    coef_plot_df['error_lower'] = coef_plot_df['Coefficient'] - coef_plot_df['CI_lower']
                    coef_plot_df['error_upper'] = coef_plot_df['CI_upper'] - coef_plot_df['Coefficient']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=coef_plot_df['Coefficient'],
                        y=coef_plot_df['Predictor'],
                        mode='markers',
                        marker=dict(size=10, color='blue'),
                        error_x=dict(
                            type='data',
                            symmetric=False,
                            array=coef_plot_df['error_upper'],
                            arrayminus=coef_plot_df['error_lower']
                        ),
                        name='Coefficients'
                    ))
                    fig.add_vline(x=0, line_dash="dash", line_color="red")
                    fig.update_layout(
                        title='Regression Coefficients with 95% Confidence Intervals',
                        xaxis_title='Coefficient Value',
                        yaxis_title='Predictor'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error performing regression: {str(e)}")
            
            # Overall comparison
            st.markdown("---")
            st.markdown("<h3>Model Comparison Summary</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            **Key Findings:**
            
            1. **Simple Regression Models**:
            - Study hours shows a borderline significant positive relationship with stress
            - Year of study shows a non-significant positive relationship with stress  
            - Sleep quality shows a significant negative relationship with anxiety
            
            2. **Multiple Regression Models**:
            - The stress model has limited predictive power (R² = 0.069) and multicollinearity issues
            - The sleep quality model is more effective (R² = 0.216) but has severe multicollinearity
            
            3. **Practical Implications**:
            - Sleep quality is an important factor for mental health outcomes
            - Academic factors (study hours, year of study) have limited direct effects on stress
            - Multicollinearity between stress and anxiety measures complicates multivariate analysis
            
            4. **Model Quality**:
            - Most models violate normality assumptions for residuals
            - The sleep quality model shows the best overall fit and assumption compliance
            """)

with tab5:
    st.markdown('<h2 class="section-header">Raw Data</h2>', unsafe_allow_html=True)
    
    # Data explorer
    st.dataframe(filtered_df, use_container_width=True)
    
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_student_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")


# # Add quick analysis tips
# with st.expander("💡 Analysis Tips"):
#     st.write("""
#     **Quick Insights:**
#     - Use filters to compare different student groups
#     - Check hypothesis testing for statistical significance
#     - Explore regression models for predictive insights
#     - Download filtered data for further analysis
    
#     **Filtering Tips:**
#     - Start broad, then narrow down using multiple filters
#     - Compare employed vs unemployed students
#     - Analyze by academic year for progression trends
#     - Use stress level ranges for targeted analysis
#     """)