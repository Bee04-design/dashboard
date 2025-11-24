import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.utils import resample
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. PAGE CONFIG & BASIC SETUP ---
st.set_page_config(
    page_title="AI Sentinel: Focused Risk Analytics",
    page_icon="🔍",
    layout="wide"
)

# Title and Version Info
st.title("🔍 AI Sentinel: Focused Risk Analytics Dashboard")
st.markdown("_Showing only Model Evaluation and Core Risk Trends_")

# --- 2. DATA LOADING & MODEL/METRICS CACHING ---
MODEL_LAST_TRAINED = "2025-11-24 14:30:00"

@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_data():
    """Loads and performs initial cleaning/feature engineering on the dataset."""
    try:
        # Assuming the standard file name for the insurance dataset
        df = pd.read_csv("eswatini_insurance_final_dataset (5).csv")

        # Standardizing column names
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('szl', 'SZL')

        # Feature Engineering: Creating the Target Variable (Risk_Target)
        df['Risk_Target'] = (df['coverage_type'].str.lower().str.contains('premium|high-risk|high') | 
                             (df['claim_amount_SZL'] > df['claim_amount_SZL'].median() * 1.5)).astype(int)

        # Basic Data Cleaning
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].mean())
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna('Missing')
            
        # Feature for Temporal Analysis
        df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce').fillna(pd.to_datetime('2024-01-01'))

        return df

    except Exception as e:
        st.error(f"Error loading data. Ensure 'eswatini_insurance_final_dataset (5).csv' is available. Error: {str(e)}")
        return pd.DataFrame()

# Bootstrapping Function for Robust Metrics
def calculate_bootstrapped_metrics(model, X, y, n_iterations=100, confidence_level=90):
    """Calculates bootstrapped AUC for robust evaluation (your forecasting metric)."""
    stats = []
    if len(X) < 100: return 0.0, (0.0, 0.0) 

    for i in range(n_iterations):
        X_boot, y_boot = resample(X, y, random_state=i)
        y_pred_proba = model.predict_proba(X_boot)[:, 1]
        fpr, tpr, _ = roc_curve(y_boot, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        stats.append(roc_auc)
    
    stats.sort()
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    
    lower_bound = np.percentile(stats, lower_percentile)
    upper_bound = np.percentile(stats, upper_percentile)
    mean_auc = np.mean(stats)
    
    return mean_auc, (lower_bound, upper_bound)


@st.cache_resource(show_spinner="Training Model and Calculating Metrics...")
def get_model_and_metrics(df):
    """Trains a Random Forest model and calculates all requested metrics."""
    df = df.copy()
    exclude_cols = ['Risk_Target', 'claim_id', 'customer_id', 'claim_date', 'coverage_type']
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns], errors='ignore')
    y = df['Risk_Target']

    categorical_features = X.select_dtypes(include='object').columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Use class_weight='balanced' for better handling of imbalanced data
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    pipeline.fit(X, y)
    
    # Predictions
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)
    
    # Metrics
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    # Bootstrapped Metrics (The robust line graph/forecasting substitute)
    mean_auc, auc_ci = calculate_bootstrapped_metrics(pipeline, X, y)
    
    df_with_predictions = df.copy()
    df_with_predictions['predicted_risk'] = y_pred
    df_with_predictions['prediction_proba'] = y_pred_proba
    
    return df_with_predictions, mean_auc, auc_ci, cm, report

# Load Data and Model
df = load_data()

if not df.empty:
    try:
        (df_preds, mean_auc, auc_ci, cm, report) = get_model_and_metrics(df.copy())
        
        # Store for session access
        st.session_state['df_preds'] = df_preds
        st.session_state['mean_auc'] = mean_auc
        st.session_state['auc_ci'] = auc_ci
        st.session_state['cm'] = cm
        st.session_state['report'] = report
        
    except Exception as e:
        st.error(f"Error during model pipeline processing: {e}")
        st.stop()
else:
    st.stop()

# --- 3. DASHBOARD COMPONENTS ---

st.markdown("---")

# --- 3.1 Model Evaluation Metrics ---
st.header("📈 Model Evaluation Metrics")
st.markdown("This section verifies the model's predictive performance and robustness.")

col_auc, col_cm = st.columns([1, 2])

with col_auc:
    # Metric for Bootstrapped AUC (Robust Evaluation)
    st.subheader("Robust Model Evaluation")
    st.metric(
        label="Bootstrapped AUC (90% CI)", 
        value=f"{mean_auc:.4f}", 
        delta=f"Range: {auc_ci[0]:.4f} - {auc_ci[1]:.4f}",
        delta_color="normal"
    )
    st.caption("Bootstrapped AUC provides a stable estimate of performance across many samples, serving as a reliable benchmark for future performance.")

with col_cm:
    st.subheader("Confusion Matrix")
    # Plotting the Confusion Matrix
    fig_cm = px.imshow(cm, text_auto=True, 
                       labels=dict(x="Predicted Class", y="True Class", color="Count"),
                       x=['Low Risk (0)', 'High Risk (1)'],
                       y=['Low Risk (0)', 'High Risk (1)'],
                       color_continuous_scale="blues")
    fig_cm.update_layout(title_text='Classification Accuracy Breakdown')
    st.plotly_chart(fig_cm, use_container_width=True)

st.subheader("Classification Report")
report_df = pd.DataFrame(report).transpose().style.format({
    'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:,.0f}'
})
st.dataframe(report_df, use_container_width=True)
st.caption("Focus on the Recall for the High Risk (1) class; this indicates the model's ability to catch true high-risk cases.")

st.markdown("---")

# --- 3.2 Risk Trend Graph (Line Graph for Claims) ---
st.header("📊 Claim Risk Trend and Cumulative Exposure")
st.markdown("This line graph tracks the historical accumulation of financial exposure from high-risk claims, providing a clear **risk trend**.")

# Time Series Aggregation
# Group by claim_date and calculate key metrics
claims_by_day = df_preds.groupby('claim_date').agg(
    Total_Claims=('claim_id', 'count'),
    High_Risk_Claims=('predicted_risk', 'sum'),
    High_Risk_Exposure=('claim_amount_SZL', lambda x: df_preds.loc[x.index, 'claim_amount_SZL'][df_preds.loc[x.index, 'predicted_risk'] == 1].sum())
).reset_index()

# Calculate Cumulative Exposure over time for the trend line
claims_by_day['Cumulative_Risk_Exposure'] = claims_by_day['High_Risk_Exposure'].cumsum()

# Create the Plotly Line Graph
fig_trend = go.Figure()

# Trace 1: Cumulative Financial Exposure (The main trend line)
fig_trend.add_trace(go.Scatter(
    x=claims_by_day['claim_date'], 
    y=claims_by_day['Cumulative_Risk_Exposure'], 
    mode='lines+markers', 
    name='Cumulative High-Risk Loss (SZL)', 
    line=dict(color='orange', width=3),
    marker=dict(size=4)
))

# Trace 2: Daily High-Risk Claims (Secondary trend, more volatile)
fig_trend.add_trace(go.Scatter(
    x=claims_by_day['claim_date'], 
    y=claims_by_day['High_Risk_Claims'], 
    mode='lines', 
    name='Daily Count of High-Risk Claims',
    yaxis='y2', # Use secondary y-axis
    line=dict(color='blue', width=1, dash='dot')
))

# Update layout for dual Y-axis
fig_trend.update_layout(
    title_text='Historical Claim Risk Trend: Cumulative Financial Exposure', 
    xaxis_title='Claim Date',
    yaxis=dict(
        title='Cumulative High-Risk Loss (SZL)',
        titlefont=dict(color='orange'),
        tickfont=dict(color='orange'),
        showgrid=False
    ),
    yaxis2=dict(
        title='Daily High-Risk Claim Count',
        titlefont=dict(color='blue'),
        tickfont=dict(color='blue'),
        overlaying='y',
        side='right',
        showgrid=False
    ),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)')
)

st.plotly_chart(fig_trend, use_container_width=True)
st.caption("The orange line shows the total accumulated financial impact of predicted high-risk claims over time. The blue line shows the daily frequency of these high-risk claims.")
