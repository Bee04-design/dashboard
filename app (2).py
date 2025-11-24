from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import shap
import json
import logging
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.impute import SimpleImputer
# Using standard sklearn pipeline, no imblearn dependency in this stable version
from sklearn.utils import resample
from scipy.stats import ks_2samp
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import random
import io

# Setup Logging with Version Control
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
MODEL_VERSION = "v0.9 (Bootstrapped Metrics)"
DATASET_VERSION = "2025-05-20"
MODEL_LAST_TRAINED = "2025-05-20 12:10:00"

# Define save_dir globally
save_dir = './'
os.makedirs(save_dir, exist_ok=True)

# Page Setup for Wide Layout
st.set_page_config(page_title="AI Sentinel Dashboard", page_icon="🛡️", layout="wide")

# Title and Version Info
st.title("🛡️ AI Sentinel: Insurance Risk Dashboard")
st.markdown(f"_Prototype v0.4.5 | Model: **RandomForest {MODEL_VERSION}** | Last Trained: {MODEL_LAST_TRAINED}_")

# --- Helper Functions ---

@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_data():
    """Loads and performs initial cleaning/feature engineering on the dataset."""
    try:
        df = pd.read_csv("eswatini_insurance_final_dataset (5).csv")
        logger.info("Dataset loaded successfully.")

        # Basic Preprocessing
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('szl', 'szl')

        # Create the Target Variable: 'risk_flag' (1 for High Risk/Premium, 0 otherwise)
        df['risk_flag'] = (df['coverage_type'].str.lower().str.contains('premium|high-risk|high') | (df['claim_amount_szl'] > df['claim_amount_szl'].median() * 1.5)).astype(int)

        # Handle NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].mean())
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna('Missing')

        return df

    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_bootstrapped_metrics(model, X, y, n_iterations=100, confidence_level=90):
    """Calculates bootstrapped metrics (e.g., AUC) for robust evaluation."""
    stats = []
    
    # Check if we have enough data for bootstrapping
    if len(X) < 100:
        logger.warning("Skipping bootstrapping: insufficient samples.")
        return 0.0, (0.0, 0.0) # Return safe defaults

    for i in range(n_iterations):
        # Resample data with replacement
        X_boot, y_boot = resample(X, y, random_state=i)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_boot)[:, 1]
        
        # Calculate AUC for this bootstrap sample
        fpr, tpr, _ = roc_curve(y_boot, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        stats.append(roc_auc)
    
    stats.sort()
    
    # Calculate confidence interval
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    
    lower_bound = np.percentile(stats, lower_percentile)
    upper_bound = np.percentile(stats, upper_percentile)
    
    mean_auc = np.mean(stats)
    
    return mean_auc, (lower_bound, upper_bound)


@st.cache_resource(show_spinner="Training/Loading Model and Calculating SHAP values...")
def get_model_and_shap_data(df):
    """
    Trains a Random Forest model using standard processing and calculates SHAP values.
    """
    df = df.copy()
    X = df.drop(columns=['risk_flag', 'claim_id', 'customer_id', 'claim_date', 'coverage_type'], errors='ignore')
    y = df['risk_flag']

    # Define feature types
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Create preprocessor pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # 2. Pipeline Definition (Standard Pipeline)
    # Using class_weight='balanced' to handle imbalance instead of ADASYN
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # 3. Model Training
    pipeline.fit(X, y)

    # 4. Predictions on the whole dataset
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)
    
    # Calculate metrics
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y, y_pred)
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)
    recall_class_1 = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # 5. Bootstrapped Metrics (The key feature of this version)
    mean_auc, auc_ci = calculate_bootstrapped_metrics(pipeline, X, y)
    
    # 6. SHAP Value Calculation
    ohe = pipeline['preprocessor'].named_transformers_['cat']
    feature_names = numerical_features + list(ohe.get_feature_names_out(categorical_features))
    
    X_transformed = pipeline['preprocessor'].transform(X)
    explainer = shap.TreeExplainer(pipeline['classifier'])
    
    # Sample 1000 observations for performance
    sample_indices = random.sample(range(X_transformed.shape[0]), min(1000, X_transformed.shape[0]))
    X_shap = X_transformed[sample_indices]
    
    shap_values = explainer.shap_values(X_shap)
    
    if isinstance(shap_values, list): 
        shap_values_target = shap_values[1]
    else:
        shap_values_target = shap_values
    
    mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
    
    if len(feature_names) == len(mean_abs_shap):
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': mean_abs_shap
        }).sort_values(by='SHAP Value', ascending=False)
    else:
        logger.error("SHAP feature count mismatch. SHAP DataFrame initialization skipped.")
        shap_df = pd.DataFrame() 

    # Save SHAP plot for download
    try:
        if not shap_df.empty:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_target, X_shap, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'shap_plot.png'))
            plt.close()
    except Exception as e:
        logger.warning(f"SHAP plot generation failed: {str(e)}")

    # Prepare final DataFrame with predictions
    df_with_predictions = df.copy()
    df_with_predictions['predicted_risk'] = y_pred
    df_with_predictions['prediction_proba'] = y_pred_proba

    logger.info(f"Model trained. AUC: {roc_auc:.4f}, Recall (High Risk): {recall_class_1:.4f}. Bootstrapped AUC: {mean_auc:.4f}")

    # Return all necessary results, including the CI
    return pipeline, feature_names, df_with_predictions, roc_auc, recall_class_1, shap_df, mean_auc, auc_ci

# --- Main Application Logic ---

df = load_data()

model = None
roc_auc = 0.0
recall_class_1 = 0.0
shap_df = pd.DataFrame()
df_with_predictions = pd.DataFrame()
mean_auc = 0.0
auc_ci = (0.0, 0.0)

if df is not None:
    try:
        (model, feature_names, df_with_predictions, roc_auc, 
         recall_class_1, shap_df, mean_auc, auc_ci) = get_model_and_shap_data(df.copy())

        # Save results to session state
        st.session_state['df_with_predictions'] = df_with_predictions
        st.session_state['roc_auc'] = roc_auc
        st.session_state['recall_class_1'] = recall_class_1
        st.session_state['shap_df'] = shap_df
        st.session_state['model'] = model
        st.session_state['mean_auc'] = mean_auc
        st.session_state['auc_ci'] = auc_ci

        st.success("Model and SHAP data loaded successfully!")

    except Exception as e:
        st.error(f"FATAL ERROR during model initialization or SHAP calculation: {str(e)}")
        logger.error(f"FATAL ERROR in model/shap loading: {str(e)}")

# --- Dashboard Rendering ---

if df_with_predictions.empty:
    st.info("Awaiting model results... Ensure data is loaded correctly.")

else:
    # --- 1. Key Performance Indicators (KPIs) ---
    st.subheader("📊 Key Performance Indicators")
    total_policies = len(df_with_predictions)
    high_risk_policies = df_with_predictions['predicted_risk'].sum()
    high_risk_percent = (high_risk_policies / total_policies) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Policies Analyzed", value=f"{total_policies:,}")

    with col2:
        st.metric(label="Predicted High-Risk Cases", value=f"{high_risk_policies:,}", delta=f"{high_risk_percent:.1f}%")

    with col3:
        st.metric(label="Bootstrapped Model AUC", 
                  value=f"{mean_auc:.4f}", 
                  delta=f"90% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f}")
        st.caption("Bootstrapping provides a more stable evaluation range.")

    with col4:
        st.metric(label="Recall (High-Risk Class)", value=f"{recall_class_1:.4f}", delta="Goal: >0.95 (Critical)")

    # --- 2. Interpretability (SHAP Analysis) ---
    st.markdown("---")
    st.subheader("🧠 Model Interpretability: SHAP Analysis")

    col5, col6 = st.columns([1, 2])

    with col5:
        st.markdown("#### Top Risk Drivers")
        if not shap_df.empty:
            st.dataframe(shap_df.head(10).style.format({'SHAP Value': '{:.4f}'}), use_container_width=True, hide_index=True)
            st.caption("SHAP Value indicates the feature's average magnitude of impact on the risk prediction.")
        else:
            st.warning("SHAP data is not available.")

    with col6:
        st.markdown("#### Feature Impact Visualization")
        if os.path.exists(os.path.join(save_dir, 'shap_plot.png')):
            st.image(os.path.join(save_dir, 'shap_plot.png'), caption="Global Feature Importance (SHAP Summary Plot)")
        else:
            st.warning("SHAP plot image not found.")

    # --- 3. Geographical Risk Analysis ---
    st.markdown("---")
    st.subheader("🗺️ Geographical & Regional Risk Analysis")

    risk_by_location = df_with_predictions.groupby('claim_processing_branch')['predicted_risk'].agg(
        total_claims='count',
        high_risk_count='sum',
    ).reset_index()
    risk_by_location['risk_rate'] = risk_by_location['high_risk_count'] / risk_by_location['total_claims']

    col7, col8 = st.columns([1, 2])

    with col7:
        st.markdown("#### Risk Rate by Branch")
        st.dataframe(
            risk_by_location.sort_values('risk_rate', ascending=False).style.format({
                'risk_rate': '{:.2%}',
                'total_claims': '{:,}',
                'high_risk_count': '{:,}'
            }),
            use_container_width=True,
            hide_index=True
        )

    with col8:
        st.markdown("#### Interactive Risk Map (Mock Data)")
        try:
            branch_coords = {
                'Manzini': (-26.4952, 31.3789),
                'Mbabane': (-26.3197, 31.1345),
                'Nhlangano': (-27.1352, 31.9317),
                'Piggs Peak': (-25.9610, 31.2369),
                'Other': (-26.5000, 31.5000) 
            }
            m = folium.Map(location=[-26.5, 31.4], zoom_start=9, tiles="cartodbpositron")
            heat_data = []
            
            for index, row in risk_by_location.iterrows():
                branch = row['claim_processing_branch']
                lat, lon = branch_coords.get(branch, branch_coords['Other'])
                heat_data.append([lat, lon, row['risk_rate'] * 10])
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=row['risk_rate'] * 50, 
                    color='red' if row['risk_rate'] > risk_by_location['risk_rate'].mean() else 'blue',
                    fill=True,
                    fill_color='red' if row['risk_rate'] > risk_by_location['risk_rate'].mean() else 'blue',
                    tooltip=f"{branch}: {row['risk_rate']:.2%}"
                ).add_to(m)

            HeatMap(heat_data).add_to(m)
            folium_static(m, width=700, height=400)
            logger.info("Interactive map and region analysis rendered")
        except Exception as e:
            st.error(f"Map rendering or analysis failed: {str(e)}")
            logger.error(f"Map rendering or analysis failed: {str(e)}")


# --- 4. User Input for Real-Time Prediction (Simplified) ---
st.markdown("---")
st.subheader("🔍 Real-Time Claim Prediction")

if not df.empty:
    sample = df.sample(1).iloc[0]
else:
    st.warning("Data not loaded, skipping prediction form.")
    
input_cols = [
    'claim_amount_szl', 'claim_type', 'accident_location', 'age', 'gender',
    'rural_vs_urban', 'policy_premium_szl', 'policy_deductible_szl'
]

input_data = {}
col_input = st.columns(4)

st.markdown("_Modify the values below to see the real-time risk score:_")

for i, feature in enumerate(input_cols):
    if feature not in df.columns: continue
    
    with col_input[i % 4]:
        if df[feature].dtype == 'object':
            options = df[feature].unique().tolist()
            initial_index = options.index(sample[feature]) if sample[feature] in options else 0
            input_data[feature] = st.selectbox(
                f"{feature.replace('_', ' ').title()}",
                options=options,
                index=initial_index,
                key=f"input_{feature}"
            )
        elif df[feature].dtype in ['int64', 'float64']:
            min_val = df[feature].min()
            max_val = df[feature].max()
            initial_value = float(sample[feature])
            
            input_data[feature] = st.number_input(
                f"{feature.replace('_', ' ').title()}",
                min_value=min_val,
                max_value=max_val,
                value=initial_value,
                key=f"input_{feature}"
            )

if st.button("Predict Risk Score") and model is not None:
    try:
        input_df = pd.DataFrame([input_data])
        X_train_cols_base = df.drop(columns=['risk_flag', 'claim_id', 'customer_id', 'claim_date', 'coverage_type'], errors='ignore').columns.tolist()
        
        # Fill missing required columns with placeholder values
        missing_cols = set(X_train_cols_base) - set(input_df.columns)
        for col in missing_cols:
            if df[col].dtype == 'object':
                input_df[col] = df[col].mode()[0]
            else:
                input_df[col] = df[col].mean()

        input_df = input_df.reindex(columns=X_train_cols_base, fill_value=0)

        # Make prediction
        pred_proba = model.predict_proba(input_df)[:, 1][0]
        prediction = model.predict(input_df)[0]
        
        st.markdown(f"**Predicted Risk Score:** `{pred_proba:.4f}`")
        if prediction == 1:
            st.error("This claim is **HIGH RISK** and should be flagged for immediate investigation.")
        else:
            st.success("This claim is **LOW RISK** and can proceed normally.")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}. Check console for details.")
        logger.error(f"Real-time prediction error: {str(e)}")

# --- 5. Downloadable Reports and Data ---
st.markdown("---")
st.subheader("⬇️ Download Reports and Data")

def generate_pdf_report(mean_auc, auc_ci, recall_class_1, shap_df):
    """Generates a simple PDF report using reportlab and returns a byte buffer."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    y_position = 750
    c.drawString(100, y_position, "AI Sentinel Insurance Risk Report")
    y_position -= 20
    c.drawString(100, y_position, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y_position -= 30
    
    c.drawString(100, y_position, "--- Model Performance ---")
    y_position -= 20
    c.drawString(120, y_position, f"Bootstrapped Model AUC: {mean_auc:.4f}")
    y_position -= 15
    c.drawString(120, y_position, f"90% AUC CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f}")
    y_position -= 15
    c.drawString(120, y_position, f"Recall for High Risk: {recall_class_1:.4f}")
    y_position -= 30
    
    c.drawString(100, y_position, "--- Top Features (SHAP Global Importance) ---")
    y_position -= 20
    
    if not shap_df.empty:
        for i, row in shap_df.head(10).iterrows():
            c.drawString(120, y_position, f"{row['Feature']}: {row['SHAP Value']:.4f}")
            y_position -= 15
            if y_position < 50: 
                c.showPage()
                y_position = 750
                c.drawString(100, y_position, "Report Continued...")
                y_position -= 20

    c.save()
    buffer.seek(0)
    return buffer

col9, col10, col11 = st.columns(3)

if not df_with_predictions.empty:
    with col9:
        predictions_csv = df_with_predictions.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Full Predictions (CSV)", 
            data=predictions_csv, 
            file_name="risk_predictions.csv",
            mime="text/csv"
        )
    
    with col10:
        if 'shap_df' in st.session_state and not st.session_state['shap_df'].empty:
            shap_csv = st.session_state['shap_df'].to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download SHAP Analysis (CSV)", 
                data=shap_csv, 
                file_name="shap_analysis.csv",
                mime="text/csv"
            )

    with col11:
        current_mean_auc = st.session_state.get('mean_auc', 0.0)
        current_auc_ci = st.session_state.get('auc_ci', (0.0, 0.0))
        current_recall = st.session_state.get('recall_class_1', 0.0)
        current_shap_df = st.session_state.get('shap_df', pd.DataFrame())
        
        pdf_buffer = generate_pdf_report(current_mean_auc, current_auc_ci, current_recall, current_shap_df)
        st.download_button(
            "Download PDF Report", 
            data=pdf_buffer, 
            file_name="insurance_risk_report.pdf",
            mime="application/pdf"
        )
