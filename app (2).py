from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
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
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import ks_2samp
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import random
import io
from sklearn.utils.class_weight import compute_class_weight

# Setup Logging with Version Control
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
MODEL_VERSION = "v1.1 (ADASYN Safety Implemented)"
DATASET_VERSION = "2025-05-20"
MODEL_LAST_TRAINED = "2025-05-20 12:10:00"

# Define save_dir globally
save_dir = './'
os.makedirs(save_dir, exist_ok=True)

# Page Setup for Wide Layout
st.set_page_config(page_title="Insurance Risk Dashboard", page_icon="📊", layout="wide")

# Title and Version Info
st.title("Insurance Risk Streamlit Dashboard")
st.markdown(f"_Prototype v0.4.8 | Model: **RandomForest {MODEL_VERSION}** | Last Trained: {MODEL_LAST_TRAINED}_")

# --- Helper Functions ---

@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_data():
    """Loads and performs initial cleaning/feature engineering on the dataset."""
    try:
        # NOTE: This line assumes you have uploaded 'eswatini_insurance_final_dataset (5).csv'
        df = pd.read_csv("eswatini_insurance_final_dataset (5).csv")
        logger.info("Dataset loaded successfully.")

        # Basic Preprocessing
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('szl', 'szl')

        # Create the Target Variable: 'risk_flag' (1 for High Risk/Premium, 0 otherwise)
        # Based on the abstract, "High-Risk Claim Classification" is the goal.
        df['risk_flag'] = (df['coverage_type'].str.lower().str.contains('premium|high-risk|high') | (df['claim_amount_szl'] > df['claim_amount_szl'].median() * 1.5)).astype(int)

        # Handle NaNs and missing values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop rows with NaN in the target variable if any were introduced
        df.dropna(subset=['risk_flag'], inplace=True) 

        # For simplicity, filling numeric NaNs with mean and categorical with 'Missing'
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].mean())
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna('Missing')

        return df

    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource(show_spinner="Training/Loading Model and Calculating SHAP values...")
def get_model_and_shap_data(df):
    """
    Trains a Random Forest model, optionally using ADASYN for imbalance correction, 
    and calculates SHAP values for interpretability.
    """
    # 1. Feature Engineering and Setup
    df = df.copy()
    
    # Check if the target column exists and has variability
    if 'risk_flag' not in df.columns or df['risk_flag'].nunique() < 2:
        raise ValueError("Target variable 'risk_flag' is missing or only contains one class.")

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

    # 2. Imbalance Check and Pipeline Definition (ADASYN + RandomForest)
    class_counts = y.value_counts()
    minority_class = class_counts.min()
    
    # ADASYN requires enough samples in the minority class (typically >= n_neighbors + 1)
    # Using a safe threshold (e.g., 20) for ADASYN stability
    ADASYN_THRESHOLD = 20 

    # Determine if ADASYN should be used
    use_adasyn = minority_class >= ADASYN_THRESHOLD
    
    if use_adasyn:
        logger.info(f"Using ADASYN: Minority class size ({minority_class}) >= {ADASYN_THRESHOLD}")
        pipeline_steps = [
            ('preprocessor', preprocessor),
            # ADASYN parameters can be tuned, n_neighbors must be < minority class size
            ('oversample', ADASYN(random_state=42, n_neighbors=min(5, minority_class - 1))), 
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight=None))
        ]
        pipeline = ImbPipeline(steps=pipeline_steps)
    else:
        logger.warning(f"ADASYN skipped: Minority class size ({minority_class}) < {ADASYN_THRESHOLD}. Using class_weight='balanced'.")
        # If ADASYN is skipped, we use the standard sklearn pipeline and rely on class weights
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        pipeline = SklearnPipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

    # 3. Model Training
    pipeline.fit(X, y)

    # 4. Predictions
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)
    
    # Add predictions to DataFrame
    df_with_predictions = df.copy()
    df_with_predictions['predicted_risk'] = y_pred
    df_with_predictions['prediction_proba'] = y_pred_proba

    # 5. Model Evaluation
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate recall for the positive class (High Risk = 1)
    cm = confusion_matrix(y, y_pred)
    # cm is [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)
    recall_class_1 = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # 6. SHAP Value Calculation
    
    # Extract feature names after preprocessing
    if use_adasyn:
        # If ImbPipeline is used, preprocessor is the first step
        ohe = pipeline['preprocessor'].named_transformers_['cat']
        classifier = pipeline['classifier']
    else:
        # If SklearnPipeline is used, preprocessor is the first step
        ohe = pipeline['preprocessor'].named_transformers_['cat']
        classifier = pipeline['classifier']
        
    feature_names = numerical_features + list(ohe.get_feature_names_out(categorical_features))
    
    # Transform data for SHAP (only preprocessing step needed)
    X_transformed = pipeline['preprocessor'].transform(X)

    # SHAP Explainer (using the trained Random Forest model)
    explainer = shap.TreeExplainer(classifier)
    
    # Use a small sample for SHAP for performance in Streamlit
    sample_indices = random.sample(range(X_transformed.shape[0]), min(1000, X_transformed.shape[0]))
    X_shap = X_transformed[sample_indices]
    
    shap_values = explainer.shap_values(X_shap)
    
    # SHAP Summary DF
    if isinstance(shap_values, list): # For multi-class, get SHAP for target class (1)
        # Ensure the classifier is trained and shap_values[1] exists
        if len(shap_values) > 1:
            shap_values_target = shap_values[1]
        else:
            shap_values_target = shap_values[0] # Fallback if only one class predicted
    else:
        shap_values_target = shap_values

    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
    
    # Handle potential mismatch in feature_names length (e.g., if one-hot encoding failed)
    if len(feature_names) == len(mean_abs_shap):
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': mean_abs_shap
        }).sort_values(by='SHAP Value', ascending=False)
    else:
        logger.error("SHAP feature count mismatch. SHAP DataFrame initialization skipped.")
        shap_df = pd.DataFrame() # Return empty DataFrame on failure

    # Save the feature names (needed for displaying SHAP)
    st.session_state['feature_names'] = feature_names
    
    # Save SHAP plot for download
    try:
        if not shap_df.empty:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_target, X_shap, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'shap_plot.png'))
            plt.close()
        else:
             logger.warning("Skipping SHAP plot generation due to empty SHAP dataframe.")
    except Exception as e:
        logger.warning(f"SHAP plot generation failed: {str(e)}")


    logger.info(f"Model trained. AUC: {roc_auc:.4f}, Recall (High Risk): {recall_class_1:.4f}")

    return pipeline, feature_names, df_with_predictions, roc_auc, recall_class_1, shap_df

# --- Main Application Logic ---

df = load_data()

model = None
roc_auc = 0.0
recall_class_1 = 0.0
shap_df = pd.DataFrame()
df_with_predictions = pd.DataFrame()
feature_names = []

if df is not None:
    try:
        model, feature_names, df_with_predictions, roc_auc, recall_class_1, shap_df = get_model_and_shap_data(df.copy())

        # Save results to session state
        st.session_state['df_with_predictions'] = df_with_predictions
        st.session_state['roc_auc'] = roc_auc
        st.session_state['recall_class_1'] = recall_class_1
        st.session_state['shap_df'] = shap_df
        st.session_state['model'] = model
        st.session_state['feature_names'] = feature_names

        st.success("Model and SHAP data loaded successfully!")

    except ValueError as ve:
        st.error(f"Data Error: {str(ve)}")
        logger.error(f"Data Validation Error: {str(ve)}")
    except Exception as e:
        st.error(f"FATAL ERROR during model initialization or SHAP calculation: {str(e)}")
        st.warning("Please check the 'get_model_and_shap_data' function and the input data.")
        logger.error(f"FATAL ERROR in model/shap loading: {str(e)}")


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
        st.metric(label="Model AUC (Risk Prediction)", value=f"{roc_auc:.4f}", delta="Target: >0.90")

    with col4:
        st.metric(label="Recall (High-Risk Class)", value=f"{recall_class_1:.4f}", delta="Goal: >0.95 (Critical for Fraud)")

    # --- 2. Interpretability (SHAP Analysis) ---
    st.markdown("---")
    st.subheader("🧠 Model Interpretability: SHAP Analysis")

    col5, col6 = st.columns([1, 2])

    with col5:
        st.markdown("#### Top Risk Drivers")
        if not shap_df.empty:
            # Displaying top 10 features
            st.dataframe(shap_df.head(10).style.format({'SHAP Value': '{:.4f}'}), use_container_width=True, hide_index=True)
            st.caption("SHAP Value indicates the feature's average magnitude of impact on the risk prediction.")
        else:
            st.warning("SHAP data is not available.")

    with col6:
        st.markdown("#### Feature Impact Visualization")
        # Display the SHAP summary plot if generated
        if os.path.exists(os.path.join(save_dir, 'shap_plot.png')):
            st.image(os.path.join(save_dir, 'shap_plot.png'), caption="Global Feature Importance (SHAP Summary Plot)")
        else:
            st.warning("SHAP plot image not found.")

    # --- 3. Geographical Risk Analysis (Abstract mentioned location is key) ---
    st.markdown("---")
    st.subheader("🗺️ Geographical & Regional Risk Analysis")

    # Aggregate risk by claim processing branch
    risk_by_location = df_with_predictions.groupby('claim_processing_branch')['predicted_risk'].agg(
        total_claims='count',
        high_risk_count='sum',
    ).reset_index()
    risk_by_location['risk_rate'] = risk_by_location['high_risk_count'] / risk_by_location['total_claims']

    col7, col8 = st.columns([1, 2])

    with col7:
        st.markdown("#### Risk Rate by Branch")
        # Highlight the highest risk rate
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
            # Mock GPS coordinates for Eswatini branches
            # Manzini (High Risk - as per abstract), Mbabane, Nhlangano, Piggs Peak
            branch_coords = {
                'Manzini': (-26.4952, 31.3789),
                'Mbabane': (-26.3197, 31.1345),
                'Nhlangano': (-27.1352, 31.9317),
                'Piggs Peak': (-25.9610, 31.2369),
                'Other': (-26.5000, 31.5000) # Centroid for others
            }
            
            # Map initialization (centered on Eswatini)
            m = folium.Map(location=[-26.5, 31.4], zoom_start=9, tiles="cartodbpositron")
            
            # Prepare data for HeatMap (using the risk rate to simulate intensity)
            heat_data = []
            for index, row in risk_by_location.iterrows():
                branch = row['claim_processing_branch']
                
                # Use a coordinate for the branch if available, otherwise use 'Other'
                lat, lon = branch_coords.get(branch, branch_coords['Other'])

                # Weight the heatmap by risk rate
                heat_data.append([lat, lon, row['risk_rate'] * 10])
                
                # Add markers for branches
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=row['risk_rate'] * 50, # Radius scaled by risk rate
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


# --- 4. User Input for Real-Time Prediction ---
st.markdown("---")
st.subheader("🔍 Real-Time Claim Prediction")

# Collect a random sample for demonstration
if not df.empty:
    # Ensure the sample is not empty before accessing iloc[0]
    sample = df.sample(1).iloc[0]
else:
    # Use the original columns for the empty series
    sample = pd.Series(index=['claim_amount_szl', 'claim_type', 'accident_location', 'cause_of_fire',
                             'age', 'gender', 'location', 'rural_vs_urban', 'highest_education_level',
                             'main_income_source', 'policy_premium_szl', 'policy_deductible_szl']) 
    
input_cols = [
    'claim_amount_szl', 'claim_type', 'accident_location', 'cause_of_fire',
    'age', 'gender', 'location', 'rural_vs_urban', 'highest_education_level',
    'main_income_source', 'policy_premium_szl', 'policy_deductible_szl'
]

input_data = {}
col_input = st.columns(4)

st.markdown("_Modify the values below to see the real-time risk score:_")

for i, feature in enumerate(input_cols):
    # Ensure the feature exists in the DataFrame
    if feature not in df.columns:
        continue

    with col_input[i % 4]:
        if df[feature].dtype == 'object':
            options = df[feature].unique().tolist()
            # Ensure index calculation is safe
            initial_index = options.index(sample[feature]) if feature in sample and sample[feature] in options else 0
            input_data[feature] = st.selectbox(
                f"{feature.replace('_', ' ').title()}",
                options=options,
                index=initial_index,
                key=f"input_{feature}"
            )
        elif df[feature].dtype in ['int64', 'float64']:
            min_val = df[feature].min()
            max_val = df[feature].max()
            # Ensure value is within bounds
            initial_value = float(sample[feature]) if feature in sample else (min_val + max_val) / 2
            initial_value = max(min_val, min(max_val, initial_value))
            
            input_data[feature] = st.number_input(
                f"{feature.replace('_', ' ').title()}",
                min_value=min_val,
                max_value=max_val,
                value=initial_value,
                key=f"input_{feature}"
            )

if st.button("Predict Risk Score"):
    try:
        if 'model' not in st.session_state or st.session_state['model'] is None:
            st.error("Model is not loaded. Please ensure data loading and training succeeded.")
            raise Exception("Model not available in session state.")

        model = st.session_state['model']
        
        # Create a DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        
        # Determine the base columns used during training
        X_train_cols_base = df.drop(columns=['risk_flag', 'claim_id', 'customer_id', 'claim_date', 'coverage_type'], errors='ignore').columns.tolist()
        
        # Fill missing columns in input_df to match X_train_cols_base
        missing_cols = set(X_train_cols_base) - set(input_df.columns)
        for col in missing_cols:
            if df[col].dtype == 'object':
                input_df[col] = 'Missing'
            else:
                input_df[col] = df[col].mean()

        # Reorder columns to match the trained model's feature space (X)
        input_df = input_df.reindex(columns=X_train_cols_base)

        # Make prediction
        pred_proba = model.predict_proba(input_df)[:, 1][0]
        prediction = model.predict(input_df)[0]
        
        st.markdown(f"**Predicted Risk Score:** `{pred_proba:.4f}`")
        if prediction == 1:
            st.error("This claim is **HIGH RISK** and should be flagged for immediate investigation.")
        else:
            st.success("This claim is **LOW RISK** and can proceed normally.")

        # Real-time SHAP explanation
        st.markdown("##### Local Explanation (Mock SHAP Values)")
        
        # Use the preprocessor from the model pipeline to transform the single input sample
        preprocessor = model['preprocessor']
        input_transformed = preprocessor.transform(input_df)
        
        # Get the classifier component
        classifier = model['classifier']
        
        # Only use TreeExplainer if the classifier supports it (RandomForest does)
        explainer = shap.TreeExplainer(classifier)
        
        # Calculate SHAP values for the single transformed input
        shap_values_input = explainer.shap_values(input_transformed)
        
        if isinstance(shap_values_input, list):
            shap_values_input_target = shap_values_input[1][0] # Get values for class 1
        else:
            shap_values_input_target = shap_values_input[0]
            
        # Get feature names from session state (created in get_model_and_shap_data)
        feature_names_shap = st.session_state.get('feature_names', [])
        
        if len(feature_names_shap) == len(shap_values_input_target):
            # Create DataFrame for local explanation
            local_shap_df = pd.DataFrame({
                'Feature': feature_names_shap,
                'Contribution': shap_values_input_target
            }).sort_values(by='Contribution', key=abs, ascending=False).head(10)
            
            # Generate a waterfall plot for local explanation 
            fig_waterfall = px.bar(local_shap_df, 
                                x='Contribution', 
                                y='Feature', 
                                orientation='h',
                                title="Feature Contribution to Predicted Risk Score",
                                color=local_shap_df['Contribution'] > 0,
                                color_discrete_map={True: 'red', False: 'blue'})
            
            st.plotly_chart(fig_waterfall, use_container_width=True)
            st.caption("This visualization shows the exact feature contributions pushing the predicted score up (red) or down (blue).")
        else:
            st.warning("Could not generate local SHAP plot due to feature count mismatch.")


    except Exception as e:
        st.error(f"Prediction failed: {str(e)}. Ensure the model is loaded correctly.")
        logger.error(f"Real-time prediction error: {str(e)}")

# --- 5. Downloadable Reports and Data ---
st.markdown("---")
st.subheader("⬇️ Download Reports and Data")

def generate_pdf_report(roc_auc, recall_class_1, shap_df):
    """Generates a simple PDF report using reportlab and returns a byte buffer."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    y_position = 750
    c.drawString(100, y_position, "Insurance Risk Report (XAI Dashboard)")
    y_position -= 20
    c.drawString(100, y_position, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y_position -= 30
    
    c.drawString(100, y_position, "--- Model Performance ---")
    y_position -= 20
    c.drawString(120, y_position, f"Model AUC: {roc_auc:.4f}")
    y_position -= 15
    c.drawString(120, y_position, f"Recall for High Risk: {recall_class_1:.4f}")
    y_position -= 30
    
    c.drawString(100, y_position, "--- Top Features (SHAP Global Importance) ---")
    y_position -= 20
    
    if not shap_df.empty:
        for i, row in shap_df.head(10).iterrows():
            c.drawString(120, y_position, f"{row['Feature']}: {row['SHAP Value']:.4f}")
            y_position -= 15
            if y_position < 50: # Check for page overflow
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
        if not shap_df.empty:
            shap_csv = shap_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download SHAP Analysis (CSV)", 
                data=shap_csv, 
                file_name="shap_analysis.csv",
                mime="text/csv"
            )

    with col11:
        # Use the latest metrics from session state if available
        current_roc_auc = st.session_state.get('roc_auc', 0.0)
        current_recall = st.session_state.get('recall_class_1', 0.0)
        current_shap_df = st.session_state.get('shap_df', pd.DataFrame())
        
        pdf_buffer = generate_pdf_report(current_roc_auc, current_recall, current_shap_df)
        st.download_button(
            "Download PDF Report", 
            data=pdf_buffer, 
            file_name="insurance_risk_report.pdf",
            mime="application/pdf"
        )
else:
    st.info("Download buttons will appear once the model has successfully loaded and made predictions.")
