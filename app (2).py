import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
import logging
import os
from datetime import datetime

# Setup Logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page Setup
st.set_page_config(page_title="Generic Insurance Risk Dashboard", page_icon="📊", layout="wide")
st.title("Generic Insurance Risk Dashboard")
st.markdown(f"_Prototype v0.1 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload Insurance Dataset (CSV)")
    target_col = st.selectbox("Select Target Column (e.g., Risk Indicator)", options=[])
    numeric_cols = st.multiselect("Select Numeric Features", options=[])
    cat_cols = st.multiselect("Select Categorical Features", options=[])
    date_cols = st.multiselect("Select Date Columns", options=[])
    missing_strategy = st.selectbox("Missing Value Strategy", ["Drop", "Median", "Mode", "Unknown"])
    st.info("Configure settings after uploading a file.")

# Load and Initial Setup
if uploaded_file is not None:
    @st.cache_data
    def load_data(path):
        logger.info("Loading data...")
        df = pd.read_csv(path)
        logger.info("Data loaded successfully")
        return df

    try:
        df = load_data(uploaded_file)
        st.session_state['df'] = df
        # Populate configuration options dynamically
        target_col = target_col or st.selectbox("Select Target Column", df.columns, key="target_col")
        numeric_cols = numeric_cols or st.multiselect("Select Numeric Features", [col for col in df.columns if df[col].dtype in ['int64', 'float64']], key="numeric_cols")
        cat_cols = cat_cols or st.multiselect("Select Categorical Features", [col for col in df.columns if df[col].dtype == 'object' and col != target_col], key="cat_cols")
        date_cols = date_cols or st.multiselect("Select Date Columns", [col for col in df.columns if 'date' in col.lower()], key="date_cols")
    except Exception as e:
        st.error(f"Dataset loading failed: {str(e)}")
        logger.error(f"Dataset loading failed: {str(e)}")
        st.stop()

# Preprocessing Function
def preprocess_data(df, target_col, numeric_cols, cat_cols, date_cols, missing_strategy):
    df = df.copy()
    # Handle missing values
    if missing_strategy == "Drop":
        df = df.dropna()
    elif missing_strategy == "Median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif missing_strategy == "Mode":
        df = df.fillna(df.mode().iloc[0])
    else:  # Unknown
        df = df.fillna('Unknown')

    # Convert dates to numeric features
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df = df.drop(columns=[col])

    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Ensure target column exists and is numeric
    if target_col not in df_encoded.columns:
        st.error(f"Target column '{target_col}' not found after encoding.")
        return None, None
    y = df_encoded[target_col]
    if y.dtype not in ['int64', 'float64']:
        le = LabelEncoder()
        y = le.fit_transform(y)
    X = df_encoded.drop(columns=[target_col])

    return X, y

# Train Model Function
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    return model, X_test, y_test, y_pred, report, roc_auc

# Main Execution
if uploaded_file is not None and target_col and (numeric_cols or cat_cols or date_cols):
    X, y = preprocess_data(df, target_col, numeric_cols, cat_cols, date_cols, missing_strategy)
    if X is not None and y is not None:
        model, X_test, y_test, y_pred, report, roc_auc = train_model(X, y)

        # KPI Cards
        st.header("Key Performance Indicators")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Model AUC", f"{roc_auc:.2f}")
        col3.metric("High Risk Recall", f"{report['1']['recall']:.2f}" if '1' in report else "N/A")

        # Model Performance
        st.header("Model Performance")
        st.text(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        # Basic Visualization
        st.header("Feature Importance")
        importances = model.feature_importances_
        feat_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)
        fig = px.bar(feat_importance, x='Importance', y='Feature', orientation='h', title="Top 10 Feature Importances")
        st.plotly_chart(fig, use_container_width=True)

        # Prediction Section
        st.header("Predict Risk")
        input_data = {}
        for col in X.columns:
            if col in numeric_cols:
                input_data[col] = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            elif col in cat_cols:
                input_data[col] = st.selectbox(f"{col}", df[col].unique())
            elif any(f'{base}_' in col for base in date_cols):
                base = next(base for base in date_cols if f'{base}_' in col)
                if 'year' in col:
                    input_data[col] = st.slider(f"{base} Year", 2000, 2025, 2020)
                elif 'month' in col:
                    input_data[col] = st.slider(f"{base} Month", 1, 12, 6)
                elif 'day' in col:
                    input_data[col] = st.slider(f"{base} Day", 1, 31, 15)

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=False)
            for col in X.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[X.columns]
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]
            st.markdown(f"**Prediction**: {'High Risk' if pred == 1 else 'Low Risk'}")
            st.metric("Probability (High Risk)", f"{prob*100:.1f}%")
            logger.info(f"Prediction: {pred}, Probability: {prob}")

        # Download Section
        st.header("Download Data")
        st.download_button("Download Processed Data", data=df.to_csv(index=False), file_name="processed_data.csv")
        predictions_df = X_test.copy()
        predictions_df['Predicted_Risk'] = y_pred
        st.download_button("Download Predictions", data=predictions_df.to_csv(index=False), file_name="predictions.csv")
    else:
        st.error("Preprocessing failed. Check configuration.")
else:
    st.info("Please upload a dataset and configure the settings.")

# Notes
st.markdown("**Note**: Adjust target and feature selections based on your dataset. Add geographic columns for maps in future updates.", unsafe_allow_html=True)
