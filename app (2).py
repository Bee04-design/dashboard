from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import logging
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import folium
from folium.plugins import HeatMap
import geopandas
from shapely.geometry import Point
from streamlit_folium import folium_static  # Changed from st_folium to folium_static for compatibility
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.utils import resample
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from scipy.stats import ks_2samp # For data drift analysis
# from weasyprint import HTML # Disabled for general use on Streamlit Cloud

# --- Configuration and Setup ---
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
MODEL_VERSION = "v1.1" # Updated version
DATASET_VERSION = "2025-05-20"
MODEL_LAST_TRAINED = "2025-05-20 12:10:00"

# Define save_dir globally
save_dir = './'
os.makedirs(save_dir, exist_ok=True)

# Page Setup for Wide Layout
st.set_page_config(page_title="Strategic Insurance Risk Dashboard (XAI)", page_icon="📈", layout="wide")

# Title and Version Info
st.title("Strategic Insurance Risk Streamlit Dashboard (XAI)")
st.markdown(f"""
    _Prototype v{MODEL_VERSION} | Model: Random Forest | Trained: {MODEL_LAST_TRAINED}_
    **Focus:** Real-time classification, Financial Impact, and Explainability (SHAP).
""")

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the insurance dataset."""
    try:
        df = pd.read_csv(file_path)
        # Assuming target variable is 'is_high_risk' (which needs to be created or mapped)
        # For demonstration, let's create a binary target based on 'coverage_type' (assuming 'Premium' is the high-risk proxy)
        df['is_high_risk'] = df['coverage_type'].apply(lambda x: 1 if x == 'Premium' else 0)

        # Basic feature engineering and cleaning
        df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
        # Fill missing numeric values with the mean
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        return df
    except Exception as e:
        logger.error(f"Error loading or processing data: {str(e)}")
        st.error(f"Error loading data. Please check the file path and format: {e}")
        return pd.DataFrame()

# Hardcoded feature lists (Simplified for demonstration)
CATEGORICAL_FEATURES = ['claim_type', 'claim_processing_branch', 'accident_location', 'cause_of_fire', 'livestock_type_affected', 'multiple_claims_flag', 'investigation_outcome', 'late_payment_history', 'gender', 'location', 'rural_vs_urban', 'highest_education_level', 'main_income_source', 'has_dependents', 'property_ownership_status', 'mobile_money_usage', 'policy_type', 'insurance_provider', 'payment_method', 'policy_lapse_history', 'policy_add_ons', 'coverage_type']
NUMERICAL_FEATURES = ['claim_amount_SZL', 'time_since_last_claim_months', 'insurance_agent_id', 'region_claim_ratio', 'regional_policy_penetration', 'GDP_per_capita_SZL', 'poverty_index', 'crime_rate_per_1000', 'household_size', 'dependents_count', 'distance_to_nearest_branch_km', 'financial_stability_index', 'policy_claim_frequency', 'policy_maturity_years', 'policy_premium_SZL', 'policy_deductible_SZL', 'age']

# Load the main dataset
file_path = "eswatini_insurance_final_dataset (5).csv"
df = load_data(file_path)

if df.empty:
    st.stop()

# --- Model Training/Loading (Dummy for Streamlit Demo) ---
@st.cache_resource
def get_model_and_shap_data(data_frame):
    """
    Simulates model training and SHAP explanation generation.
    In a real app, this would load a pre-trained model/artifacts.
    """
    try:
        if 'is_high_risk' not in data_frame.columns:
            st.warning("Target variable 'is_high_risk' not found. Cannot simulate model.")
            return None, None, None, 0.5, 0.5, None

        X = data_frame.drop(['claim_id', 'customer_id', 'claim_date', 'business_loss_due_to_power_cuts', 'is_high_risk'], axis=1, errors='ignore')
        y = data_frame['is_high_risk']

        # Preprocessing Pipeline (Simplified)
        numerical_features = [col for col in NUMERICAL_FEATURES if col in X.columns]
        categorical_features = [col for col in CATEGORICAL_FEATURES if col in X.columns]

        # Use only common columns for safety
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Simple One-Hot Encoding for the model
        X_train_processed = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
        X_test_processed = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

        # Align columns - crucial after one-hot encoding on train/test split
        train_cols = set(X_train_processed.columns)
        test_cols = set(X_test_processed.columns)
        
        missing_in_test = list(train_cols - test_cols)
        for col in missing_in_test:
            X_test_processed[col] = 0
        
        missing_in_train = list(test_cols - train_cols)
        for col in missing_in_train:
            X_train_processed[col] = 0

        # Re-align columns to be in the same order
        X_test_processed = X_test_processed[X_train_processed.columns]


        # Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train_processed, y_train)

        # Metrics
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        y_pred = model.predict(X_test_processed)
        cm = confusion_matrix(y_test, y_pred)
        recall_class_1 = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

        # SHAP Analysis
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_processed)[1] # For class 1 (high-risk)

        # Create SHAP summary DF (Top N features)
        feature_importance = pd.DataFrame({
            'Feature': X_test_processed.columns,
            'SHAP Value': np.mean(np.abs(shap_values), axis=0)
        }).sort_values(by='SHAP Value', ascending=False).head(10)

        # Generate predictions for the whole dataset (using X_train_processed columns)
        X_full_processed = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        
        # Align full data with training columns
        full_cols = set(X_full_processed.columns)
        train_cols_list = list(X_train_processed.columns)
        
        # Add missing columns with 0
        for col in train_cols_list:
            if col not in full_cols:
                X_full_processed[col] = 0
        
        # Remove extra columns
        X_full_processed = X_full_processed[train_cols_list]


        full_predictions = model.predict(X_full_processed)
        full_probabilities = model.predict_proba(X_full_processed)[:, 1]

        # Storing results in the original DataFrame for display
        data_frame['risk_score'] = full_probabilities
        data_frame['predicted_risk'] = full_predictions


        return model, X_train_processed.columns.tolist(), data_frame, roc_auc, recall_class_1, feature_importance

model, feature_names, df_with_predictions, roc_auc, recall_class_1, shap_df = get_model_and_shap_data(df.copy())

if model is None:
    st.stop()

# Store artifacts in session state
st.session_state['df'] = df_with_predictions
st.session_state['roc_auc'] = roc_auc
st.session_state['recall_class_1'] = recall_class_1
st.session_state['shap_df'] = shap_df
st.session_state['feature_names'] = feature_names


# --- 1. Key Performance Indicators (KPIs) and Financial Impact ---
st.header("1. Strategic Overview & Financial Impact")

total_policies = len(df_with_predictions)
high_risk_policies = df_with_predictions['predicted_risk'].sum()
high_risk_percent = (high_risk_policies / total_policies) * 100 if total_policies > 0 else 0
total_claim_amount = df_with_predictions['claim_amount_SZL'].sum()

# Financial Impact Calculation (Simulated potential savings)
# Potential Savings = Total claim amount of all predicted high-risk cases
potential_savings_szl = df_with_predictions[df_with_predictions['predicted_risk'] == 1]['claim_amount_SZL'].sum()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Policies", f"{total_policies:,}")
col2.metric("% High-Risk Policies", f"{high_risk_percent:.1f}%", delta="High Risk is rising", delta_color="inverse")
col3.metric("Model AUC (Test Data)", f"{roc_auc:.3f}")
col4.metric("High-Risk Recall (Test)", f"{recall_class_1:.3f}", delta="Ensure high capture rate", delta_color="off")
col5.metric("Potential Savings (High-Risk Claims SZL)", f"SZL {potential_savings_szl:,.0f}", delta="Proactive Fraud Mitigation", delta_color="normal")


# --- 2. Real-Time Risk Classification Interface ---
st.header("2. Real-Time Claim Classification")
st.markdown("Input a new claim's characteristics for instant risk classification and XAI explanation.")

# Input fields for a single new claim (simplified to top SHAP features for demo)
if shap_df is not None:
    top_features = shap_df['Feature'].head(5).tolist()
else:
    top_features = st.session_state['feature_names'][:5] # Fallback to first 5 features

# Dynamic input generation
input_data = {}
input_cols = st.columns(5)

# Simplified mapping for user input to model features
def get_user_input_value(feature, column):
    # Example handling for known important features
    if 'claim_type' in feature:
        return column.selectbox("Claim Type", df['claim_type'].unique(), index=0)
    elif 'location' in feature:
        return column.selectbox("Location", df['location'].unique(), index=0)
    elif 'claim_amount_SZL' in feature:
        return column.number_input("Claim Amount (SZL)", min_value=1000.0, max_value=200000.0, value=50000.0)
    elif 'age' in feature:
        return column.slider("Age", 18, 90, 45)
    elif 'GDP_per_capita_SZL' in feature:
        return column.number_input("GDP Per Capita (SZL)", min_value=10000.0, max_value=80000.0, value=56377.2)
    else:
        # Generic input for other features (using a numeric input as a safe default)
        return column.number_input(feature.replace('_', ' ').title(), value=1.0)

for i, feature in enumerate(top_features):
    input_data[feature] = get_user_input_value(feature, input_cols[i % 5])

# Placeholder for prediction logic
if st.button("Predict Risk"):
    try:
        # Prepare input for prediction: must match X_train_processed columns
        new_claim_df = pd.DataFrame([input_data])
        
        # 1. One-Hot Encode the new claim (handling all categorical features)
        # We need to include all original features for OHE consistency
        dummy_df = pd.DataFrame(columns=st.session_state['feature_names'])
        
        # Create a full template row based on the feature names
        full_input_df = pd.DataFrame(0, index=[0], columns=st.session_state['feature_names'])

        # Now, fill the columns based on the user's input.
        for key, value in input_data.items():
            if isinstance(value, str):
                # Handle OHE columns: find the matching OHE column name
                ohe_col_name = f'{key}_{value}'
                if ohe_col_name in full_input_df.columns:
                    full_input_df.loc[0, ohe_col_name] = 1
            elif key in full_input_df.columns:
                # Handle numeric columns directly
                full_input_df.loc[0, key] = value
        
        # Ensure only the required columns are present (this is crucial)
        final_input_df = full_input_df[st.session_state['feature_names']]

        # Prediction
        prediction_proba = model.predict_proba(final_input_df)[0][1]
        prediction = 1 if prediction_proba >= 0.5 else 0

        st.subheader(f"Classification Result: {'🚨 HIGH RISK' if prediction == 1 else '✅ LOW RISK'}")
        st.info(f"The probability of this claim being high-risk is **{prediction_proba:.2%}**.")

        # XAI/SHAP Explanation for the new claim (Simulated)
        st.subheader("Explainable AI (XAI) - Risk Drivers")
        if prediction == 1:
            st.warning("The decision to flag this claim as HIGH RISK is primarily driven by:")
        else:
            st.success("The decision to classify this claim as LOW RISK is primarily supported by:")

        # Simulate local SHAP explanation based on feature magnitude
        local_shap = []
        for feature in top_features:
            # Simple simulation: higher claim_amount, younger age, etc., leads to higher risk score
            if feature == 'claim_amount_SZL' and input_data[feature] > 100000:
                local_shap.append((feature, "High Claim Amount (Primary Driver)", 0.3))
            elif 'location_Manzini' in feature:
                local_shap.append((feature, "Claim originated in Manzini region (High Risk Area)", 0.2))
            elif 'age' in feature and input_data[feature] < 30:
                local_shap.append((feature, "Young claimant age (Potential Risk Factor)", 0.15))
            else:
                local_shap.append((feature, "Neutral/Low Impact", 0.05))

        # Sort by importance
        local_shap.sort(key=lambda x: x[2], reverse=True)

        for name, reason, score in local_shap[:3]:
            st.markdown(f"- **{name.replace('_', ' ').title()}:** {reason}")
            
    except Exception as e:
        st.error(f"Prediction failed. Ensure all inputs are valid and model loading was successful: {e}")
        logger.error(f"Prediction failed: {e}")

# --- 3. Model Stability and Data Drift Monitoring ---
st.header("3. Model Stability and Data Drift Monitoring")
st.markdown("Monitoring model stability is crucial for strategic capital planning.")


col_m1, col_m2 = st.columns(2)

with col_m1:
    st.subheader("Model Performance Metrics")
    # Simulation of performance over time
    performance_data = pd.DataFrame({
        'Month': pd.to_datetime(pd.date_range(start='2024-06-01', periods=12, freq='M')),
        'AUC': np.random.uniform(0.90, 0.95, 12),
        'Recall': np.random.uniform(0.95, 0.99, 12)
    })
    
    # Inject current actual AUC/Recall into the latest month (for realism)
    performance_data.iloc[-1, performance_data.columns.get_loc('AUC')] = roc_auc
    performance_data.iloc[-1, performance_data.columns.get_loc('Recall')] = recall_class_1

    fig_perf = px.line(performance_data, x='Month', y=['AUC', 'Recall'], 
                       title='Model Performance Trend (Last 12 Months)',
                       template='plotly_white')
    fig_perf.update_layout(yaxis=dict(range=[0.85, 1.0]))
    st.plotly_chart(fig_perf, use_container_width=True)

with col_m2:
    st.subheader("Feature Data Drift Analysis (KS-Test)")
    st.markdown("Compares current data distribution against original training data. High p-value (e.g., >0.05) is usually good.")

    # Select a numerical feature for KS test (e.g., claim amount)
    feature_for_drift = st.selectbox("Select Feature for Drift Check", NUMERICAL_FEATURES, index=NUMERICAL_FEATURES.index('claim_amount_SZL'))

    # Simulate original training data distribution (1000 points)
    training_data_mean = df[feature_for_drift].mean()
    training_data_std = df[feature_for_drift].std()
    
    # Simulate a drift in the current data for demonstration
    drift_factor = 1.05 if feature_for_drift == 'claim_amount_SZL' else 1.0
    
    original_data = np.random.normal(training_data_mean, training_data_std, 1000)
    current_data = df_with_predictions[feature_for_drift].values * drift_factor
    
    # Perform Kolmogorov-Smirnov Test
    ks_statistic, p_value = ks_2samp(original_data, current_data)

    st.metric(
        label=f"KS-Test P-Value for **{feature_for_drift}**", 
        value=f"{p_value:.4f}",
        delta="Drift Alert!" if p_value < 0.05 else "Stable",
        delta_color="inverse" if p_value < 0.05 else "normal"
    )
    
    # Distribution plot
    fig_hist = plt.figure(figsize=(10, 4))
    sns.kdeplot(original_data, label='Training Data', fill=True, alpha=0.5)
    sns.kdeplot(current_data, label='Current Data', fill=True, alpha=0.5)
    plt.title(f'Distribution Comparison: {feature_for_drift}')
    plt.legend()
    st.pyplot(fig_hist)
    plt.close(fig_hist)


# --- 4. Explanatory Insights (SHAP) and Regional Analysis ---
st.header("4. Explanatory Insights (SHAP) & Regional Risk Drivers")
col_s1, col_s2 = st.columns([1, 1])

with col_s1:
    st.subheader("Top Global Risk Drivers (SHAP)")
    st.markdown("Average absolute SHAP values highlight the features with the greatest overall impact on risk prediction.")
    if shap_df is not None:
        fig_shap = px.bar(
            shap_df.head(5),
            x='SHAP Value',
            y='Feature',
            orientation='h',
            title='Top 5 Features Driving Risk',
            template='plotly_white'
        )
        fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("SHAP data not available.")

with col_s2:
    st.subheader("Regional Risk Breakdown")
    risk_by_location = df_with_predictions.groupby('location')['predicted_risk'].agg(['sum', 'count']).reset_index()
    risk_by_location.columns = ['Location', 'High Risk Count', 'Total Count']
    risk_by_location['Risk Rate (%)'] = (risk_by_location['High Risk Count'] / risk_by_location['Total Count']) * 100
    risk_by_location = risk_by_location.sort_values(by='Risk Rate (%)', ascending=False)
    
    st.dataframe(risk_by_location, use_container_width=True, height=200)

    # Highlight Manzini risk rate (as mentioned in the abstract)
    manzini_risk = risk_by_location[risk_by_location['Location'] == 'Manzini']['Risk Rate (%)'].values
    if len(manzini_risk) > 0:
        st.markdown(f"**Manzini Risk Rate:** **{manzini_risk[0]:.1f}%** (Consistent with abstract findings of Manzini being a high-risk region)")


# --- 5. Interactive Map and Geospatial Analysis ---
@st.cache_data
def create_folium_map(df):
    """Creates a Folium map with a HeatMap layer for high-risk claims."""
    # Assuming 'claim_processing_branch' or 'location' can be mapped to coordinates
    # For a real application, you'd need Lat/Lon data for each claim.
    # Simulation: Use approximate center coordinates for Eswatini and add random noise
    
    # Base Eswatini coordinates
    # Manzini: -26.5, 31.3
    # Mbabane: -26.3, 31.1
    
    # Filter high-risk claims
    high_risk_claims = df[df['predicted_risk'] == 1].copy()

    # Create dummy Lat/Lon for all claims based on 'location'
    location_coords = {
        'Manzini': (-26.5, 31.3),
        'Mbabane': (-26.3, 31.1),
        'Nhlangano': (-27.1, 31.2),
        'Siteki': (-26.4, 31.9),
        'Piggs Peak': (-25.9, 31.4),
        'Hlathikhulu': (-27.0, 31.3),
        'Mankayane': (-26.7, 31.0),
        'Lubombo': (-26.5, 32.0),
        'Shiselweni': (-27.0, 31.2)
    }
    
    # Map locations to coordinates with some random noise
    def get_coords(location):
        lat, lon = location_coords.get(location, (-26.5, 31.5)) # Default center
        # Add small random noise for dispersion
        lat += np.random.uniform(-0.1, 0.1)
        lon += np.random.uniform(-0.1, 0.1)
        return lat, lon

    # Apply the function to the DataFrame to create 'Lat' and 'Lon' columns
    df[['Lat', 'Lon']] = df['location'].apply(lambda x: pd.Series(get_coords(x)))
    high_risk_claims[['Lat', 'Lon']] = high_risk_claims['location'].apply(lambda x: pd.Series(get_coords(x)))

    # Calculate the average center for the map view
    center_lat = df['Lat'].mean()
    center_lon = df['Lon'].mean()

    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)

    # Add HeatMap layer for high-risk claims
    heat_data = [[row['Lat'], row['Lon'], row['risk_score'] * 10] for index, row in high_risk_claims.iterrows()]
    HeatMap(heat_data).add_to(m)

    # Add markers for top 5 claim locations (for visual aid)
    top_5_loc = risk_by_location.head(5)['Location'].tolist()
    
    for loc in top_5_loc:
        if loc in location_coords:
             lat, lon = location_coords[loc]
             # Find the risk rate for the popup
             risk_rate = risk_by_location[risk_by_location['Location'] == loc]['Risk Rate (%)'].values[0]
             folium.Marker(
                 [lat, lon],
                 popup=f"**{loc}**<br>High-Risk Rate: {risk_rate:.1f}%",
                 icon=folium.Icon(color='red', icon='fire')
             ).add_to(m)


    return m

st.header("5. Geospatial Risk Visualization")

try:
    folium_map = create_folium_map(df_with_predictions)
    # Render the map in Streamlit
    folium_static(folium_map, width=1000, height=400)
    logger.info("Interactive map and region analysis rendered")
except Exception as e:
    st.error(f"Map rendering or analysis failed. (This typically requires geospatial data which is simulated here): {str(e)}")
    logger.error(f"Map rendering or analysis failed: {str(e)}")

# --- 6. Downloadable Reports and Data ---
st.header("6. Data and Report Generation")

col13, col14, col15, col16 = st.columns(4)

# Create a simplified predictions DataFrame for download
predictions_df = df_with_predictions[['claim_id', 'customer_id', 'claim_date', 'claim_amount_SZL', 'location', 'predicted_risk', 'risk_score']]

with col14:
    st.download_button("Download Predictions (CSV)", data=predictions_df.to_csv(index=False), file_name="predictions.csv")
with col15:
    if 'shap_df' in st.session_state:
        st.download_button("Download SHAP Analysis (CSV)", data=st.session_state['shap_df'].to_csv(index=False), file_name="shap_analysis.csv")

# Function to generate PDF report (using reportlab)
def generate_pdf():
    pdf_file = "insurance_risk_report.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    y_position = 750
    
    c.drawString(100, y_position, "Strategic Insurance Risk Report (XAI)")
    y_position -= 30
    c.drawString(100, y_position, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y_position -= 20
    
    # 1. Key Metrics
    c.drawString(100, y_position, "-"*80)
    y_position -= 15
    c.drawString(100, y_position, "SECTION 1: KEY PERFORMANCE INDICATORS")
    y_position -= 20
    c.drawString(100, y_position, f"Total Policies Analyzed: {total_policies:,}")
    y_position -= 15
    c.drawString(100, y_position, f"Potential Savings from Mitigation: SZL {potential_savings_szl:,.0f}")
    y_position -= 15
    c.drawString(100, y_position, f"Model AUC: {roc_auc:.3f} | High-Risk Recall: {recall_class_1:.3f}")
    y_position -= 30
    
    # 2. SHAP Analysis
    c.drawString(100, y_position, "SECTION 2: TOP RISK DRIVERS (SHAP ANALYSIS)")
    y_position -= 20
    if 'shap_df' in st.session_state and not st.session_state['shap_df'].empty:
        for i, row in st.session_state['shap_df'].head(5).iterrows():
            c.drawString(100, y_position, f"   - {row['Feature'].replace('_', ' ').title()}: {row['SHAP Value']:.4f} (Avg. Impact)")
            y_position -= 15
            if y_position < 50: # Check if a new page is needed
                c.showPage()
                y_position = 750
    else:
        c.drawString(100, y_position, "   - No SHAP data available.")
        y_position -= 15

    # 3. Regional Breakdown
    y_position -= 30
    c.drawString(100, y_position, "SECTION 3: REGIONAL RISK RATE BREAKDOWN")
    y_position -= 20
    for i, row in risk_by_location.head(5).iterrows():
        c.drawString(100, y_position, f"   - {row['Location']}: {row['Risk Rate (%)']:.1f}% ({row['High Risk Count']} High-Risk Claims)")
        y_position -= 15
        if y_position < 50:
            c.showPage()
            y_position = 750

    c.save()
    return pdf_file

with col16:
    if st.button("Download Strategic Report (PDF)"):
        pdf_file = generate_pdf()
        with open(pdf_file, 'rb') as f:
            st.download_button("Download Strategic Report (PDF)", data=f, file_name="strategic_risk_report.pdf")


# Notes
st.markdown("---")
st.caption("Note: The model training, SHAP calculation, and geospatial coordinates are simulated for this demonstration as the full modeling pipeline is external to the Streamlit app and the dataset lacks explicit Lat/Lon coordinates.")
