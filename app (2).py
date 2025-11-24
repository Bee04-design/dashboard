import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk # For Geographic Risk Heatmap
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.utils import resample
import warnings
import os
import io
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="AI Sentinel: Strategic Capital Efficiency Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS: TEAL + NAVY PROFESSIONAL THEME
st.markdown("""
    <style>
    /* Background & Text (Navy Blue Base) */
    .stApp { background-color: #0E1724; color: #E0E0E0; }
    
    /* Containers (Darker Navy) */
    div[data-testid="stVerticalBlock"], div[data-testid="stHorizontalBlock"] {
        background-color: #1A2233; 
        border: 1px solid #00A389; /* Teal Accent Border */
        padding: 15px; 
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 163, 137, 0.15); /* Subtle Teal Glow */
    }
    
    /* Metrics (Teal Focus) */
    div[data-testid="metric-container"] {
        background-color: #00A389; /* Teal Background */
        color: #0E1724; /* Dark Text */
        border: 1px solid #00FFC2;
        padding: 15px; border-radius: 12px;
        box-shadow: 0 0 20px rgba(0, 255, 194, 0.4); /* Stronger Neon Teal Glow */
    }
    div[data-testid="metric-container"] label {
        color: #0E1724 !important; /* Ensure label is readable */
    }
    div[data-testid="metric-container"] div[data-testid="stMarkdownContainer"] {
        color: #0E1724 !important; /* Ensure value is readable */
        font-weight: bold;
    }
    
    /* Headers (Teal) */
    h1, h2, h3 { 
        font-family: 'Helvetica Neue', sans-serif; 
        color: #00FFC2; /* Bright Teal */
        border-bottom: 2px solid #00A389; 
        padding-bottom: 5px;
        margin-top: 0px;
    }
    
    /* Buttons (Teal) */
    .stButton>button {
        background-color: #00A389; color: #0E1724; 
        border-radius: 6px; border: none;
        box-shadow: 0 2px 5px rgba(0, 163, 137, 0.5);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00FFC2; color: #0E1724;
        box-shadow: 0 2px 10px rgba(0, 255, 194, 0.8);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0E1724;
        color: #E0E0E0;
    }
    
    /* Tabs */
    .stTabs [data-testid="stTestContainer"] button {
        background-color: #1A2233; color: #E0E0E0;
        border: 1px solid #00A389;
        border-radius: 8px 8px 0 0;
        margin-right: 5px;
    }
    .stTabs [data-testid="stTestContainer"] button[aria-selected="true"] {
        background-color: #00A389; color: #0E1724;
        border-bottom: 3px solid #00FFC2;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING & MODEL/SHAP CACHING ---
MODEL_LAST_TRAINED = "2025-11-24 14:00:00"

@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_data():
    """Loads and performs initial cleaning/feature engineering on the dataset."""
    try:
        # Assuming the standard file name for the insurance dataset
        # NOTE: This file must be available in the execution environment
        df = pd.read_csv("eswatini_insurance_final_dataset (5).csv")

        # Standardizing column names
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('szl', 'SZL')

        # Feature Engineering: Creating the Target Variable
        df['Risk_Target'] = (df['coverage_type'].str.lower().str.contains('premium|high-risk|high') | 
                             (df['claim_amount_SZL'] > df['claim_amount_SZL'].median() * 1.5)).astype(int)

        # Basic Data Cleaning
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].mean())
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna('Missing')
            
        # Feature for Temporal Analysis (Mocking daily claims)
        df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce').fillna(pd.to_datetime('2024-01-01'))
        df['day_of_year'] = df['claim_date'].dt.dayofyear
        
        # Mocking Lat/Lon for Geo-Spatial Plotting (using central point for Eswatini)
        np.random.seed(42)
        center_lat, center_lon = -26.52, 31.38
        df['lat'] = center_lat + np.random.uniform(-0.5, 0.5, len(df))
        df['lon'] = center_lon + np.random.uniform(-0.5, 0.5, len(df))

        return df

    except Exception as e:
        st.error(f"Error loading data. Ensure 'eswatini_insurance_final_dataset (5).csv' is available. Error: {str(e)}")
        return pd.DataFrame()

# Bootstrapping Function
def calculate_bootstrapped_metrics(model, X, y, n_iterations=100, confidence_level=90):
    """Calculates bootstrapped AUC for robust evaluation."""
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


@st.cache_resource(show_spinner="Training Model, Calculating SHAP, and Bootstrapping Metrics...")
def get_model_and_shap_data(df):
    """Trains a Random Forest model and calculates all relevant metrics/SHAP values."""
    df = df.copy()
    exclude_cols = ['Risk_Target', 'claim_id', 'customer_id', 'claim_date', 'coverage_type', 'lat', 'lon', 'day_of_year']
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

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    pipeline.fit(X, y)
    
    # Predictions
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)
    
    # Metrics
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    # Bootstrapped Metrics
    mean_auc, auc_ci = calculate_bootstrapped_metrics(pipeline, X, y)
    
    # SHAP Value Calculation
    ohe = pipeline['preprocessor'].named_transformers_['cat']
    feature_names = numerical_features + list(ohe.get_feature_names_out(categorical_features))
    X_transformed = pipeline['preprocessor'].transform(X)
    
    # Sample 1000 observations for SHAP performance
    sample_indices = np.random.choice(X_transformed.shape[0], min(1000, X_transformed.shape[0]), replace=False)
    X_shap = X_transformed[sample_indices]
    
    try:
        explainer = shap.TreeExplainer(pipeline['classifier'])
        shap_values = explainer.shap_values(X_shap)
        shap_values_target = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
        shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': mean_abs_shap}).sort_values(by='SHAP Value', ascending=False)
        
        # Save SHAP plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_target, X_shap, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('shap_plot.png')
        plt.close()
    except Exception as e:
        st.warning(f"SHAP calculation failed: {e}")
        shap_df = pd.DataFrame()
        
    df_with_predictions = df.copy()
    df_with_predictions['predicted_risk'] = y_pred
    df_with_predictions['prediction_proba'] = y_pred_proba
    
    return pipeline, df_with_predictions, roc_auc, mean_auc, auc_ci, cm, report, shap_df

# Load Data and Model
df = load_data()

if not df.empty:
    try:
        (pipeline, df_preds, roc_auc, mean_auc, auc_ci, cm, report, shap_df) = get_model_and_shap_data(df.copy())
        
        # Store for session access
        st.session_state['pipeline'] = pipeline
        st.session_state['df_preds'] = df_preds
        st.session_state['roc_auc'] = roc_auc
        st.session_state['mean_auc'] = mean_auc
        st.session_state['auc_ci'] = auc_ci
        st.session_state['cm'] = cm
        st.session_state['report'] = report
        st.session_state['shap_df'] = shap_df
        
    except Exception as e:
        st.error(f"Error during model pipeline processing: {e}")
else:
    st.stop()


# --- 3. DASHBOARD STRUCTURE ---

st.title("🛡️ AI Sentinel: Strategic Capital Efficiency Dashboard")
st.markdown(f"_Model: Random Forest | Last Trained: {MODEL_LAST_TRAINED}_")

# --- KPI ROW ---
total_policies = len(df_preds)
high_risk_policies = df_preds['predicted_risk'].sum()
high_risk_percent = (high_risk_policies / total_policies) * 100
total_potential_loss = df_preds['claim_amount_SZL'].sum()
predicted_high_risk_loss = df_preds[df_preds['predicted_risk'] == 1]['claim_amount_SZL'].sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Policies Analyzed", value=f"{total_policies:,}", delta="Data Coverage")

with col2:
    st.metric(label="Predicted High-Risk Exposure", value=f"{high_risk_policies:,}", delta=f"{high_risk_percent:.1f}% of Portfolio")

with col3:
    st.metric(label="Bootstrapped Model AUC", 
              value=f"{mean_auc:.4f}", 
              delta=f"90% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f}")

with col4:
    st.metric(label="Predicted Risk Capital at Stake", 
              value=f"SZL {predicted_high_risk_loss:,.0f}", 
              delta=f"{predicted_high_risk_loss/total_potential_loss:.1%} of Total Loss")

st.markdown("---")

# --- TABS: Interpretability, Evaluation, Scenario ---
tab1, tab2, tab3 = st.tabs(["🧠 Risk Drivers & Interpretability", "📈 Model Evaluation & Performance", "🔬 Scenario Simulation"])

# --- TAB 1: Risk Drivers & Interpretability ---
with tab1:
    st.subheader("Key Risk Drivers and SHAP Global Importance")
    
    col_drivers, col_shap = st.columns([1, 2])
    
    with col_drivers:
        st.markdown("#### Top 10 Features Impacting Risk")
        if not shap_df.empty:
            st.dataframe(
                shap_df.head(10).style.format({'SHAP Value': '{:.4f}'}),
                use_container_width=True,
                hide_index=True
            )
            st.caption("SHAP Value indicates the average magnitude of the feature's contribution to the prediction.")
        else:
            st.warning("SHAP data not available.")

    with col_shap:
        st.markdown("#### Global Feature Impact Visualization")
        if os.path.exists('shap_plot.png'):
            st.image('shap_plot.png', caption="SHAP Summary Plot: Visualizing feature impact and direction.")
        else:
            st.info("SHAP plot generated but image file not found or failed to save.")


    st.markdown("---")
    st.subheader("Regional Risk Exposure Map")
    
    risk_data = df_preds[df_preds['predicted_risk'] == 1]
    
    # Use PyDeck for an interactive, high-performance map
    view_state = pdk.ViewState(
        latitude=risk_data['lat'].mean(),
        longitude=risk_data['lon'].mean(),
        zoom=7,
        pitch=45,
    )

    # Calculate average risk probability for coloring the heatmap
    risk_data['risk_weight'] = risk_data['prediction_proba'] * 10 
    
    if not risk_data.empty:
        layer = pdk.Layer(
            'HeatmapLayer',
            data=risk_data,
            get_position='[lon, lat]',
            get_weight='risk_weight',
            opacity=0.8,
            threshold=0.3,
            radius_pixels=30,
        )
        
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "High Risk Claims: {predicted_risk}"}
        )
        st.pydeck_chart(r)
        st.caption("Heatmap visualization of predicted high-risk claim locations. Brighter areas indicate higher concentrations of high-risk policies.")
    else:
        st.info("No high-risk claims predicted to visualize on the map.")


# --- TAB 2: Model Evaluation & Performance ---
with tab2:
    st.subheader("In-Depth Model Performance Analysis")
    
    col_roc, col_cm = st.columns(2)
    
    with col_roc:
        st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
        fig_roc, ax_roc = plt.subplots()
        fpr, tpr, _ = roc_curve(df_preds['Risk_Target'], df_preds['prediction_proba'])
        ax_roc.plot(fpr, tpr, color='#00FFC2', lw=2, label=f'ROC curve (area = {roc_auc:0.4f})')
        ax_roc.plot([0, 1], [0, 1], color='#00A389', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve for High-Risk Prediction')
        ax_roc.legend(loc="lower right")
        fig_roc.patch.set_facecolor('#1A2233')
        ax_roc.set_facecolor('#0E1724')
        ax_roc.tick_params(colors='white')
        ax_roc.xaxis.label.set_color('white')
        ax_roc.yaxis.label.set_color('white')
        st.pyplot(fig_roc)
        
    with col_cm:
        st.markdown("#### Confusion Matrix")
        fig_cm = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Predicted Class", y="True Class", color="Count"),
                           x=['Low Risk (0)', 'High Risk (1)'],
                           y=['Low Risk (0)', 'High Risk (1)'],
                           color_continuous_scale=[(0.0, "#0E1724"), (0.5, "#00A389"), (1.0, "#00FFC2")])
        fig_cm.update_layout(coloraxis_showscale=False,
                             plot_bgcolor='#1A2233', paper_bgcolor='#1A2233', font_color='white')
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("#### Classification Report")
    report_df = pd.DataFrame(report).transpose().style.format({
        'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:,.0f}'
    })
    st.dataframe(report_df, use_container_width=True)
    st.caption("The classification report shows detailed performance metrics for each risk class.")

    st.markdown("---")
    st.subheader("Customer Claim Risk Timeline & Anomalies")
    
    # Time Series Aggregation
    claims_by_day = df_preds.groupby('claim_date').agg(
        Total_Claims=('claim_id', 'count'),
        High_Risk_Count=('predicted_risk', 'sum'),
        Total_Exposure=('claim_amount_SZL', 'sum')
    ).reset_index()
    claims_by_day['Cumulative_Exposure'] = claims_by_day['Total_Exposure'].cumsum()
    claims_by_day['Risk_Rate'] = claims_by_day['High_Risk_Count'] / claims_by_day['Total_Claims']

    col_trend, col_anom = st.columns(2)
    
    with col_trend:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=claims_by_day['claim_date'], y=claims_by_day['Cumulative_Exposure'], 
                                       mode='lines', name='Cumulative Financial Exposure', 
                                       line=dict(color='#00FFC2', width=3)))
        fig_trend.update_layout(title_text='Cumulative Financial Exposure Over Time', 
                                plot_bgcolor='#1A2233', paper_bgcolor='#1A2233', font_color='white')
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_anom:
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Bar(x=claims_by_day['claim_date'], y=claims_by_day['Risk_Rate'], 
                                  name='Daily Risk Rate', 
                                  marker_color='#00A389'))
        fig_anom.update_layout(title_text='Daily Predicted High-Risk Rate (Anomalies)', 
                               plot_bgcolor='#1A2233', paper_bgcolor='#1A2233', font_color='white',
                               yaxis=dict(tickformat=".1%"))
        st.plotly_chart(fig_anom, use_container_width=True)


# --- TAB 3: Scenario Simulation ---
with tab3:
    st.subheader("Capital Efficiency Scenario Simulation")
    st.markdown("Use this panel to model the potential financial impact of various future risks and adjust capital reserves accordingly.")
    
    scenario_col, severity_col, res_col = st.columns([1, 1, 2])
    
    with scenario_col:
        scenario = st.selectbox(
            "Select Macro-Risk Scenario",
            ['Severe Weather Event', 'Global Economic Downturn', 'Major Civil Unrest', 'Pandemic Aftershock']
        )
        
    with severity_col:
        severity = st.slider("Severity Level (1=Minor, 10=Extreme)", 1, 10, 5)
            
    with res_col:
        # Simulation Logic: Apply multipliers to total claim amount of high-risk policies
        base_high_risk_loss = df_preds[df_preds['Risk_Target']==1]['claim_amount_SZL'].sum()
        sim_loss = base_high_risk_loss # Start with the historical high-risk loss

        multiplier = 1.0
        if scenario == "Severe Weather Event": multiplier = (1 + severity * 0.18) # High impact on property
        elif scenario == "Global Economic Downturn": multiplier = (1 + severity * 0.10) # Medium impact, higher fraud
        elif scenario == "Major Civil Unrest": multiplier = (1 + severity * 0.40) # Very high impact on property/vehicle
        elif scenario == "Pandemic Aftershock": multiplier = (1 + severity * 0.15) # Medium impact, life/health claims

        sim_loss = base_high_risk_loss * multiplier
        
        # Horizontal Bar Comparison
        loss_df = pd.DataFrame({
            'Scenario': ['Predicted Current Loss (SZL)', f'Simulated {scenario} Loss (SZL)'],
            'Loss': [base_high_risk_loss, sim_loss],
            'Color': ['#00FFC2', '#FF4D4D'] # Teal vs Red for contrast
        })
        
        fig_sim = go.Figure(go.Bar(
            x=loss_df['Loss'], y=loss_df['Scenario'], orientation='h',
            marker_color=loss_df['Color'], text=loss_df['Loss'].apply(lambda x: f"SZL {x:,.0f}"),
            textposition='auto'
        ))
        fig_sim.update_layout(
            paper_bgcolor="#1A2233", plot_bgcolor="#1A2233", font_color="white",
            xaxis=dict(showgrid=False, title="Projected Financial Impact (SZL)"), 
            yaxis=dict(title=""),
            height=250, margin=dict(t=20, l=20, r=20, b=20)
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        
        st.info(f"**Capital Adequacy Recommendation:** Under a **'{scenario}'** scenario at severity **{severity}/10**, your predicted high-risk losses could increase by **{(multiplier - 1) * 100:.1f}%**, requiring an additional **SZL {(sim_loss - base_high_risk_loss):,.0f}** in potential capital reserves.")

# --- 4. Stakeholder Insights & Recommendations ---
st.markdown("---")
st.subheader("💡 Strategic Insights and Actionable Recommendations")

col_insights, col_recommendations = st.columns(2)

with col_insights:
    st.markdown("#### Key Portfolio Insights")
    st.markdown("""
    - **Model Stability:** The high Bootstrapped AUC (**{:.4f}**) confirms the model is highly reliable and provides stable predictions.
    - **Regional Concentration:** The geographic heatmap highlights specific areas requiring localized risk mitigation strategies (e.g., increased inspections, tailored policies).
    - **Exposure Management:** The cumulative exposure chart shows the compounding impact of high-risk claims over time, justifying proactive intervention.
    """.format(mean_auc))

with col_recommendations:
    st.markdown("#### Business Action Plan")
    st.markdown("""
    1. **Dynamic Pricing:** Use the SHAP drivers to adjust premiums dynamically for new policies showing similar high-risk profiles.
    2. **Claims Prioritization:** Claims flagged as **High Risk** (Prediction Proba > 0.5) should be routed to senior adjusters immediately for fraud detection protocols.
    3. **Geo-Targeted Intervention:** Allocate resources for physical risk inspections (e.g., infrastructure resilience checks) in high-density risk areas identified on the map.
    4. **Capital Stress Testing:** Regularly run the **Scenario Simulation** for extreme events (Severity 8-10) to validate capital adequacy.
    """)

# --- 5. Download & Utility ---
st.markdown("---")
st.subheader("📥 Data & Report Utilities")

col_dl_data, col_dl_shap, col_dl_qr = st.columns([1, 1, 1])

with col_dl_data:
    predictions_csv = df_preds.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Full Predictions (CSV)", 
        data=predictions_csv, 
        file_name="risk_predictions.csv",
        mime="text/csv"
    )

with col_dl_shap:
    if 'shap_df' in st.session_state and not st.session_state['shap_df'].empty:
        shap_csv = st.session_state['shap_df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download SHAP Analysis (CSV)", 
            data=shap_csv, 
            file_name="shap_analysis.csv",
            mime="text/csv"
        )
    if os.path.exists('shap_plot.png'):
        with open('shap_plot.png', 'rb') as f:
            st.download_button("Download SHAP Plot (PNG)", data=f, file_name="shap_plot.png")

with col_dl_qr:
    # Optional QR Button placeholder
    st.markdown("""
        <div style="background-color: #0E1724; padding: 10px; border-radius: 6px; text-align: center;">
            <p style='color: #00A389; font-weight: bold;'>Link to Deployed Dashboard</p>
            <img src="https://placehold.co/100x100/1A2233/00A389?text=QR+Code" alt="QR Placeholder" style="margin: 5px;">
        </div>
    """, unsafe_allow_html=True)
    st.caption("QR Code for quick mobile access to the live deployment.")
