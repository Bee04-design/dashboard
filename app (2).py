import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from scipy.stats import ks_2samp
import warnings
import json
import requests
import time

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

# Global API Key (As per constraint, it remains an empty string for the Canvas environment)
API_KEY = ""
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"


# --- 2. GENERATIVE AI FUNCTION ---

def generate_ai_content(system_prompt: str, user_query: str, max_retries=5):
    """
    Calls the Gemini API to generate content with exponential backoff.
    """
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    headers = {'Content-Type': 'application/json'}
    
    # Exponential backoff logic
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'AI response unavailable.')
            
            return text
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                # Handle rate limit (Too Many Requests)
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                return f"Error: API Request Failed ({e})"
        except requests.exceptions.RequestException as e:
            return f"Error: Network or Request Issue ({e})"
        except Exception as e:
            return f"Error: An unexpected error occurred ({e})"
    
    return "Error: Maximum retries exceeded. The service is currently unavailable."


# --- 3. DATA GENERATION & PREPROCESSING ---

# Mock Data Generation (Simplified for demo)
def generate_mock_data(n_rows=500):
    np.random.seed(42)
    data = {
        'claim_id': [f'CL{i:04d}' for i in range(n_rows)],
        'age': np.random.randint(25, 65, n_rows),
        'premium_paid_SZL': np.random.normal(5000, 1500, n_rows).round(2),
        'claim_amount_SZL': np.random.normal(20000, 8000, n_rows).round(2),
        'region': np.random.choice(['Manzini', 'Mbabane', 'Lubombo', 'Shiselweni'], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
        'claim_type': np.random.choice(['Vehicle', 'Home', 'Life', 'Health'], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
        'claim_date': pd.to_datetime(pd.date_range('2023-01-01', periods=n_rows, freq='D') - pd.to_timedelta(np.random.randint(0, 365, n_rows), unit='D')),
        'latitude': np.random.uniform(-26.5, -26.0, n_rows),
        'longitude': np.random.uniform(31.0, 31.5, n_rows),
        'Risk_Target': np.random.choice([0, 1], n_rows, p=[0.9, 0.1]) # 10% fraudulent
    }
    df = pd.DataFrame(data)
    
    # Introduce anomalies in 'fraudulent' claims for better model training
    fraud_indices = df[df['Risk_Target'] == 1].index
    df.loc[fraud_indices, 'claim_amount_SZL'] *= np.random.uniform(1.5, 3.0, len(fraud_indices))
    df.loc[fraud_indices, 'premium_paid_SZL'] *= np.random.uniform(0.5, 0.8, len(fraud_indices))

    return df

@st.cache_data
def load_data_and_train_model():
    df = generate_mock_data()
    
    # Feature Engineering and Encoding
    df['age_group'] = pd.cut(df['age'], bins=[20, 35, 50, 65], labels=['Young', 'Middle', 'Senior'])
    
    features = ['age', 'premium_paid_SZL', 'claim_amount_SZL']
    categorical_features = ['region', 'claim_type', 'age_group']
    
    # One-hot encoding for categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Define features for the model
    X = df_encoded.drop(['claim_id', 'claim_date', 'latitude', 'longitude', 'Risk_Target'], axis=1)
    y = df_encoded['Risk_Target']
    
    # Ensure all columns are numeric
    X = X.select_dtypes(include=np.number)
    
    # Handle missing/Inf values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

    # Train Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    
    # Predict risk score (probability of fraud)
    df['risk_score'] = model.predict_proba(X)[:, 1].round(4)
    
    # Calculate SHAP values for explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    return df, model, X, shap_values, explainer

df, model, X, shap_values, explainer = load_data_and_train_model()

# --- 4. STREAMLIT APP LAYOUT ---

# --- Sidebar ---
st.sidebar.title("AI Sentinel 🛡️")
st.sidebar.markdown(f"**App ID:** `{__app_id if '__app_id' in globals() else 'local-dev'}`")
st.sidebar.divider()

# --- AI Assistant Section (New Integration) ---
with st.sidebar.expander("🤖 AI Assistant (GenAI)", expanded=False):
    st.subheader("Actionable Insights")
    
    selected_claim_id = st.selectbox(
        "Select High-Risk Claim for Summary:",
        options=df[df['risk_score'] > 0.5]['claim_id'].sort_values().tolist()
    )
    
    if st.button("Generate Executive Summary", use_container_width=True):
        st.info("AI is synthesizing data, please wait...")
        
        # 1. Gather Context Data for the selected claim
        claim_data = df[df['claim_id'] == selected_claim_id].iloc[0]
        
        # 2. Get top SHAP drivers for the selected claim (to ground the AI)
        claim_index = df[df['claim_id'] == selected_claim_id].index[0]
        shap_values_instance = shap_values[1][claim_index] # Get SHAP values for positive class (fraud)
        
        # Find the feature names corresponding to the SHAP values instance
        feature_names = X.columns
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values_instance
        }).sort_values(by='shap_value', ascending=False)
        
        top_drivers = shap_df.head(3)['feature'].tolist()

        # 3. Construct detailed user query
        query_details = f"""
        Claim ID: {selected_claim_id}
        Risk Score: {claim_data['risk_score'] * 100:.2f}%
        Claim Amount: SZL {claim_data['claim_amount_SZL']:,.0f}
        Claim Type: {claim_data['claim_type']}
        Region: {claim_data['region']}
        Top Risk Drivers: {', '.join(top_drivers)}
        
        Task: Write a concise, professional 3-sentence executive summary about this high-risk claim.
        The summary must include the risk score and mention the claim amount and the top risk drivers.
        """
        
        # 4. Define System Prompt (The AI's Persona)
        system_prompt = "You are a senior financial risk analyst at a major Eswatini insurance firm. Your tone is professional, urgent, and focused on financial integrity."
        
        # 5. Call the Generative AI function
        summary_text = generate_ai_content(system_prompt, query_details)
        
        st.success("Summary Generated:")
        st.markdown(summary_text)

st.sidebar.divider()
# --- Main Dashboard Section ---
st.title("AI Sentinel: Strategic Capital Efficiency Dashboard 🛡️")
st.markdown("A real-time risk analytics and capital simulation platform for the Eswatini insurance market.")

# ----------------------------------------------------
# TAB 1: Real-Time Risk Engine
# ----------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔥 Risk Engine", "🌍 Geospatial Intelligence", "📈 Capital Simulation"])

with tab1:
    st.header("Real-Time Claim Risk Assessment")

    # --- Claim Selection ---
    risk_df = df.sort_values(by='risk_score', ascending=False).reset_index(drop=True)
    risk_df['Display'] = risk_df.apply(lambda row: f"{row['claim_id']} (Score: {row['risk_score']*100:.1f}%)", axis=1)

    col_select, col_info = st.columns([1, 2])
    
    with col_select:
        selected_claim_display = st.selectbox(
            "Select Claim ID (Sorted by Highest Risk)",
            options=risk_df['Display'].tolist(),
            index=0
        )
        selected_claim = risk_df[risk_df['Display'] == selected_claim_display].iloc[0]
        st.markdown(f"**Claim ID:** `{selected_claim['claim_id']}`")
        st.markdown(f"**Region:** `{selected_claim['region']}`")
        st.markdown(f"**Claim Type:** `{selected_claim['claim_type']}`")
        st.markdown(f"**Claim Date:** `{selected_claim['claim_date'].strftime('%Y-%m-%d')}`")

    with col_info:
        risk_score = selected_claim['risk_score'] * 100
        color = "#FF2B2B" if risk_score >= 50 else "#00FF9D"
        
        # Risk Score KPI
        st.markdown(f"""
            <div style='background-color: #1E2130; padding: 20px; border-radius: 12px; border: 1px solid {color}; box-shadow: 0 0 20px {color}44;'>
                <h3 style='color: {color}; margin-bottom: 0;'>Fraud Risk Score</h3>
                <p style='font-size: 60px; font-weight: bold; color: {color}; margin-top: 5px;'>{risk_score:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()

    # --- SHAP Explainer ---
    st.subheader("Model Explainability (SHAP Analysis)")
    
    # Prepare data for SHAP visualization for the selected claim
    claim_index = df[df['claim_id'] == selected_claim['claim_id']].index[0]
    
    # Use the pre-computed X and shap_values
    instance_to_explain = X.loc[claim_index]
    
    # Calculate SHAP values for this specific instance
    shap_val_instance = explainer.shap_values(instance_to_explain)
    
    fig_shap, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(shap.Explanation(
        values=shap_val_instance[1], # Risk_Target=1 (Fraud)
        base_values=explainer.expected_value[1],
        data=instance_to_explain.values,
        feature_names=X.columns.tolist()
    ), max_display=10, show=False)
    
    ax.set_facecolor("#0E1117")
    fig_shap.patch.set_facecolor("#0E1117")
    plt.yticks(color='white')
    plt.xticks(color='white')
    plt.title(f"SHAP Waterfall for {selected_claim['claim_id']}", color='white')
    plt.xlabel("Contribution to Risk Score", color='white')
    
    st.pyplot(fig_shap, use_container_width=True)

# ----------------------------------------------------
# TAB 2: Geospatial Intelligence
# ----------------------------------------------------
with tab2:
    st.header("Geospatial Risk Density Map")

    # Map configuration for Eswatini region
    eswatini_center = [-26.5225, 31.3267] # Approximate center of Eswatini

    # Filter for high risk claims
    high_risk_df = df[df['risk_score'] > 0.5].copy()

    # Pydeck Hexagon Layer for density
    layer = pdk.Layer(
        "HexagonLayer",
        data=high_risk_df,
        get_position=["longitude", "latitude"],
        radius=5000, # 5km radius for aggregation
        elevation_scale=1000,
        elevation_range=[0, 3000],
        extruded=True,
        # Color based on average risk score (optional, but good for visualization)
        get_fill_color="[255 * (risk_score), 100, 150, 255]",
        pickable=True,
    )

    # Pydeck View State
    view_state = pdk.ViewState(
        latitude=eswatini_center[0],
        longitude=eswatini_center[1],
        zoom=8.5,
        pitch=50,
        bearing=0
    )

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Claims: {elevationValue} \nAvg Risk: {risk_score}"}
    ))

# ----------------------------------------------------
# TAB 3: Capital Simulation
# ----------------------------------------------------
with tab3:
    st.header("Stress Test & Capital Reserve Simulation")
    st.markdown("Assess the potential impact of major external events on current capital reserves (Value at Risk).")
    
    sim_col, res_col = st.columns([1, 2])
    
    with sim_col:
        st.subheader("Scenario Parameters")
        scenario = st.selectbox(
            "Select Stress Scenario:",
            options=["Severe Weather", "Pandemic", "Civil Unrest"]
        )
        severity = st.slider("Severity Level (1 = Low, 10 = Extreme)", 1, 10, 5)
            
    with res_col:
        # Simulation Logic
        base_loss = df[df['Risk_Target']==1]['claim_amount_SZL'].sum()
        sim_loss = base_loss
        
        # Simplified Loss Multiplier based on scenario and severity
        if scenario == "Severe Weather": sim_loss *= (1 + severity * 0.2)
        elif scenario == "Pandemic": sim_loss *= (1 + severity * 0.15)
        elif scenario == "Civil Unrest": sim_loss *= (1 + severity * 0.4)
        
        # Horizontal Bar Comparison
        loss_df = pd.DataFrame({
            'Scenario': ['Current Fraud Loss', f'Simulated: {scenario}'],
            'Loss': [base_loss, sim_loss],
            'Color': ['#00FF9D', '#FF2B2B']
        })
        
        fig_sim = go.Figure(go.Bar(
            x=loss_df['Loss'], y=loss_df['Scenario'], orientation='h',
            marker_color=loss_df['Color'], text=loss_df['Loss'].apply(lambda x: f"SZL {x:,.0f}"),
            textposition='auto'
        ))
        fig_sim.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white",
            xaxis=dict(showgrid=False), height=250, margin=dict(t=20, l=20, r=20, b=20)
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        
        # KPI for difference
        difference = sim_loss - base_loss
        st.markdown(f"""
        <div style='text-align: center; padding-top: 10px;'>
            <p style='color: #E0E0E0; font-size: 18px;'>Projected Additional Loss:</p>
            <p style='font-size: 32px; font-weight: bold; color: #FF2B2B;'>SZL {difference:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
