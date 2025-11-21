import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# --- 1. PAGE CONFIGURATION & CYBER THEME ---
st.set_page_config(
    page_title="KEAS: AI Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NEON / DARK MODE CSS ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    /* Metric Cards (Glowing Borders) */
    div[data-testid="metric-container"] {
        background-color: #21262D;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    /* Success/Error Colors (Neon) */
    .st-emotion-cache-1wivap2 { color: #00FF9D !important; } /* Green */
    .st-emotion-cache-16idsys p { color: #FF4B4B !important; } /* Red */
    
    /* Headers */
    h1, h2, h3 { font-family: 'Courier New', monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('eswatini_insurance_final_dataset (5).csv')
    except:
        return None

    df['Risk_Target'] = (df['investigation_outcome'] == 'Confirmed Fraud').astype(int)
    
    # Fill Dynamic Fields
    fill_cols = ['accident_location', 'cause_of_fire', 'livestock_type_affected']
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna("N/A")

    # Coordinates
    coords = {
        "Manzini": [-26.50, 31.36], "Mbabane": [-26.31, 31.13], "Siteki": [-26.45, 31.95],
        "Big Bend": [-26.81, 31.93], "Lobamba": [-26.46, 31.20], "Piggs Peak": [-25.96, 31.25],
        "Nhlangano": [-27.11, 31.20], "Simunye": [-26.21, 31.91]
    }
    df['lat'] = df['location'].map(lambda x: coords.get(x, [None, None])[0])
    df['lon'] = df['location'].map(lambda x: coords.get(x, [None, None])[1])
    
    return df

# --- 3. MODEL TRAINING ---
@st.cache_resource
def train_model(df):
    feature_cols = ['claim_type', 'location', 'claim_amount_SZL', 'rural_vs_urban', 
                    'policy_premium_SZL', 'policy_maturity_years',
                    'accident_location', 'cause_of_fire', 'livestock_type_affected']
    X = df[feature_cols].copy()
    y = df['Risk_Target']
    
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    model = RandomForestClassifier(n_estimators=80, max_depth=10, class_weight='balanced', random_state=42)
    model.fit(X, y)
    
    return model, encoders

# --- HELPER: RADAR CHART DATA ---
def get_radar_data(df, input_row):
    scaler = MinMaxScaler()
    cols = ['claim_amount_SZL', 'policy_premium_SZL', 'policy_maturity_years', 'age']
    avg_fraud = df[df['Risk_Target'] == 1][cols].mean()
    avg_legit = df[df['Risk_Target'] == 0][cols].mean()
    max_vals = df[cols].max()
    
    return {
        'categories': ['Claim Amount', 'Premium', 'Policy Age', 'Driver Age'],
        'Current': [input_row[c]/max_vals[c] for c in cols],
        'Fraud_Avg': [avg_fraud[c]/max_vals[c] for c in cols],
        'Legit_Avg': [avg_legit[c]/max_vals[c] for c in cols]
    }

# --- 4. MAIN APP ---
def main():
    # LIVE TICKER (Simulated)
    ticker_ph = st.empty()
    
    st.title("🛡️ KEAS: AI Sentinel v2.0")
    st.markdown("**Real-Time Fraud Detection System** | *Live Monitoring Active*")
    
    df = load_data()
    if df is None: st.error("Data not found."); return
    model, encoders = train_model(df)

    # --- LIVE TICKER LOGIC (Runs once on load to simulate activity) ---
    # In a real app, this would be a websocket. Here, we mock it for the "Wow" factor.
    recent_claim = df.sample(1).iloc[0]
    status = "🔴 CRITICAL" if recent_claim['Risk_Target'] == 1 else "🟢 VERIFIED"
    ticker_html = f"""
    <div style='background-color: #21262D; padding: 10px; border-radius: 5px; border-left: 5px solid #00FF9D; font-family: monospace;'>
        📡 <b>LIVE FEED:</b> Incoming Claim from <b>{recent_claim['location']}</b> | Type: {recent_claim['claim_type']} | Amount: SZL {recent_claim['claim_amount_SZL']:,.2f} | AI Status: {status}
    </div>
    """
    st.markdown(ticker_html, unsafe_allow_html=True)
    st.markdown("---")

    # --- SIDEBAR (INTERACTIVE SLIDERS) ---
    with st.sidebar:
        st.header("🎛️ Simulation Console")
        st.caption("Adjust values to test AI sensitivity.")
        
        c_type = st.selectbox("Claim Type", sorted(df['claim_type'].unique()))
        loc = st.selectbox("Region", sorted(df['location'].unique()))
        
        # SLIDERS FOR INSTANT FEEDBACK
        amt = st.slider("Claim Amount (SZL)", 0.0, 150000.0, 50000.0, step=1000.0)
        premium = st.slider("Premium (SZL)", 200.0, 15000.0, 5000.0)
        maturity = st.slider("Policy Maturity (Years)", 0, 30, 5)
        age = st.slider("Driver Age", 18, 80, 35)
        
        # Context Fields
        acc_loc = "N/A"
        fire_cause = "N/A"
        if c_type == 'road accident':
            acc_loc = st.selectbox("Accident Context", [x for x in df['accident_location'].unique() if x != "N/A"])
        elif c_type == 'fire damage':
            fire_cause = st.selectbox("Fire Cause", [x for x in df['cause_of_fire'].unique() if x != "N/A"])
        
        # "Auto-Run" (We remove the submit button for 'Real-Time' feel in this mode)
        # The app will rerun every time a slider moves.

    # --- PREDICTION ENGINE (ALWAYS ON) ---
    input_dict = {
        'claim_type': c_type, 'location': loc, 'claim_amount_SZL': amt,
        'rural_vs_urban': 'urban', 'policy_premium_SZL': premium, 
        'policy_maturity_years': maturity, 'accident_location': acc_loc,
        'cause_of_fire': fire_cause, 'livestock_type_affected': 'N/A', 'age': age
    }
    
    # Encode & Predict
    input_df = pd.DataFrame([input_dict])
    enc_df = input_df.copy().drop('age', axis=1)
    for col, le in encoders.items():
        val = str(enc_df[col].iloc[0])
        enc_df[col] = le.transform([val if val in le.classes_ else le.classes_[0]])
    
    prob = model.predict_proba(enc_df)[0][1]

    # --- LAYOUT: 3 COLUMNS (DASHBOARD STYLE) ---
    col1, col2, col3 = st.columns([1.2, 2, 1.2])

    # COLUMN 1: THE GAUGE (Big & Bold)
    with col1:
        st.markdown("### 🚨 Threat Level")
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            title = {'text': "Fraud Probability"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#FF4B4B" if prob > 0.5 else "#00FF9D"},
                'bgcolor': "#21262D",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#21262D'},
                    {'range': [50, 100], 'color': '#30363D'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85}}))
        fig_gauge.update_layout(height=300, margin=dict(l=10,r=10,t=50,b=10), paper_bgcolor="#0E1117", font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if prob > 0.5:
            st.error("RECOMMENDATION: AUDIT")
        else:
            st.success("RECOMMENDATION: APPROVE")

    # COLUMN 2: THE MAP (3D & Interactive)
    with col2:
        st.markdown("### 🌐 Geospatial Intelligence")
        # 3D Map
        layer = pdk.Layer(
            "HexagonLayer",
            df[['lon', 'lat']],
            get_position=['lon', 'lat'],
            auto_highlight=True,
            elevation_scale=50,
            pickable=True,
            elevation_range=[0, 3000],
            extruded=True,
            coverage=1
        )
        view_state = pdk.ViewState(longitude=31.465, latitude=-26.52, zoom=8, pitch=45)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Density"}))

    # COLUMN 3: THE RADAR (Fingerprint)
    with col3:
        st.markdown("### 🧬 Risk DNA")
        radar_data = get_radar_data(df, input_dict)
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_data['Fraud_Avg'], theta=radar_data['categories'],
            fill='toself', name='Fraud Pattern', line_color='#FF4B4B', opacity=0.5
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_data['Current'], theta=radar_data['categories'],
            fill='toself', name='Current Claim', line_color='#00FF9D', opacity=0.5
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, showticklabels=False), bgcolor='#21262D'),
            paper_bgcolor="#0E1117", font={'color': "white"}, height=300, showlegend=False,
            margin=dict(l=30,r=30,t=30,b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Green Shape vs Red Shape (Fraud Profile)")

    st.divider()

    # --- BOTTOM SECTION: EXPLAINABILITY ---
    st.subheader("🔍 Forensic Breakdown (SHAP)")
    col_shap, col_peers = st.columns([2, 1])
    
    with col_shap:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(enc_df)
        vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals[..., 1]
        
        # Custom Colors for SHAP (Neon)
        fig_shap, ax = plt.subplots(figsize=(10, 3))
        fig_shap.patch.set_facecolor('#0E1117') # Dark background
        ax.set_facecolor('#0E1117')
        
        # We can't easily style the waterfall strictly with matplotlib args, but we can trust the default
        # or stick to the standard plot. For "Stunning," let's keep it simple but dark.
        shap.plots.waterfall(shap.Explanation(vals[0], explainer.expected_value[1], input_df.drop('age', axis=1).iloc[0]), max_display=7, show=False)
        st.pyplot(fig_shap) # SHAP's native plot is hard to colorize fully without hacking, but dark bg helps.

    with col_peers:
        st.markdown("**Peer Comparison**")
        # Simple Neon Bar Chart
        peer_data = df[df['claim_type'] == c_type]
        avg_peer_amt = peer_data['claim_amount_SZL'].mean()
        
        fig_bar = go.Figure([go.Bar(
            x=['Peer Avg', 'Current Claim'], 
            y=[avg_peer_amt, amt],
            marker_color=['#30363D', '#00FF9D' if amt < avg_peer_amt * 1.2 else '#FF4B4B']
        )])
        fig_bar.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font={'color': "white"}, height=250)
        st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()
