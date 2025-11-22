import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="KEAS: AI Sentinel Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    div[data-testid="metric-container"] {
        background-color: #21262D; border: 1px solid #30363D; padding: 15px; border-radius: 10px;
        box-shadow: 0 0 10px rgba(79, 139, 249, 0.1);
    }
    .stButton>button { width: 100%; font-weight: bold; border-radius: 5px; }
    h1, h2, h3 { font-family: 'Courier New', monospace; letter-spacing: -1px; }
    
    /* Ticker Animation */
    @keyframes ticker {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    .ticker-wrap {
        width: 100%; overflow: hidden; background-color: #21262D; padding: 10px; border-bottom: 1px solid #30363D;
    }
    .ticker {
        display: inline-block; white-space: nowrap; animation: ticker 30s linear infinite; color: #00FF9D; font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    files = ['eswatini_insurance_final_dataset (5).csv', 'eswatini_insurance_final_dataset.csv']
    df = None
    for f in files:
        try:
            df = pd.read_csv(f)
            break
        except:
            continue
    if df is None: return None

    if 'investigation_outcome' in df.columns:
        df['Risk_Target'] = (df['investigation_outcome'] == 'Confirmed Fraud').astype(int)
    else:
        df['Risk_Target'] = (df['claim_amount_SZL'] >= df['claim_amount_SZL'].quantile(0.75)).astype(int)

    fill_cols = ['accident_location', 'cause_of_fire', 'livestock_type_affected']
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna("N/A")

    coords = {
        "Manzini": [-26.50, 31.36], "Mbabane": [-26.31, 31.13], "Siteki": [-26.45, 31.95],
        "Big Bend": [-26.81, 31.93], "Lobamba": [-26.46, 31.20], "Piggs Peak": [-25.96, 31.25],
        "Nhlangano": [-27.11, 31.20], "Simunye": [-26.21, 31.91]
    }
    df['lat'] = df['location'].map(lambda x: coords.get(x, [None, None])[0])
    df['lon'] = df['location'].map(lambda x: coords.get(x, [None, None])[1])
    df['lat'] += np.random.normal(0, 0.02, len(df))
    df['lon'] += np.random.normal(0, 0.02, len(df))
    
    return df

# --- 3. MODELING ---
@st.cache_resource
def train_models(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df_numeric = df[numeric_cols].fillna(0)
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['segment'] = kmeans.fit_predict(df_numeric).astype(str)
    
    feature_cols = ['claim_type', 'location', 'claim_amount_SZL', 'rural_vs_urban', 
                    'policy_premium_SZL', 'policy_maturity_years',
                    'accident_location', 'cause_of_fire', 'segment']
    
    X = df[feature_cols].copy()
    y = df['Risk_Target']
    
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42)
    model.fit(X, y)
    return model, encoders, kmeans

def get_radar_data(df, input_row):
    cols = ['claim_amount_SZL', 'policy_premium_SZL', 'policy_maturity_years']
    max_vals = df[cols].max()
    avg_fraud = df[df['Risk_Target'] == 1][cols].mean()
    return {
        'categories': ['Claim Amount', 'Premium', 'Policy Age'],
        'Current': [input_row[c]/max_vals[c] for c in cols],
        'Fraud_Avg': [avg_fraud[c]/max_vals[c] for c in cols]
    }

# --- 4. GAME STATE ---
if 'game_score' not in st.session_state:
    st.session_state.game_score = {'human': 0, 'ai': 0, 'rounds': 0}
if 'current_case' not in st.session_state:
    st.session_state.current_case = None
def new_game_round(df):
    st.session_state.current_case = df.sample(1).iloc[0]

# --- 5. MAIN APP ---
def main():
    df = load_data()
    if df is None: st.error("Data not found"); return
    model, encoders, kmeans = train_models(df)
    
    # --- GLOBAL STATE FOR SCENARIO ---
    if 'scenario_mode' not in st.session_state:
        st.session_state.scenario_mode = "None (Baseline)"

    # --- DYNAMIC NEWS TICKER (The "Current Events" Feel) ---
    news_text = "📡 SYSTEM STATUS: All Systems Nominal | Live Feed Active | Monitoring 8 Regions..."
    if st.session_state.scenario_mode == "Pandemic (Viral Outbreak)":
        news_text = "⚠️ BREAKING: Health Ministry reports surge in viral cases | Hospitals at 85% Capacity | Motor traffic down 40%..."
    elif st.session_state.scenario_mode == "Civil Unrest":
        news_text = "🚨 ALERT: Security incidents reported in Manzini & Mbabane | Business interruptions likely | Fire risk ELEVATED..."
    elif st.session_state.scenario_mode == "Economic Crash (Inflation)":
        news_text = "📉 MARKET UPDATE: Inflation hits 12.5% | Auto parts import costs rising | Policy lapse risk increasing..."

    st.markdown(f"""
    <div class="ticker-wrap">
    <div class="ticker">{news_text} &nbsp;&nbsp;&nbsp;&nbsp; /// &nbsp;&nbsp;&nbsp;&nbsp; {news_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # TITLE
    st.title("🛡️ KEAS: AI Sentinel Command Center")
    st.markdown("**Real-Time Intelligent Risk & Catastrophe Engine**")

    # --- GLOBAL SIMULATOR ---
    st.markdown("---")
    with st.expander("🎛️ SINGLE CLAIM SIMULATOR (Click to Expand)", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        sim_type = c1.selectbox("Claim Type", sorted(df['claim_type'].unique()))
        sim_loc = c2.selectbox("Location", sorted(df['location'].unique()))
        sim_amt = c3.slider("Claim Amount (SZL)", 0, 150000, 50000, step=1000)
        sim_prem = c4.slider("Premium", 500, 20000, 7500)
        sim_age = c5.slider("Age", 18, 90, 35)
        
        sim_ctx = "N/A"
        if sim_type == 'road accident':
            sim_ctx = st.selectbox("Accident Context", [x for x in df['accident_location'].unique() if x != "N/A"])
    
    input_data = {
        'claim_type': sim_type, 'location': sim_loc, 'claim_amount_SZL': sim_amt,
        'rural_vs_urban': 'urban', 'policy_premium_SZL': sim_prem, 
        'policy_maturity_years': 5, 'accident_location': sim_ctx,
        'cause_of_fire': 'N/A', 'segment': '0'
    }
    input_df = pd.DataFrame([input_data])
    enc_df = input_df.copy()
    for col, le in encoders.items():
        val = str(enc_df[col].iloc[0])
        enc_df[col] = le.transform([val if val in le.classes_ else le.classes_[0]])
    base_prob = model.predict_proba(enc_df)[0][1]

    # --- TABS (ALL 6 INCLUDED) ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚦 Risk Engine", "🌐 Strategic Map", "💰 Policy Lab", "🎮 Human vs AI", "🔮 Crystal Ball", "📈 Future Horizons"
    ])

    # TAB 1: RISK ENGINE
    with tab1:
        c_gauge, c_xai = st.columns([1, 2])
        with c_gauge:
            st.subheader("Threat Level")
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=base_prob*100, title={'text': "Risk"}, gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#FF4B4B" if base_prob > 0.5 else "#00FF9D"}, 'bgcolor': "#21262D"}))
            fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="#0E1117", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)
        with c_xai:
            st.subheader("Explainability (SHAP)")
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(enc_df)
            vals = vals[1] if isinstance(vals, list) else vals[..., 1]
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 3))
            shap.plots.waterfall(shap.Explanation(vals[0], explainer.expected_value[1], input_df.iloc[0]), max_display=6, show=False)
            st.pyplot(fig)

    # TAB 2: STRATEGIC MAP
    with tab2:
        st.subheader("3D Risk Density")
        layer = pdk.Layer("HexagonLayer", df[['lon', 'lat']], get_position=['lon', 'lat'], auto_highlight=True, elevation_scale=50, pickable=True, elevation_range=[0, 3000], extruded=True, coverage=1)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(longitude=31.465, latitude=-26.52, zoom=8.2, pitch=55)))

    # TAB 3: POLICY LAB
    with tab3:
        st.header("Policy Optimization")
        p1, p2, p3 = st.columns(3)
        p1.metric("Recommended Premium", f"SZL {sim_prem * (1 + (base_prob-0.2)*0.5):,.2f}")
        p2.metric("Retention Score", "High" if base_prob < 0.3 else "Low")
        p3.info("Cross-Sell: " + ("Funeral Cover" if sim_age > 50 else "Gap Cover"))

    # TAB 4: HUMAN VS AI
    with tab4:
        st.header("🎮 Challenge Mode")
        if st.button("New Case"): new_game_round(df)
        if st.session_state.current_case is not None:
            case = st.session_state.current_case
            st.write(f"**Case:** {case['claim_type']} in {case['location']} | Amount: {case['claim_amount_SZL']}")
            c1, c2 = st.columns(2)
            if c1.button("Approve"): 
                res = "Correct" if case['Risk_Target'] == 0 else "Wrong"
                st.write(res)
            if c2.button("Reject"):
                res = "Correct" if case['Risk_Target'] == 1 else "Wrong"
                st.write(res)

    # TAB 5: CRYSTAL BALL (MACRO-STRESS TEST)
    with tab5:
        st.header("🔮 Macro-Economic Stress Test")
        col_scen, col_viz = st.columns([1, 2])
        with col_scen:
            # Update session state based on selection
            scen = st.selectbox("Select Crisis Scenario", 
                ["None (Baseline)", "Pandemic (Viral Outbreak)", "Economic Crash (Inflation)", "Civil Unrest"],
                key="scenario_select")
            st.session_state.scenario_mode = scen # Update global state for ticker
            
            severity = st.slider("Severity", 1, 10, 5)
            st.info("👆 Changing this updates the News Ticker above!")

        with col_viz:
            sim_df = df.copy()
            total_loss_baseline = sim_df[sim_df['Risk_Target'] == 1]['claim_amount_SZL'].sum()
            
            if scen == "Pandemic (Viral Outbreak)":
                mask_health = sim_df['claim_type'].isin(['health emergency', 'livestock loss'])
                mask_motor = sim_df['claim_type'] == 'road accident'
                sim_df.loc[mask_health, 'claim_amount_SZL'] *= (1 + severity * 0.2)
                sim_df.loc[mask_motor, 'claim_amount_SZL'] *= (1 - severity * 0.08)
            elif scen == "Economic Crash (Inflation)":
                sim_df['claim_amount_SZL'] *= (1 + severity * 0.05)
            elif scen == "Civil Unrest":
                mask_unrest = (sim_df['claim_type'].isin(['fire damage', 'theft'])) & (sim_df['location'].isin(['Manzini', 'Mbabane']))
                sim_df.loc[mask_unrest, 'claim_amount_SZL'] *= (1 + severity * 0.4)

            total_loss_scenario = sim_df[sim_df['Risk_Target'] == 1]['claim_amount_SZL'].sum()
            
            fig_stress = px.bar(
                x=['Baseline', 'Simulated Crisis'], 
                y=[total_loss_baseline, total_loss_scenario], 
                color=['Baseline', 'Simulated Crisis'],
                color_discrete_sequence=['#00FF9D', '#FF4B4B']
            )
            fig_stress.update_layout(height=300, paper_bgcolor="#0E1117", font={'color': "white"})
            st.plotly_chart(fig_stress, use_container_width=True)
            
            diff = total_loss_scenario - total_loss_baseline
            st.metric("Net Capital Impact", f"SZL {diff:,.0f}", f"{(diff/total_loss_baseline)*100:.1f}% Increase", delta_color="inverse")

    # TAB 6: FUTURE HORIZONS (FORECASTING)
    with tab6:
        st.header("📈 Future Claims Horizon (2026-2027)")
        
        # Dummy Time Series Generation
        months_history = 24
        base_volume = len(df) / months_history
        dates = pd.date_range(end=pd.Timestamp.now(), periods=months_history, freq='M')
        trend_factor = np.linspace(1, 1.5, months_history) 
        historical_claims = base_volume * trend_factor * np.random.normal(1, 0.1, months_history)
        history_df = pd.DataFrame({'Date': dates, 'Claims': historical_claims, 'Type': 'Historical'})
        
        # Forecast
        X_time = np.array(range(len(history_df))).reshape(-1, 1)
        y_count = history_df['Claims'].values
        forecaster = LinearRegression()
        forecaster.fit(X_time, y_count)
        
        future_months = 24
        X_future = np.array(range(len(history_df), len(history_df) + future_months)).reshape(-1, 1)
        future_pred = forecaster.predict(X_future)
        future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=future_months, freq='M')
        future_df = pd.DataFrame({'Date': future_dates, 'Claims': future_pred, 'Type': 'Forecast'})
        
        full_trend = pd.concat([history_df, future_df])
        
        c_chart, c_metric = st.columns([2, 1])
        with c_chart:
            fig_forecast = px.line(full_trend, x='Date', y='Claims', color='Type', color_discrete_map={'Historical': '#00FF9D', 'Forecast': '#FF4B4B'})
            fig_forecast.add_hline(y=max(historical_claims)*1.2, line_dash="dot", line_color="white", annotation_text="Capacity Limit")
            fig_forecast.update_layout(height=350, paper_bgcolor="#0E1117", font={'color': "white"})
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        with c_metric:
            growth_pct = ((np.mean(future_pred[-3:]) - np.mean(historical_claims[-3:])) / np.mean(historical_claims[-3:])) * 100
            st.metric("Projected 2-Year Growth", f"+{growth_pct:.1f}%", "Action Required", delta_color="inverse")
            if growth_pct > 20:
                st.warning("⚠️ **Capacity Warning:** Volume will exceed staff limits by Nov 2026.")

if __name__ == "__main__":
    main()
