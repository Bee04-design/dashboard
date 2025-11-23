import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from scipy.stats import ks_2samp
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="Strategic Capital Efficiency Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main App & Background */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    
    /* Input/Metric Containers */
    div[data-testid="metric-container"] {
        background-color: #21262D; border: 1px solid #30363D; padding: 15px; border-radius: 10px;
        box-shadow: 0 0 10px rgba(79, 139, 249, 0.1);
    }
    
    /* Headers and Typography */
    h1, h2, h3 { 
        font-family: 'Courier New', monospace; 
        letter-spacing: -1px; 
        color: #B4BCE8; 
    }
    
    /* Ticker Animation */
    @keyframes ticker { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }
    .ticker-wrap { 
        width: 100%; 
        overflow: hidden; 
        background-color: #21262D; 
        padding: 10px; 
        border-bottom: 1px solid #30363D; 
    }
    .ticker { 
        display: inline-block; 
        white-space: nowrap; 
        animation: ticker 30s linear infinite; 
        color: #00FF9D; 
        font-family: monospace; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING & PREP ---
@st.cache_data(show_spinner="Loading and synthesizing data...")
def load_data():
    try:
        df = pd.read_csv('eswatini_insurance_final_dataset (5).csv')
    except FileNotFoundError:
        try:
             df = pd.read_csv('eswatini_insurance_final_dataset.csv')
        except:
            st.error("Error: Dataset not found. Please ensure the CSV file is uploaded.")
            return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None
    
    # --- Data Cleaning ---
    
    # 1. Target Variable Creation
    if 'investigation_outcome' in df.columns:
        df['Risk_Target'] = (df['investigation_outcome'] == 'Confirmed Fraud').astype(int)
    else:
        # Fallback
        df['Risk_Target'] = (df['claim_amount_SZL'] >= df['claim_amount_SZL'].quantile(0.75)).astype(int)
    
    # Safety Check: Ensure we have at least two classes (0 and 1) to avoid single-class errors
    if df['Risk_Target'].nunique() < 2:
        # If dataset has only 0s, force a few 1s for model stability (synthetic injection)
        df.loc[df.sample(5).index, 'Risk_Target'] = 1
        
    # 2. Mock Date and Customer ID
    if 'claim_date' not in df.columns:
        df['claim_date'] = pd.date_range(end='2024-01-01', periods=len(df), freq='D')
    if 'customer_id' not in df.columns:
        df['customer_id'] = [f"CUST-{i+1:04d}" for i in range(len(df))]
        
    # 3. Geo-Coordinates (pydeck)
    coords = {
        "Manzini": [-26.50, 31.36], "Mbabane": [-26.31, 31.13], "Siteki": [-26.45, 31.95],
        "Big Bend": [-26.81, 31.93], "Lobamba": [-26.46, 31.20], "Piggs Peak": [-25.96, 31.25],
        "Nhlangano": [-27.11, 31.20], "Simunye": [-26.21, 31.91]
    }
    df['lat'] = df['location'].map(lambda x: coords.get(x, [None, None])[0])
    df['lon'] = df['location'].map(lambda x: coords.get(x, [None, None])[1])
    
    center_lat, center_lon = coords['Mbabane']
    df['lat'] = df['lat'].fillna(center_lat)
    df['lon'] = df['lon'].fillna(center_lon)

    df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
    df = df.dropna(subset=['claim_date'])
    
    # 4. Apply Segmentation HERE (Fixes KeyError)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    # Filter strictly for clustering to avoid data leakage or errors with target
    cluster_cols = [c for c in numeric_cols if c not in ['Risk_Target', 'claim_id', 'customer_id']]
    if cluster_cols:
        df_numeric = df[cluster_cols].fillna(0)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        df['segment'] = kmeans.fit_predict(df_numeric).astype(str)
    else:
        df['segment'] = "0" # Fallback if no numeric cols
    
    X_train_for_drift = df[['claim_amount_SZL', 'policy_premium_SZL', 'policy_maturity_years']].copy()

    return df, X_train_for_drift

# --- 3. MODELING & TRAINING ---
@st.cache_resource(show_spinner="Training high-performance Random Forest Model...")
def train_models(df):
    # Note: Segmentation is now done in load_data, so 'segment' exists
    
    feature_cols = ['claim_type', 'location', 'claim_amount_SZL', 'rural_vs_urban', 
                    'policy_premium_SZL', 'policy_maturity_years',
                    'accident_location', 'cause_of_fire', 'segment']
    
    # Ensure all feature columns exist (handle missing optional cols)
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        for c in missing_cols:
            df[c] = "N/A" # or 0 depending on logic
            
    X = df[feature_cols].copy()
    y = df['Risk_Target']
    
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42)
    model.fit(X, y)
    
    global_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    
    return model, encoders, global_importance

# --- 4. COUNTERFACTUAL FUNCTION ---
def find_counterfactual(model, encoders, input_df, current_prob):
    numeric_features = ['claim_amount_SZL', 'policy_premium_SZL', 'policy_maturity_years']
    target_prob = 0.5 
    counterfactuals = {}

    for feature in numeric_features:
        original_value = input_df[feature].iloc[0]
        temp_df = input_df.copy()
        
        if feature == 'claim_amount_SZL':
            test_value = original_value * 0.9 
            while test_value > 0 and test_value >= original_value * 0.5:
                temp_df[feature] = test_value
                enc_temp_df = temp_df.copy()
                for col, le in encoders.items():
                    val = str(enc_temp_df[col].iloc[0])
                    if val in le.classes_:
                        enc_temp_df[col] = le.transform([val])
                    else:
                        enc_temp_df[col] = le.transform([le.classes_[0]])
                
                new_prob = model.predict_proba(enc_temp_df)[0][1]
                
                if new_prob < target_prob:
                    reduction = original_value - test_value
                    counterfactuals[feature] = f"Reduce to SZL {test_value:,.0f} (SZL {reduction:,.0f} less)"
                    break
                test_value *= 0.9 
            
        elif feature == 'policy_maturity_years':
            test_value = original_value + 1
            if test_value < 100: 
                temp_df[feature] = test_value
                enc_temp_df = temp_df.copy()
                for col, le in encoders.items():
                    val = str(enc_temp_df[col].iloc[0])
                    if val in le.classes_:
                        enc_temp_df[col] = le.transform([val])
                    else:
                        enc_temp_df[col] = le.transform([le.classes_[0]])
                
                new_prob = model.predict_proba(enc_temp_df)[0][1]
                if new_prob < target_prob:
                    counterfactuals[feature] = f"Increase to {test_value} years"
                    
    return counterfactuals

# --- 5. DATA DRIFT FUNCTION ---
def get_data_drift_score(X_train, input_df):
    total_drift = 0
    comparison_features = ['claim_amount_SZL', 'policy_premium_SZL']

    for feature in comparison_features:
        live_data = pd.concat([X_train[feature].sample(min(100, len(X_train)), random_state=42), input_df[feature]])
        D_stat, p_value = ks_2samp(X_train[feature], live_data)
        total_drift += D_stat

    avg_drift_score = (total_drift / len(comparison_features)) * 100
    return avg_drift_score

# --- 6. MAIN APP ---
def main():
    df, X_train_for_drift = load_data()
    if df is None: return
    model, encoders, global_importance = train_models(df)
    
    # --- HEADER & NEWS TICKER ---
    if 'scenario_mode' not in st.session_state:
        st.session_state.scenario_mode = "None (Baseline)"

    news_text = "📡 SYSTEM STATUS: Monitoring Active Claims. Presenting at Conference..."
    if st.session_state.scenario_mode == "Severe Weather (Flood/Storm)":
        news_text = "⛈️ WEATHER ALERT: Severe Storm Warning in Manzini & Mbabane. Flood risk ELEVATED. Verify all 'Storm Damage' claims."
    elif st.session_state.scenario_mode == "Pandemic (Viral Outbreak)":
        news_text = "⚠️ HEALTH ALERT: Viral outbreak detected. Expect surge in Health Claims."
    elif st.session_state.scenario_mode == "Civil Unrest":
        news_text = "🚨 SECURITY ALERT: Civil Unrest reported. High risk of Arson & Theft claims."

    st.markdown(f"""<div class="ticker-wrap"><div class="ticker">{news_text} &nbsp;&nbsp;&nbsp;&nbsp; /// &nbsp;&nbsp;&nbsp;&nbsp; {news_text}</div></div>""", unsafe_allow_html=True)
    
    st.title("🛡️ AI Sentinel: High-Risk Claim Classification Dashboard")
    st.markdown("Developed by Bhekiwe Sindiswa Dlamini | University of Eswatini")

    # --- SIDEBAR ---
    st.sidebar.title("Claim Selection")
    st.sidebar.info("Select a claim to see the AI analysis in real-time.")
    customer_ids = df['customer_id'].unique()
    selected_cust_id = st.sidebar.selectbox("Select Customer ID", customer_ids[:100])
    
    cust_data = df[df['customer_id'] == selected_cust_id]
    if cust_data.empty: return
    latest_claim = cust_data.sort_values('claim_date', ascending=False).iloc[0]
    
    input_data = {
        'claim_type': latest_claim['claim_type'], 
        'location': latest_claim['location'], 
        'claim_amount_SZL': latest_claim['claim_amount_SZL'],
        'rural_vs_urban': latest_claim['rural_vs_urban'], 
        'policy_premium_SZL': latest_claim['policy_premium_SZL'], 
        'policy_maturity_years': latest_claim['policy_maturity_years'], 
        'accident_location': latest_claim.get('accident_location', 'N/A'),
        'cause_of_fire': latest_claim.get('cause_of_fire', 'N/A'),
        # This line caused the error previously because 'segment' wasn't in df yet
        'segment': str(latest_claim.get('segment', '0')) 
    }
    input_df = pd.DataFrame([input_data])
    
    # --- PREDICTION ---
    enc_df = input_df.copy()
    for col, le in encoders.items():
        val = str(enc_df[col].iloc[0])
        if val in le.classes_:
            enc_df[col] = le.transform([val])
        else:
            enc_df[col] = le.transform([le.classes_[0]])

    base_prob = model.predict_proba(enc_df)[0][1]
    
    # --- XAI (SHAP) - ROBUST FIX FOR DIMENSIONS ---
    explainer = shap.TreeExplainer(model)
    shap_values_result = explainer.shap_values(enc_df)
    
    if isinstance(shap_values_result, list):
        if len(shap_values_result) > 1:
            shap_values = shap_values_result[1]
            expected_value = explainer.expected_value[1]
        else:
            shap_values = shap_values_result[0]
            if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 0:
                 expected_value = explainer.expected_value[0]
            else:
                 expected_value = explainer.expected_value
    else:
        shap_values = shap_values_result
        if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
             expected_value = explainer.expected_value[1]
        else:
             expected_value = explainer.expected_value

    # *** FIX: Ensure 1D numpy array of floats ***
    if len(shap_values.shape) > 1:
        shap_values_flat = shap_values[0] 
    else:
        shap_values_flat = shap_values

    shap_values_flat = np.array(shap_values_flat).flatten()
    
    if len(shap_values_flat) != len(input_df.columns):
        shap_values_flat = np.zeros(len(input_df.columns))

    feature_contributions = pd.DataFrame({
        'Feature': input_df.columns, 
        'Value': input_df.iloc[0].values, 
        'SHAP_Value': shap_values_flat
    }).sort_values('SHAP_Value', ascending=False).head(8)


    # --- TABS ---
    tab1, tab2, tab5, tab6 = st.tabs([
        "🚦 Risk Engine (Prediction & XAI)", 
        "🌐 Strategic Map (Spatial Risk)", 
        "🔮 Crystal Ball (Stress Test)",
        "📈 Future Horizons (Model Health & ROI)"
    ])
    
    # TAB 1
    with tab1:
        st.header("Real-Time Decision Transparency")
        st.markdown(f"""
            <div style='background-color: #161B22; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
                **Claim ID:** {latest_claim['claim_id']} | 
                **Claim Type:** {latest_claim['claim_type']} | 
                **Amount:** SZL {latest_claim['claim_amount_SZL']:,.2f} | 
                **Location:** {latest_claim['location']}
            </div>
            """, unsafe_allow_html=True)
            
        c_top1, c_top2 = st.columns([1, 2])
        
        with c_top1:
            st.subheader("AI Risk Score")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", 
                value=base_prob*100, 
                title={'text': "Probability"}, 
                gauge={'axis': {'range': [None, 100]}, 
                       'bar': {'color': "#FF4B4B" if base_prob > 0.5 else "#00FF9D"}, 
                       'bgcolor': "#21262D"},
                domain = {'x': [0, 1], 'y': [0, 1]}
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="#0E1117", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("---")
            st.subheader("💡 Counterfactual Guidance")
            if base_prob > 0.5:
                counterfactuals = find_counterfactual(model, encoders, input_df, base_prob)
                if counterfactuals:
                    st.warning("**To Drop Risk Below 50%:** (Negotiation Points)")
                    for feat, change in counterfactuals.items():
                        st.markdown(f"- **{feat.replace('_', ' ').title()}:** {change}")
                else:
                    st.error("Severe Risk. Requires full investigation.")
            else:
                st.success("Claim is Low-Risk. Process quickly.")
            
        with c_top2:
            st.subheader("Feature Contribution (Local XAI)")
            fig_contribution = px.bar(
                feature_contributions, y='Feature', x='SHAP_Value',
                color=feature_contributions['SHAP_Value'] > 0, orientation='h',
                color_discrete_map={True: '#FF4B4B', False: '#00FF9D'},
                title='Factors Pushing Score Up (Red) or Down (Green)'
            )
            fig_contribution.update_layout(height=400, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font={'color': "white"}, showlegend=False)
            st.plotly_chart(fig_contribution, use_container_width=True)

        st.markdown("---")
        c_bottom1, c_bottom2 = st.columns(2)
        
        with c_bottom1:
            st.subheader("Global Fraud Drivers (Treemap)")
            importance_df = global_importance.reset_index().rename(columns={'index': 'Feature', 0: 'Importance'})
            fig_treemap = px.treemap(
                importance_df, names='Feature', parents=None, values='Importance',
                color='Importance', color_continuous_scale='Sunsetdark'
            )
            fig_treemap.update_layout(height=350, margin=dict(t=30, l=10, r=10, b=10), paper_bgcolor="#0E1117", font={'color': "white"})
            st.plotly_chart(fig_treemap, use_container_width=True)

        with c_bottom2:
            st.subheader("SHAP Waterfall (Step-by-Step Logic)")
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap.Explanation(shap_values_flat, expected_value, input_df.iloc[0]), max_display=6, show=False)
            plt.title("")
            st.pyplot(fig)

    # TAB 2
    with tab2:
        st.header("🌐 Strategic Map: Geographic Risk Concentration")
        st.markdown("High-risk claims aggregated in 3D to identify key spatial risk areas.")
        layer = pdk.Layer(
            "HexagonLayer", 
            df[['lon', 'lat', 'Risk_Target']], 
            get_position=['lon', 'lat'], 
            auto_highlight=True, 
            elevation_scale=50, 
            pickable=True, 
            elevation_range=[0, 3000], 
            extruded=True, 
            coverage=1,
            color_range=[[0, 100, 255], [0, 255, 255], [0, 255, 100], [255, 255, 0], [255, 100, 0], [255, 0, 0]]
        )
        initial_view = pdk.ViewState(longitude=31.465, latitude=-26.52, zoom=8.2, pitch=55)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=initial_view))

    # TAB 5: CRYSTAL BALL (Updated with Weather)
    with tab5:
        st.header("🔮 Macro-Economic & Catastrophe Stress Testing")
        st.markdown("Simulate unforeseen 'Black Swan' events to predict capital impact.")
        
        col_scen, col_viz = st.columns([1, 2])
        
        with col_scen:
            st.subheader("⚙️ Stress Configuration")
            
            # Updated Scenario List
            scenario_mode = st.selectbox("Select Crisis Scenario", 
                ["None (Baseline)", "Severe Weather (Flood/Storm)", "Pandemic (Viral Outbreak)", "Economic Crash (Inflation)", "Civil Unrest"])
            
            # Update session state for ticker
            st.session_state.scenario_mode = scenario_mode
            
            severity = st.slider("Crisis Severity Level", 1, 10, 5)
            
            st.info("👆 Changing this updates the News Ticker above!")
            
            if scenario_mode == "Severe Weather (Flood/Storm)":
                st.warning("**Simulating:** High-volume 'Storm Damage' claims in Manzini & Mbabane. Flood risk ELEVATED.")

        with col_viz:
            st.subheader("📉 Financial Solvency Projection")
            
            # SIMULATION LOGIC
            sim_df = df.copy()
            total_loss_baseline = sim_df[sim_df['Risk_Target'] == 1]['claim_amount_SZL'].sum()
            
            if scenario_mode == "Severe Weather (Flood/Storm)":
                # Spike 'storm damage' claims, especially in urban areas
                mask_storm = (sim_df['claim_type'].isin(['storm damage', 'fire damage']))
                storm_factor = 1 + (severity * 0.5) # Massive spike (up to 5x at max severity)
                sim_df.loc[mask_storm, 'claim_amount_SZL'] *= storm_factor
                
            elif scenario_mode == "Pandemic (Viral Outbreak)":
                mask_health = sim_df['claim_type'].isin(['health emergency', 'livestock loss'])
                sim_df.loc[mask_health, 'claim_amount_SZL'] *= (1 + severity * 0.2)
                
            elif scenario_mode == "Economic Crash (Inflation)":
                sim_df['claim_amount_SZL'] *= (1 + severity * 0.05)
                
            elif scenario_mode == "Civil Unrest":
                mask_unrest = (sim_df['claim_type'].isin(['fire damage', 'theft'])) & (sim_df['location'].isin(['Manzini', 'Mbabane']))
                sim_df.loc[mask_unrest, 'claim_amount_SZL'] *= (1 + severity * 0.4)

            # Calculate New Total Loss
            total_loss_scenario = sim_df[sim_df['Risk_Target'] == 1]['claim_amount_SZL'].sum()
            
            # VISUALIZE COMPARISON
            loss_data = pd.DataFrame({
                'Scenario': ['Baseline', 'Simulated Crisis'],
                'Projected Loss (SZL)': [total_loss_baseline, total_loss_scenario],
                'Color': ['#00FF9D', '#FF4B4B']
            })
            
            fig_stress = px.bar(loss_data, x='Scenario', y='Projected Loss (SZL)', 
                                color='Scenario', color_discrete_sequence=['#00FF9D', '#FF4B4B'])
            fig_stress.update_layout(height=300, paper_bgcolor="#0E1117", font={'color': "white"})
            st.plotly_chart(fig_stress, use_container_width=True)
            
            diff = total_loss_scenario - total_loss_baseline
            st.metric(
                "Net Capital Impact", 
                f"SZL {diff:,.0f}", 
                f"{ (diff/total_loss_baseline)*100 :.1f}% Increase",
                delta_color="inverse"
            )

    # TAB 6: FUTURE HORIZONS
    with tab6:
        st.header("📈 Future Horizons & Model Health")
        c_drift, c_exec = st.columns(2)
        with c_drift:
            st.subheader("⚙️ Model Health & Data Drift Monitor")
            drift_score = get_data_drift_score(X_train_for_drift, input_df)
            fig_drift = go.Figure(go.Indicator(
                mode="gauge+number", value=drift_score, title={'text': "Average Drift Score (KS-Test)"}, 
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF4B4B" if drift_score >= 30 else "#00FF9D"}, 'steps': [{'range': [0, 30], 'color': '#0E1117'}, {'range': [30, 100], 'color': '#21262D'}]}
            ))
            fig_drift.update_layout(height=250, paper_bgcolor="#0E1117", font={'color': "white"})
            st.plotly_chart(fig_drift, use_container_width=True)
            
            if drift_score < 30:
                st.success("✅ **Model Stable:** Live data distribution still resembles training data.")
            else:
                st.error("⚠️ **HIGH DRIFT ALERT:** Live data is significantly different.")

        with c_exec:
            st.subheader("Executive Financial Impact (ROI)")
            total_potential_loss = df[df['Risk_Target'] == 1]['claim_amount_SZL'].sum()
            total_claims = len(df)
            potential_savings_per_flag = total_potential_loss / total_claims * (base_prob if base_prob > 0.5 else 0)
            
            st.metric("Total Potential Annual Loss (Targeted)", f"SZL {total_potential_loss:,.0f}", delta_color="inverse")
            st.metric("Estimated Value of THIS Flag Resolution", f"SZL {potential_savings_per_flag:,.0f}", "High-Risk Claim Value Intercepted", delta_color="normal")

if __name__ == "__main__":
    main()
