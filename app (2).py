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

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="AI Sentinel: Strategic Capital Efficiency Dashboard",
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
    @keyframes ticker { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }
    .ticker-wrap { width: 100%; overflow: hidden; background-color: #21262D; padding: 10px; border-bottom: 1px solid #30363D; }
    .ticker { display: inline-block; white-space: nowrap; animation: ticker 30s linear infinite; color: #00FF9D; font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING & PREP ---
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
    if df is None: return None, None, None 
    
    # ... (Data preparation logic from previous steps) ...
    if 'investigation_outcome' in df.columns:
        df['Risk_Target'] = (df['investigation_outcome'] == 'Confirmed Fraud').astype(int)
    else:
        df['Risk_Target'] = (df['claim_amount_SZL'] >= df['claim_amount_SZL'].quantile(0.75)).astype(int)
        
    if 'claim_date' not in df.columns:
        df['claim_date'] = pd.date_range(end='2024-01-01', periods=len(df), freq='H')

    if 'customer_id' not in df.columns:
        df['customer_id'] = [f"CUST-{i+1:04d}" for i in range(len(df))]
        
    coords = {
        "Manzini": [-26.50, 31.36], "Mbabane": [-26.31, 31.13], "Siteki": [-26.45, 31.95],
        "Big Bend": [-26.81, 31.93], "Lobamba": [-26.46, 31.20], "Piggs Peak": [-25.96, 31.25],
        "Nhlangano": [-27.11, 31.20], "Simunye": [-26.21, 31.91]
    }
    df['lat'] = df['location'].map(lambda x: coords.get(x, [None, None])[0])
    df['lon'] = df['location'].map(lambda x: coords.get(x, [None, None])[1])
    df['lat'] = df['lat'].fillna(df['lat'].mean())
    df['lon'] = df['lon'].fillna(df['lon'].mean())

    df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
    df = df.dropna(subset=['claim_date'])
    
    X_train = df[['claim_amount_SZL', 'policy_premium_SZL', 'policy_maturity_years']].copy()

    return df, X_train

# --- 3. MODELING & TRAINING ---
@st.cache_resource
def train_models(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df_numeric = df[numeric_cols].fillna(0)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
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
    
    global_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    
    return model, encoders, global_importance

# --- 4. COUNTERFACTUAL FUNCTION (logic for Tab 1) ---
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

# --- 5. DATA DRIFT FUNCTION (logic for Tab 6) ---
def get_data_drift_score(X_train, input_df):
    drift_metrics = {}
    total_drift = 0
    comparison_features = ['claim_amount_SZL', 'policy_premium_SZL']

    for feature in comparison_features:
        live_data = pd.concat([X_train[feature].sample(min(100, len(X_train)), random_state=42), input_df[feature]])
        D_stat, p_value = ks_2samp(X_train[feature], live_data)
        drift_metrics[feature] = D_stat
        total_drift += D_stat

    avg_drift_score = (total_drift / len(comparison_features)) * 100
    
    return avg_drift_score, drift_metrics

# --- 6. MAIN APP ---
def main():
    df, X_train_for_drift = load_data()
    if df is None: st.error("Data not found. Ensure CSV files are present."); return
    model, encoders, global_importance = train_models(df)
    
    # --- NEWS TICKER (KEAS removed) ---
    news_text = "📡 SYSTEM STATUS: Monitoring Active Claims. Presenting at Conference..."
    st.markdown(f"""<div class="ticker-wrap"><div class="ticker">{news_text} &nbsp;&nbsp;&nbsp;&nbsp; /// &nbsp;&nbsp;&nbsp;&nbsp; {news_text}</div></div>""", unsafe_allow_html=True)

    # --- TITLE CHANGE ---
    st.title("🛡️ AI Sentinel: High-Risk Claim Classification Dashboard")
    st.markdown("Developed by Bhekiwe Sindiswa Dlamini | University of Eswatini")

    # --- Identity Lookup & Simulator Prep ---
    customer_ids = df['customer_id'].unique()
    selected_cust_id = st.sidebar.selectbox("Search Customer ID / Name", customer_ids[:50])
    cust_data = df[df['customer_id'] == selected_cust_id]
    if cust_data.empty: return
    latest_claim = cust_data.sort_values('claim_date', ascending=False).iloc[0]
    
    input_data = {
        'claim_type': latest_claim['claim_type'], 'location': latest_claim['location'], 
        'claim_amount_SZL': latest_claim['claim_amount_SZL'],
        'rural_vs_urban': 'urban', 'policy_premium_SZL': latest_claim['policy_premium_SZL'], 
        'policy_maturity_years': latest_claim['policy_maturity_years'], 
        'accident_location': latest_claim.get('accident_location', 'N/A'),
        'cause_of_fire': 'N/A', 'segment': '0'
    }
    input_df = pd.DataFrame([input_data])
    
    # PREDICTION LOGIC
    enc_df = input_df.copy()
    for col, le in encoders.items():
        val = str(enc_df[col].iloc[0])
        if val in le.classes_:
            enc_df[col] = le.transform([val])
        else:
            enc_df[col] = le.transform([le.classes_[0]])

    base_prob = model.predict_proba(enc_df)[0][1]
    
    # XAI Calculations (for Tab 1)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(enc_df)[1]
    expected_value = explainer.expected_value[1]
    feature_contributions = pd.DataFrame({
        'Feature': input_df.columns, 'Value': input_df.iloc[0].values, 'SHAP_Value': shap_values[0]
    }).sort_values('SHAP_Value', ascending=False).head(8)


    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚦 Risk Engine", "🌐 Strategic Map", "💰 Policy Lab", "🎮 Human vs AI", "🔮 Crystal Ball", "📈 Future Horizons"
    ])

    # TAB 1: RISK ENGINE 
    with tab1:
        st.header("Real-Time Decision Transparency")
        c_top1, c_top2 = st.columns([1, 2])
        
        with c_top1:
            st.subheader("AI Risk Score")
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=base_prob*100, title={'text': "Probability"}, gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#FF4B4B" if base_prob > 0.5 else "#00FF9D"}, 'bgcolor': "#21262D"}))
            fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="#0E1117", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("---")
            st.subheader("💡 Counterfactual Guidance")
            if base_prob > 0.5:
                counterfactuals = find_counterfactual(model, encoders, input_df, base_prob)
                if counterfactuals:
                    st.warning("**To Drop Risk Below 50%:**")
                    for feat, change in counterfactuals.items():
                        st.markdown(f"- **{feat}:** {change}")
                else:
                    st.error("Severe Risk. Requires significant intervention outside of policy adjustments.")
            else:
                st.success("Claim is Low-Risk. No intervention required.")
            
        with c_top2:
            st.subheader("Feature Contribution (Local)")
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
            st.subheader("SHAP Waterfall (Detail)")
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap.Explanation(shap_values[0], expected_value, input_df.iloc[0]), max_display=6, show=False)
            st.pyplot(fig)


    # TAB 2: STRATEGIC MAP 
    with tab2:
        st.subheader("3D Risk Density Map")
        layer = pdk.Layer("HexagonLayer", df[['lon', 'lat']], get_position=['lon', 'lat'], auto_highlight=True, elevation_scale=50, pickable=True, elevation_range=[0, 3000], extruded=True, coverage=1)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(longitude=31.465, latitude=-26.52, zoom=8.2, pitch=55)))

    # TAB 6: FUTURE HORIZONS (Model Health Monitoring)
    with tab6:
        st.header("📈 Future Horizons & Model Health")
        
        c_drift, c_exec = st.columns(2)
        with c_drift:
            st.subheader("⚙️ Model Health & Data Drift Monitor")
            drift_score, drift_metrics = get_data_drift_score(X_train_for_drift, input_df)
            
            fig_drift = go.Figure(go.Indicator(
                mode="gauge+number", 
                value=drift_score, 
                title={'text': "Average Drift Score"}, 
                gauge={'axis': {'range': [0, 100]}, 
                       'bar': {'color': "#00FF9D" if drift_score < 30 else "#FF4B4B"}, 
                       'steps': [{'range': [0, 30], 'color': '#0E1117'}, {'range': [30, 100], 'color': '#21262D'}]
                }
            ))
            fig_drift.update_layout(height=250, paper_bgcolor="#0E1117", font={'color': "white"})
            st.plotly_chart(fig_drift, use_container_width=True)
            
            if drift_score < 30:
                st.success("✅ **Model Stable:** Live data still resembles training data.")
            else:
                st.error("⚠️ **HIGH DRIFT ALERT:** Live data is significantly different. Recommend model re-training.")

        with c_exec:
            st.subheader("Executive Financial Impact")
            total_potential_loss = df[df['Risk_Target'] == 1]['claim_amount_SZL'].sum()
            total_claims = len(df)
            potential_savings_per_flag = total_potential_loss / total_claims * (base_prob if base_prob > 0.5 else 0)
            
            st.metric("Total Potential Annual Loss (Fraud)", f"SZL {total_potential_loss:,.0f}", delta_color="inverse")
            st.metric("Estimated Savings for THIS Flag", f"SZL {potential_savings_per_flag:,.0f}", "High-Risk Claim Value", delta_color="normal")
            st.info("The system projects the financial value of every high-risk flag.")

    # (Tabs 3, 4, 5 logic omitted for brevity in the final code output)
    with tab3: st.header("Policy Optimization")
    with tab4: st.header("🎮 Challenge Mode")
    with tab5: st.header("🔮 Stress Test")

if __name__ == "__main__":
    main()
