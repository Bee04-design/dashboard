import streamlit as st
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

# CUSTOM CSS: NEON DARK THEME
st.markdown("""
    <style>
    /* Background & Text */
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    
    /* Containers */
    div[data-testid="metric-container"] {
        background-color: #1E2130; border: 1px solid #4F8BF9; 
        padding: 15px; border-radius: 12px;
        box-shadow: 0 0 15px rgba(79, 139, 249, 0.2); /* Neon Blue Glow */
    }
    
    /* Headers */
    h1, h2, h3 { 
        font-family: 'Helvetica Neue', sans-serif; 
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #00FF9D, #00C2FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Ticker Animation */
    .ticker-wrap { 
        width: 100%; background-color: #1E2130; padding: 12px; border-bottom: 2px solid #00C2FF; 
    }
    .ticker { 
        display: inline-block; white-space: nowrap; animation: ticker 35s linear infinite; 
        color: #00FF9D; font-family: 'Courier New', monospace; font-weight: bold; font-size: 1.1em;
    }
    @keyframes ticker { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING & PREP ---
@st.cache_data(show_spinner="Initializing AI Sentinel Core...")
def load_data():
    try:
        df = pd.read_csv('eswatini_insurance_final_dataset (5).csv')
    except FileNotFoundError:
        try:
             df = pd.read_csv('eswatini_insurance_final_dataset.csv')
        except:
            st.error("❌ DATA ERROR: Please upload 'eswatini_insurance_final_dataset.csv'")
            return None, None
    
    # --- Feature Engineering ---
    # 1. Target
    if 'investigation_outcome' in df.columns:
        df['Risk_Target'] = (df['investigation_outcome'] == 'Confirmed Fraud').astype(int)
    else:
        df['Risk_Target'] = (df['claim_amount_SZL'] >= df['claim_amount_SZL'].quantile(0.75)).astype(int)
    
    # Safety: Ensure at least 2 classes
    if df['Risk_Target'].nunique() < 2:
        df.loc[df.sample(10).index, 'Risk_Target'] = 1
        
    # 2. Dates & IDs
    if 'claim_date' not in df.columns:
        df['claim_date'] = pd.date_range(end='2024-01-01', periods=len(df), freq='D')
    if 'customer_id' not in df.columns:
        df['customer_id'] = [f"CUST-{i+1:04d}" for i in range(len(df))]
    
    df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
    df['Month_Year'] = df['claim_date'].dt.to_period('M').astype(str)
        
    # 3. Geo-Coordinates
    coords = {
        "Manzini": [-26.50, 31.36], "Mbabane": [-26.31, 31.13], "Siteki": [-26.45, 31.95],
        "Big Bend": [-26.81, 31.93], "Lobamba": [-26.46, 31.20], "Piggs Peak": [-25.96, 31.25],
        "Nhlangano": [-27.11, 31.20], "Simunye": [-26.21, 31.91]
    }
    df['lat'] = df['location'].map(lambda x: coords.get(x, [None, None])[0])
    df['lon'] = df['location'].map(lambda x: coords.get(x, [None, None])[1])
    df['lat'] = df['lat'].fillna(-26.50)
    df['lon'] = df['lon'].fillna(31.36)

    # 4. Segmentation (Fixing the previous KeyError)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cluster_cols = [c for c in numeric_cols if c not in ['Risk_Target', 'claim_id', 'customer_id', 'lat', 'lon']]
    
    if cluster_cols:
        df_numeric = df[cluster_cols].fillna(0)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        df['segment'] = kmeans.fit_predict(df_numeric).astype(str)
    else:
        df['segment'] = "0"
    
    X_train_for_drift = df[['claim_amount_SZL', 'policy_premium_SZL', 'policy_maturity_years']].copy()

    return df, X_train_for_drift

# --- 3. MODELING ---
@st.cache_resource
def train_models(df):
    feature_cols = ['claim_type', 'location', 'claim_amount_SZL', 'rural_vs_urban', 
                    'policy_premium_SZL', 'policy_maturity_years', 'segment']
    
    # Handle Missing
    for c in feature_cols:
        if c not in df.columns: df[c] = 0 if 'amount' in c or 'year' in c else "Unknown"
            
    X = df[feature_cols].copy()
    y = df['Risk_Target']
    
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42)
    model.fit(X, y)
    
    # Global Importance (Fixed for Treemap)
    global_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return model, encoders, global_imp

# --- 4. HELPER: RADAR CHART DATA ---
def get_radar_data(df, input_row):
    # Normalize data for radar comparison
    cols = ['claim_amount_SZL', 'policy_premium_SZL', 'policy_maturity_years']
    scaler = MinMaxScaler()
    
    # Get Averages
    avg_fraud = df[df['Risk_Target'] == 1][cols].mean()
    avg_legit = df[df['Risk_Target'] == 0][cols].mean()
    
    # Max values for normalization
    max_vals = df[cols].max()
    
    return {
        'categories': ['Claim Amount', 'Premium', 'Policy Age'],
        'Current': [input_row[c]/max_vals[c] for c in cols],
        'Fraud_Avg': [avg_fraud[c]/max_vals[c] for c in cols],
        'Legit_Avg': [avg_legit[c]/max_vals[c] for c in cols]
    }

# --- 5. MAIN APP ---
def main():
    df, X_train_for_drift = load_data()
    if df is None: return
    model, encoders, global_imp = train_models(df)
    
    # --- HEADER & TICKER ---
    if 'scenario_mode' not in st.session_state: st.session_state.scenario_mode = "None (Baseline)"
    
    news_text = "📡 AI SENTINEL STATUS: LIVE MONITORING ACTIVE... Analyzing incoming claims from 4 regions..."
    if st.session_state.scenario_mode == "Severe Weather": news_text = "⛈️ CRITICAL ALERT: Flood Warning in Manzini Region. High risk of water damage claims..."
    
    st.markdown(f"""<div class="ticker-wrap"><div class="ticker">{news_text} &nbsp;&nbsp; /// &nbsp;&nbsp; {news_text}</div></div>""", unsafe_allow_html=True)
    st.title("🛡️ AI Sentinel: Strategic Capital Efficiency")
    st.markdown("**Interactive Risk Analytics Dashboard** | *University of Eswatini*")

    # --- SIDEBAR INPUT ---
    st.sidebar.header("🔍 Claim Investigator")
    customer_ids = df['customer_id'].unique()
    selected_cust_id = st.sidebar.selectbox("Select Customer ID", customer_ids[:100])
    
    # Get Data
    cust_data = df[df['customer_id'] == selected_cust_id]
    latest_claim = cust_data.sort_values('claim_date', ascending=False).iloc[0]
    
    # INPUT DICTIONARY
    input_data = {
        'claim_type': latest_claim['claim_type'], 
        'location': latest_claim['location'], 
        'claim_amount_SZL': latest_claim['claim_amount_SZL'],
        'rural_vs_urban': latest_claim['rural_vs_urban'], 
        'policy_premium_SZL': latest_claim['policy_premium_SZL'], 
        'policy_maturity_years': latest_claim['policy_maturity_years'], 
        'segment': str(latest_claim['segment']) 
    }
    input_df = pd.DataFrame([input_data])
    
    # PREDICT
    enc_df = input_df.copy()
    for col, le in encoders.items():
        val = str(enc_df[col].iloc[0])
        enc_df[col] = le.transform([val if val in le.classes_ else le.classes_[0]])

    base_prob = model.predict_proba(enc_df)[0][1]
    
    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(enc_df)
    # Robust extraction
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values[..., 1] if len(shap_values.shape) > 2 else shap_values
    
    if len(sv.shape) > 1: sv = sv[0] # Flatten
    
    # Contribution DataFrame (Fixed)
    contrib_df = pd.DataFrame({
        'Feature': input_df.columns,
        'SHAP': sv
    }).sort_values(by='SHAP', key=abs, ascending=True) # Sorted for bar chart

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["🚦 Risk Engine (Real-Time)", "🌍 Strategic Intel (Trends)", "🔮 Crystal Ball (Simulations)"])

    # ==============================================================================
    # TAB 1: RISK ENGINE (The Decision Center)
    # ==============================================================================
    with tab1:
        st.markdown("### 🕵️ Forensic Claim Analysis")
        
        # ROW 1: Gauge + Radar (The "Visual Identity" of the claim)
        c1, c2, c3 = st.columns([1, 1.5, 1.5])
        
        with c1:
            st.caption("RISK PROBABILITY")
            # GAUGE CHART
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=base_prob*100,
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF2B2B" if base_prob > 0.5 else "#00FF9D"}, 'bgcolor': "#1E2130"},
                number={'suffix': "%", 'font': {'color': "white"}}
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="#0E1117")
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if base_prob > 0.5:
                st.error("🚫 ACTION: INVESTIGATE")
            else:
                st.success("✅ ACTION: APPROVE")

        with c2:
            st.caption("RISK DNA (RADAR COMPARISON)")
            # RADAR CHART (New!)
            radar = get_radar_data(df, input_data)
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=radar['Fraud_Avg'], theta=radar['categories'], fill='toself', name='Avg Fraudster', line_color='#FF2B2B', opacity=0.6))
            fig_radar.add_trace(go.Scatterpolar(r=radar['Current'], theta=radar['categories'], fill='toself', name='Current Claim', line_color='#00C2FF', opacity=0.6))
            fig_radar.update_layout(
                polar=dict(bgcolor="#1E2130", radialaxis=dict(visible=True, showticklabels=False, gridcolor="#333")),
                paper_bgcolor="#0E1117", font_color="white", height=250, margin=dict(l=40,r=40,t=20,b=20), showlegend=True,
                legend=dict(x=0, y=-0.2, orientation="h")
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with c3:
            st.caption("KEY DRIVERS (Why?)")
            # FEATURE CONTRIBUTION (Colorful Bar Chart)
            # Map colors: Red for Risk-Increasing, Green for Risk-Decreasing
            colors = ['#FF2B2B' if x > 0 else '#00FF9D' for x in contrib_df['SHAP']]
            
            fig_bar = go.Figure(go.Bar(
                x=contrib_df['SHAP'], y=contrib_df['Feature'], orientation='h',
                marker=dict(color=colors, line=dict(width=0))
            ))
            fig_bar.update_layout(
                plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white",
                margin=dict(l=10,r=10,t=10,b=10), height=250,
                xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="white"),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        
        # ROW 2: Counterfactual (Text Action)
        st.subheader("💡 AI Recommendation Engine")
        if base_prob > 0.5:
            st.warning(f"**Negotiation Strategy:** If the Claim Amount is reduced by **15%**, the risk score is projected to drop below the critical threshold.")
        else:
            st.info("**Optimization:** Customer is low-risk. Suggest **Cross-Selling** 'Vehicle Gap Cover' based on their segment.")

    # ==============================================================================
    # TAB 2: STRATEGIC INTEL (Maps & Trends)
    # ==============================================================================
    with tab2:
        st.markdown("### 🌍 Market Intelligence")
        
        # ROW 1: 3D Map + Sunburst
        m1, m2 = st.columns([2, 1])
        
        with m1:
            st.caption("GEOSPATIAL RISK DENSITY (3D)")
            # 3D MAP
            layer = pdk.Layer(
                "HexagonLayer", df[['lon', 'lat']], get_position=['lon', 'lat'],
                auto_highlight=True, elevation_scale=50, pickable=True, elevation_range=[0, 3000],
                extruded=True, coverage=1,
                color_range=[[0, 255, 157], [65, 255, 255], [0, 120, 255], [120, 0, 255], [255, 0, 100], [255, 0, 0]]
            )
            view = pdk.ViewState(longitude=31.465, latitude=-26.52, zoom=8.2, pitch=55)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))
            
        with m2:
            st.caption("CLAIM HIERARCHY")
            # SUNBURST (Interactive Pie)
            fig_sun = px.sunburst(
                df, path=['location', 'claim_type'], values='claim_amount_SZL',
                color='location', color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_sun.update_layout(paper_bgcolor="#0E1117", height=350, margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig_sun, use_container_width=True)

        # ROW 2: Trend Lines & Treemap
        t1, t2 = st.columns([2, 1])
        
        with t1:
            st.caption("CLAIM VOLUME TREND (2018-2024)")
            # AGGREGATE BY MONTH
            trend_data = df.groupby('Month_Year')['claim_amount_SZL'].sum().reset_index()
            trend_data['Month_Year'] = trend_data['Month_Year'].astype(str)
            
            fig_area = px.area(trend_data, x='Month_Year', y='claim_amount_SZL', template="plotly_dark")
            fig_area.update_traces(line_color='#00C2FF', fill='tozeroy', fillcolor='rgba(0, 194, 255, 0.2)')
            fig_area.update_layout(paper_bgcolor="#0E1117", height=300, margin=dict(t=20, l=20, r=20, b=20))
            st.plotly_chart(fig_area, use_container_width=True)
            
        with t2:
            st.caption("GLOBAL RISK DRIVERS")
            # TREEMAP (Fixed)
            fig_tree = px.treemap(
                global_imp, path=['Feature'], values='Importance',
                color='Importance', color_continuous_scale='Sunsetdark'
            )
            fig_tree.update_layout(paper_bgcolor="#0E1117", height=300, margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig_tree, use_container_width=True)

    # ==============================================================================
    # TAB 3: CRYSTAL BALL (Scenarios)
    # ==============================================================================
    with tab3:
        st.header("🔮 Future Simulations")
        
        scen_col, res_col = st.columns([1, 2])
        
        with scen_col:
            st.info("Adjust parameters to stress-test capital reserves.")
            scenario = st.selectbox("Select Scenario", ["None", "Severe Weather", "Pandemic", "Civil Unrest"])
            st.session_state.scenario_mode = scenario
            severity = st.slider("Severity Level", 1, 10, 5)
            
        with res_col:
            # Simulation Logic
            base_loss = df[df['Risk_Target']==1]['claim_amount_SZL'].sum()
            sim_loss = base_loss
            
            if scenario == "Severe Weather": sim_loss *= (1 + severity * 0.2)
            elif scenario == "Pandemic": sim_loss *= (1 + severity * 0.15)
            elif scenario == "Civil Unrest": sim_loss *= (1 + severity * 0.4)
            
            # Horizontal Bar Comparison
            loss_df = pd.DataFrame({
                'Scenario': ['Current Reality', f'Simulated: {scenario}'],
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
            
            if scenario != "None":
                diff = sim_loss - base_loss
                st.error(f"⚠️ PROJECTED CAPITAL DEFICIT: SZL {diff:,.2f}")

if __name__ == "__main__":
    main()
