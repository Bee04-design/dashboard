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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from scipy.stats import ks_2samp
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. PAGE CONFIG & TEAL/NAVY THEME ---
st.set_page_config(
    page_title="AI Sentinel: Strategic Capital Efficiency",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PROFESSIONAL BRANDING CSS
st.markdown("""
    <style>
    /* Main Background: Deep Navy */
    .stApp { background-color: #0A192F; color: #E6F1FF; }
    
    /* Sidebar: Slightly Lighter Navy */
    [data-testid="stSidebar"] { background-color: #112240; border-right: 1px solid #233554; }
    
    /* Metric Cards: Card Styling with Teal Glow */
    div[data-testid="metric-container"] {
        background-color: #112240; 
        border: 1px solid #233554; 
        padding: 15px; 
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #64FFDA; /* Teal Hover Effect */
    }
    
    /* Headers: Teal & White */
    h1, h2, h3 { 
        font-family: 'Calibri', sans-serif; 
        font-weight: 600;
        color: #E6F1FF;
    }
    h1 { background: -webkit-linear-gradient(0deg, #64FFDA, #E6F1FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    /* Buttons: Teal Accent */
    .stButton>button { 
        background-color: transparent; 
        border: 1px solid #64FFDA; 
        color: #64FFDA; 
        border-radius: 4px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: rgba(100, 255, 218, 0.1);
        color: #ffffff;
        border-color: #ffffff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #112240; border-radius: 4px 4px 0 0; color: #8892b0; }
    .stTabs [aria-selected="true"] { background-color: #64FFDA; color: #0A192F; font-weight: bold; }
    
    /* Alerts */
    .stAlert { background-color: #112240; color: #E6F1FF; border: 1px solid #64FFDA; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING & PREP ---
@st.cache_data(show_spinner="Loading Intelligence Core...")
def load_data():
    try:
        df = pd.read_csv('eswatini_insurance_final_dataset (5).csv')
    except FileNotFoundError:
        try:
             df = pd.read_csv('eswatini_insurance_final_dataset.csv')
        except:
            st.error("❌ DATA ERROR: Please upload 'eswatini_insurance_final_dataset.csv'")
            return None, None, None
    
    # --- Feature Engineering ---
    # 1. Target
    if 'investigation_outcome' in df.columns:
        df['Risk_Target'] = (df['investigation_outcome'] == 'Confirmed Fraud').astype(int)
    else:
        df['Risk_Target'] = (df['claim_amount_SZL'] >= df['claim_amount_SZL'].quantile(0.75)).astype(int)
    
    # Safety check for target classes
    if df['Risk_Target'].nunique() < 2: 
        df.loc[df.sample(10).index, 'Risk_Target'] = 1
        
    # 2. Dates & IDs
    if 'claim_date' not in df.columns: df['claim_date'] = pd.date_range(end='2024-01-01', periods=len(df), freq='D')
    if 'customer_id' not in df.columns: df['customer_id'] = [f"CUST-{i+1:04d}" for i in range(len(df))]
    
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

    # 4. Segmentation (Run here to avoid KeyError later)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cluster_cols = [c for c in numeric_cols if c not in ['Risk_Target', 'claim_id', 'customer_id', 'lat', 'lon']]
    
    if cluster_cols:
        df_numeric = df[cluster_cols].fillna(0)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        df['segment'] = kmeans.fit_predict(df_numeric).astype(str)
    else:
        df['segment'] = "0"
    
    # Split for Evaluation Metrics
    feature_cols = ['claim_type', 'location', 'claim_amount_SZL', 'rural_vs_urban', 
                    'policy_premium_SZL', 'policy_maturity_years', 'segment']
    
    # Handle Missing
    for c in feature_cols:
        if c not in df.columns: df[c] = 0 if 'amount' in c or 'year' in c else "Unknown"

    X_train_for_drift = df[['claim_amount_SZL', 'policy_premium_SZL', 'policy_maturity_years']].copy()

    return df, X_train_for_drift, feature_cols

# --- 3. MODELING ---
@st.cache_resource
def train_models(df, feature_cols):
    X = df[feature_cols].copy()
    y = df['Risk_Target']
    
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    # Train/Test Split for Evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
    model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "cm": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "fpr": roc_curve(y_test, y_prob)[0],
        "tpr": roc_curve(y_test, y_prob)[1],
        "auc": auc(roc_curve(y_test, y_prob)[0], roc_curve(y_test, y_prob)[1])
    }
    
    # Global Importance (Fixed for Treemap)
    global_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return model, encoders, global_imp, metrics

# --- 4. HELPER: RADAR CHART DATA ---
def get_radar_data(df, input_row):
    cols = ['claim_amount_SZL', 'policy_premium_SZL', 'policy_maturity_years']
    scaler = MinMaxScaler()
    avg_fraud = df[df['Risk_Target'] == 1][cols].mean()
    avg_legit = df[df['Risk_Target'] == 0][cols].mean()
    max_vals = df[cols].max()
    return {
        'categories': ['Claim Amount', 'Premium', 'Policy Age'],
        'Current': [input_row[c]/max_vals[c] for c in cols],
        'Fraud_Avg': [avg_fraud[c]/max_vals[c] for c in cols],
        'Legit_Avg': [avg_legit[c]/max_vals[c] for c in cols]
    }

# --- 5. MAIN APP ---
def main():
    # --- DATA INIT ---
    df, X_train_for_drift, feature_cols = load_data()
    if df is None: return
    model, encoders, global_imp, metrics = train_models(df, feature_cols)
    
    # --- EXECUTIVE SUMMARY BANNER ---
    st.title("🛡️ AI Sentinel: Strategic Capital Efficiency")
    c1, c2, c3, c4 = st.columns(4)
    total_risk = df['Risk_Target'].sum()
    total_val = df[df['Risk_Target']==1]['claim_amount_SZL'].sum()
    
    c1.metric("🛡️ System Status", "Active", "Monitoring")
    c2.metric("📊 Total Claims Processed", f"{len(df):,}")
    c3.metric("⚠️ High-Risk Flags", f"{total_risk:,}", f"{(total_risk/len(df)):.1%}")
    c4.metric("💰 Potential Savings", f"SZL {(total_val/1e6):.1f}M", "Annualized")
    
    st.markdown("---")

    # --- SIDEBAR INPUT ---
    st.sidebar.header("🔍 Claim Investigator")
    customer_ids = df['customer_id'].unique()
    # Ensuring unique key for the selectbox to avoid duplicates error
    selected_cust_id = st.sidebar.selectbox("Select Customer ID", customer_ids[:100], key="cust_select")
    
    # Get Data
    cust_data = df[df['customer_id'] == selected_cust_id]
    latest_claim = cust_data.sort_values('claim_date', ascending=False).iloc[0]
    
    # INPUT DICTIONARY
    input_data = {col: latest_claim[col] for col in feature_cols}
    input_df = pd.DataFrame([input_data])
    
    # PREDICT
    enc_df = input_df.copy()
    for col, le in encoders.items():
        val = str(enc_df[col].iloc[0])
        if val in le.classes_:
            enc_df[col] = le.transform([val])
        else:
            enc_df[col] = le.transform([le.classes_[0]])

    base_prob = model.predict_proba(enc_df)[0][1]
    
    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values_result = explainer.shap_values(enc_df)
    
    # Robust SHAP extraction logic
    if isinstance(shap_values_result, list):
        if len(shap_values_result) > 1:
            sv = shap_values_result[1]
        else:
            sv = shap_values_result[0]
    else:
        sv = shap_values_result
        if len(sv.shape) > 1 and sv.shape[1] > 1: # Check if it's (1, features, classes)
             sv = sv[..., 1]

    if len(sv.shape) > 1: sv = sv[0] # Flatten to 1D array
    
    contrib_df = pd.DataFrame({'Feature': input_df.columns, 'SHAP': sv}).sort_values(by='SHAP', key=abs, ascending=True)

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["⚡ Operational View (Claims)", "📊 Tactical View (Performance)", "🔮 Strategic View (Future)"])

    # ==============================================================================
    # TAB 1: OPERATIONAL VIEW (The Investigator)
    # ==============================================================================
    with tab1:
        st.subheader(f"Claim Analysis: {selected_cust_id}")
        
        # ROW 1: SCORE & RADAR
        col_left, col_mid, col_right = st.columns([1, 1.5, 1.5])
        
        with col_left:
            st.caption("RISK PROBABILITY")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=base_prob*100,
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF2B2B" if base_prob > 0.5 else "#64FFDA"}, 'bgcolor': "#112240"},
                number={'suffix': "%", 'font': {'color': "white"}}
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="#0A192F")
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if base_prob > 0.5: st.error("🚫 REJECT / AUDIT")
            else: st.success("✅ APPROVE")

        with col_mid:
            st.caption("BEHAVIORAL FINGERPRINT")
            radar = get_radar_data(df, input_data)
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=radar['Fraud_Avg'], theta=radar['categories'], fill='toself', name='Avg Fraudster', line_color='#FF2B2B'))
            fig_radar.add_trace(go.Scatterpolar(r=radar['Current'], theta=radar['categories'], fill='toself', name='Current Claim', line_color='#64FFDA'))
            fig_radar.update_layout(polar=dict(bgcolor="#112240", radialaxis=dict(visible=True, showticklabels=False)), paper_bgcolor="#0A192F", font_color="white", height=250, margin=dict(l=40,r=40,t=20,b=20))
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_right:
            st.caption("CUSTOMER RISK TIMELINE")
            # Trend Line for Customer
            dates = pd.to_datetime(cust_data['claim_date']).sort_values()
            amounts = cust_data.sort_values('claim_date')['claim_amount_SZL']
            # Simulated risk history for demo
            hist_risk = (amounts / amounts.max()) * 100 
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=dates, y=hist_risk, mode='lines+markers', name='Risk Score', line=dict(color='#64FFDA', width=2)))
            fig_trend.update_layout(paper_bgcolor="#0A192F", plot_bgcolor="#112240", font_color="white", height=250, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
            st.plotly_chart(fig_trend, use_container_width=True)

        # ROW 2: EXPLAINABILITY & ACTION
        c_xai, c_act = st.columns([2, 1])
        
        with c_xai:
            st.caption("RISK CONTRIBUTORS (SHAP)")
            colors = ['#FF2B2B' if x > 0 else '#64FFDA' for x in contrib_df['SHAP']]
            fig_bar = go.Figure(go.Bar(x=contrib_df['SHAP'], y=contrib_df['Feature'], orientation='h', marker=dict(color=colors)))
            fig_bar.update_layout(plot_bgcolor="#0A192F", paper_bgcolor="#0A192F", font_color="white", height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c_act:
            st.caption("RECOMMENDED ACTION")
            st.info("💡 **Policy Lab:**")
            if base_prob > 0.5:
                st.markdown(f"**1. Counterfactual:** Reduce claim by 15% to align with norms.")
                st.markdown(f"**2. Negotiation:** Offer settlement at SZL {(input_data['claim_amount_SZL']*0.85):,.0f}.")
            else:
                st.markdown("**1. Fast Track:** Auto-approve payment.")
                st.markdown("**2. Cross-Sell:** Offer 'Theft Excess Buyback'.")

    # ==============================================================================
    # TAB 2: TACTICAL VIEW (Model & Map)
    # ==============================================================================
    with tab2:
        m1, m2 = st.columns([1.5, 1])
        
        with m1:
            st.subheader("Geographic Risk Heatmap")
            layer = pdk.Layer(
                "HexagonLayer", df[['lon', 'lat']], get_position=['lon', 'lat'],
                auto_highlight=True, elevation_scale=50, pickable=True, elevation_range=[0, 3000],
                extruded=True, coverage=1,
                color_range=[[100, 255, 218], [0, 194, 255], [0, 120, 255], [120, 0, 255], [255, 0, 100], [255, 0, 0]]
            )
            view = pdk.ViewState(longitude=31.465, latitude=-26.52, zoom=8.2, pitch=55)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))
            
        with m2:
            st.subheader("Model Evaluation (Robustness)")
            # ROC CURVE
            fig_roc = px.area(x=metrics['fpr'], y=metrics['tpr'], title=f"ROC Curve (AUC = {metrics['auc']:.2f})", labels=dict(x='False Positive Rate', y='True Positive Rate'))
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_traces(line_color='#64FFDA', fillcolor='rgba(100, 255, 218, 0.2)')
            fig_roc.update_layout(paper_bgcolor="#0A192F", plot_bgcolor="#112240", font_color="white", height=300)
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # CONFUSION MATRIX
            cm = metrics['cm']
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Teal', title="Confusion Matrix")
            fig_cm.update_layout(paper_bgcolor="#0A192F", plot_bgcolor="#112240", font_color="white", height=300)
            st.plotly_chart(fig_cm, use_container_width=True)

    # ==============================================================================
    # TAB 3: STRATEGIC VIEW (Scenarios)
    # ==============================================================================
    with tab3:
        st.subheader("Crystal Ball: Scenario Simulation")
        
        s_col, s_res = st.columns([1, 2])
        
        with s_col:
            scenario = st.selectbox("Simulation Scenario", ["None", "Severe Weather (Flood)", "Pandemic (Health)", "Economic Crisis"], key="sim_select")
            severity = st.slider("Severity Impact", 1, 10, 5)
            
        with s_res:
            base_loss = df[df['Risk_Target']==1]['claim_amount_SZL'].sum()
            sim_loss = base_loss
            
            if scenario == "Severe Weather (Flood)": sim_loss *= (1 + severity * 0.3)
            elif scenario == "Pandemic (Health)": sim_loss *= (1 + severity * 0.2)
            elif scenario == "Economic Crisis": sim_loss *= (1 + severity * 0.1)
            
            loss_df = pd.DataFrame({
                'Scenario': ['Current Baseline', f'Simulated: {scenario}'],
                'Loss Exposure': [base_loss, sim_loss],
                'Color': ['#64FFDA', '#FF2B2B']
            })
            
            fig_sim = go.Figure(go.Bar(
                x=loss_df['Loss Exposure'], y=loss_df['Scenario'], orientation='h',
                marker_color=loss_df['Color'], text=loss_df['Loss Exposure'].apply(lambda x: f"SZL {x:,.0f}"),
                textposition='auto'
            ))
            fig_sim.update_layout(paper_bgcolor="#0A192F", plot_bgcolor="#112240", font_color="white", xaxis=dict(showgrid=False))
            st.plotly_chart(fig_sim, use_container_width=True)
            
            if scenario != "None":
                st.warning(f"⚠️ Projected Capital Impact: **+{(sim_loss-base_loss)/base_loss:.1%}** increase in liability.")

if __name__ == "__main__":
    main()
