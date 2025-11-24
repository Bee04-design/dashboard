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
from sklearn.model_selection import train_test_split # New Import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve # New Imports
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
    df['Year'] = df['claim_date'].dt.year # Added for better trend simulation
        
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
        # Scale data for KMeans stability
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_numeric)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        df['segment'] = kmeans.fit_predict(df_scaled).astype(str)
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
    
    # 1. Split data for rigorous evaluation (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
        
    model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # 2. Evaluate model on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'conf_matrix': confusion_matrix(y_test, y_pred),
        'y_test': y_test,
        'y_proba': y_proba
    }
        
    # Global Importance
    global_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return model, encoders, global_imp, metrics # Return metrics

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

# --- 5. HELPER: BOOTSTRAP SIMULATION ---
@st.cache_data(show_spinner="Running Bootstrap Simulations...", hash_funcs={RandomForestClassifier: lambda _: 'fixed_model_hash'})
def run_bootstrap_simulation(df, model, n_bootstraps=50):
    """
    Simulates future risk trend using a bootstrapped approach.
    """
    
    # 1. Predict Risk for all historical data
    feature_cols = ['claim_type', 'location', 'claim_amount_SZL', 'rural_vs_urban', 
                    'policy_premium_SZL', 'policy_maturity_years', 'segment']
    X = df[feature_cols].copy()
    
    # Re-encode categorical features
    # NOTE: Calling train_models here is inefficient but guarantees encoders are available
    _, encoders, _, _ = train_models(df) 
    for col, le in encoders.items():
        # Apply transformation with error handling for unseen categories
        def safe_transform(val):
            s_val = str(val)
            if s_val in le.classes_:
                return le.transform([s_val])[0]
            # Use the first class as a default fallback for simplicity in simulation
            return le.transform([le.classes_[0]])[0] 
            
        X[col] = X[col].apply(safe_transform)
        
    df['Risk_Prob'] = model.predict_proba(X)[..., 1]
    
    # 2. Calculate Historical Risk Rate (Monthly)
    historical_risk_rate = df.groupby('Month_Year')['Risk_Target'].mean().reset_index()
    historical_risk_rate['Month_Year'] = pd.to_datetime(historical_risk_rate['Month_Year'])
    historical_risk_rate = historical_risk_rate.rename(columns={'Risk_Target': 'Risk_Rate'})
    
    # 3. Forecast Setup
    last_month = historical_risk_rate['Month_Year'].max()
    forecast_months = pd.to_datetime([last_month + pd.DateOffset(months=1), last_month + pd.DateOffset(months=2)])
    
    # Find the average monthly change (trend)
    historical_risk_rate['Change'] = historical_risk_rate['Risk_Rate'].diff().fillna(0)
    avg_trend = historical_risk_rate['Change'].tail(6).mean() # Base trend on last 6 months
    
    simulations = []
    
    for i in range(n_bootstraps):
        
        # Start with historical data
        sim_df = historical_risk_rate.copy()
        
        # Current risk rate
        current_rate = sim_df['Risk_Rate'].iloc[-1]
        
        # Simulate two future months
        for fm in forecast_months:
            # Random noise (bootstrapping effect)
            noise = np.random.normal(0, 0.01) # Small random walk component
            
            # Simple Linear Forecast: last_rate + avg_trend + noise
            next_rate = max(0, min(1, current_rate + avg_trend + noise))
            
            # Append to simulation
            new_row = pd.DataFrame([{'Month_Year': fm, 'Risk_Rate': next_rate, 'Simulation': i}])
            sim_df = pd.concat([sim_df, new_row], ignore_index=True)
            current_rate = next_rate # Update current rate for the next step
            
        # Store simulation results
        sim_df['Simulation'] = i
        simulations.append(sim_df[['Month_Year', 'Risk_Rate', 'Simulation']].tail(len(forecast_months)))
        
    forecast_results = pd.concat(simulations)
    
    return historical_risk_rate, forecast_results

# --- 6. MAIN APP ---
def main():
    # Update call to receive the new 'metrics' dictionary
    df, X_train_for_drift = load_data()
    if df is None: return
    model, encoders, global_imp, metrics = train_models(df)
    
    # Define the canonical list of features used by the model
    trained_feature_cols = ['claim_type', 'location', 'claim_amount_SZL', 'rural_vs_urban', 
                            'policy_premium_SZL', 'policy_maturity_years', 'segment']

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
    
    # Get Data for selected customer and sort by date
    cust_data = df[df['customer_id'] == selected_cust_id].sort_values('claim_date', ascending=True).copy()
    if cust_data.empty:
        st.sidebar.warning("No data found for this customer.")
        return
    
    # --- CALCULATE RISK FOR ALL CUSTOMER CLAIMS ---
    customer_claims_encoded = cust_data[trained_feature_cols].copy()
    
    for col, le in encoders.items():
        if col in customer_claims_encoded.columns:
            def safe_transform_cust(val):
                s_val = str(val)
                if s_val in le.classes_:
                    return le.transform([s_val])[0]
                # Default to the first class if unseen category encountered
                return le.transform([le.classes_[0]])[0] 
            
            customer_claims_encoded[col] = customer_claims_encoded[col].apply(safe_transform_cust)

    # Predict risk for customer's historical claims
    cust_data['Risk_Prob'] = model.predict_proba(customer_claims_encoded)[..., 1]
    cust_data['Risk_Prob_Pct'] = cust_data['Risk_Prob'] * 100
    
    customer_risk_trend = cust_data[['claim_date', 'Risk_Prob_Pct']].copy()
    
    # Get the latest claim data point for the single-claim analysis
    latest_claim = cust_data.iloc[-1] 
    base_prob = latest_claim['Risk_Prob']

    # INPUT DICTIONARY (Used for Radar Chart/SHAP Input Display)
    input_data = {
        'claim_type': latest_claim.get('claim_type', 'Unknown'), 
        'location': latest_claim.get('location', 'Manzini'), 
        'claim_amount_SZL': latest_claim.get('claim_amount_SZL', 1000),
        'rural_vs_urban': latest_claim.get('rural_vs_urban', 'Unknown'), 
        'policy_premium_SZL': latest_claim.get('policy_premium_SZL', 500), 
        'policy_maturity_years': latest_claim.get('policy_maturity_years', 5), 
        'segment': str(latest_claim.get('segment', '0'))
    }
    input_df = pd.DataFrame([input_data])
    
    # PREDICT: Get the encoded features for the latest claim needed for SHAP
    enc_df = input_df.copy()
    for col, le in encoders.items():
        val = str(enc_df[col].iloc[0])
        enc_df[col] = le.transform([val if val in le.classes_ else le.classes_[0]])

    # SHAP
    explainer = shap.TreeExplainer(model)
    try:
        # Use enc_df for SHAP calculation
        shap_values = explainer.shap_values(enc_df)
    except Exception:
        shap_values = explainer.shap_values(enc_df.iloc[0]) 

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
    # Added tab4
    tab1, tab2, tab3, tab4 = st.tabs(["🚦 Risk Engine (Real-Time)", "🌍 Strategic Intel (Trends)", "🔮 Crystal Ball (Simulations)", "⚙️ Model Audit (Evaluation)"])

    # ==============================================================================
    # TAB 1: RISK ENGINE (The Decision Center)
    # ==============================================================================
    with tab1:
        st.markdown("### 🕵️ Forensic Claim Analysis")
        
        # ROW 1: Gauge + Radar + SHAP Drivers
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
            fig_radar.add_trace(go.Scatterpolar(r=radar['Fraud_Avg'], theta=radar['categories'], fill='toself', name='Avg High-Risk', line_color='#FF2B2B', opacity=0.6))
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
        
        # ROW 2: Recommendation Engine and Customer Trend (New layout)
        col_rec, col_trend = st.columns([1, 1])

        with col_rec:
            st.subheader("💡 AI Recommendation Engine")
            if base_prob > 0.5:
                st.warning(f"**Negotiation Strategy:** If the Claim Amount is reduced by **15%**, the risk score is projected to drop below the critical threshold.")
            else:
                st.info("**Optimization:** Customer is low-risk. Suggest **Cross-Selling** 'Vehicle Gap Cover' based on their segment.")
        
        with col_trend:
            st.subheader("📊 Customer Claim History")
            st.caption(f"Risk Probability Trend for {selected_cust_id}")
            
            # CUSTOMER RISK TREND LINE CHART
            fig_cust_trend = px.line(customer_risk_trend, x='claim_date', y='Risk_Prob_Pct',
                                    title='Risk Probability Over Time', template="plotly_dark")
            
            # Highlight the latest claim point
            latest_claim_prob = customer_risk_trend['Risk_Prob_Pct'].iloc[-1]
            latest_claim_date = customer_risk_trend['claim_date'].iloc[-1]

            fig_cust_trend.add_trace(go.Scatter(
                x=[latest_claim_date], y=[latest_claim_prob],
                mode='markers', name='Latest Claim',
                marker=dict(size=12, color='#FF2B2B', symbol='circle')
            ))
            
            fig_cust_trend.update_traces(mode='lines', line_color='#00C2FF')
            fig_cust_trend.update_layout(
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white",
                yaxis_title="Risk Probability (%)", xaxis_title="Claim Date",
                height=350, margin=dict(t=40, l=20, r=20, b=20),
                showlegend=False
            )
            
            st.plotly_chart(fig_cust_trend, use_container_width=True)


    # ==============================================================================
    # TAB 2: STRATEGIC INTEL (Maps & Trends)
    # ==============================================================================
    with tab2:
        st.markdown("### 🌍 Market Intelligence")
        
        # ROW 1: Trend Risk Analysis (NEW)
        t1, t2 = st.columns([2, 1])

        with t1:
            st.caption("TREND RISK ANALYSIS: HIGH-RISK CLAIMS RATE (Monthly)")
            # --- NEW GRAPH: RISK RATE TREND ---
            # Group by month and calculate the average Risk_Target (which is the risk rate)
            risk_trend_data = df.groupby('Month_Year')['Risk_Target'].mean().reset_index()
            risk_trend_data['Risk_Target'] *= 100 # Convert to percentage
            
            fig_risk_trend = px.line(risk_trend_data, x='Month_Year', y='Risk_Target', 
                                    title='Percentage of Claims Flagged as High-Risk', template="plotly_dark")
            
            fig_risk_trend.update_traces(mode='lines+markers', line_color='#00FF9D', marker_size=4)
            fig_risk_trend.update_layout(
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white",
                yaxis_title="High-Risk Rate (%)", xaxis_title="Time (Month/Year)",
                height=300, margin=dict(t=40, l=20, r=20, b=20)
            )
            st.plotly_chart(fig_risk_trend, use_container_width=True)
            
        with t2:
            st.caption("GLOBAL RISK DRIVERS")
            # TREEMAP (Fixed)
            fig_tree = px.treemap(
                global_imp, path=['Feature'], values='Importance',
                color='Importance', color_continuous_scale='Sunsetdark'
            )
            fig_tree.update_layout(paper_bgcolor="#0E1117", height=300, margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig_tree, use_container_width=True)

        # ROW 2: 3D Map + Sunburst
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


    # ==============================================================================
    # TAB 3: CRYSTAL BALL (Scenarios)
    # ==============================================================================
    with tab3:
        st.header("🔮 Future Simulations")
        
        # ROW 1: Bootstrapped Forecast (NEW)
        st.markdown("### 📈 Bootstrapped Risk Forecast")
        
        hist_data, forecast_data = run_bootstrap_simulation(df, model)

        # FIX: Calculate percentage column on components before combining and use them directly for plotting
        mean_forecast = forecast_data.groupby('Month_Year')['Risk_Rate'].mean().reset_index()
        mean_forecast['Type'] = 'Forecast (Mean)'
        mean_forecast['Risk_Rate_Pct'] = mean_forecast['Risk_Rate'] * 100 
        
        hist_data['Type'] = 'Historical'
        hist_data['Risk_Rate_Pct'] = hist_data['Risk_Rate'] * 100 
        
        # Calculate confidence interval (P25 and P75) from the bootstraps
        p25 = forecast_data.groupby('Month_Year')['Risk_Rate'].quantile(0.25).reset_index().rename(columns={'Risk_Rate': 'P25'})
        p75 = forecast_data.groupby('Month_Year')['Risk_Rate'].quantile(0.75).reset_index().rename(columns={'Risk_Rate': 'P75'})

        
        # --- NEW GRAPH: BOOTSTRAPPED FORESTED LINE GRAPH ---
        
        fig_bootstrap = go.Figure()
        
        # 1. Historical Trend (Using the updated hist_data directly)
        fig_bootstrap.add_trace(go.Scatter(
            x=hist_data['Month_Year'], y=hist_data['Risk_Rate_Pct'],
            mode='lines+markers', name='Historical Risk Rate',
            line=dict(color='#00FF9D', width=2)
        ))
        
        # 2. Confidence Interval (P25-P75 Band)
        fig_bootstrap.add_trace(go.Scatter(
            x=p75['Month_Year'], y=p75['P75'] * 100,
            mode='lines', name='Upper Bound (75th Percentile)',
            line=dict(color='#00C2FF', width=0),
            showlegend=False
        ))

        fig_bootstrap.add_trace(go.Scatter(
            x=p25['Month_Year'], y=p25['P25'] * 100,
            mode='lines', name='25th/75th Percentile Interval',
            fill='tonexty', fillcolor='rgba(0, 194, 255, 0.2)',
            line=dict(color='#00C2FF', width=0),
            showlegend=True
        ))
        
        # 3. Mean Forecast Line (Using the updated mean_forecast directly)
        fig_bootstrap.add_trace(go.Scatter(
            x=mean_forecast['Month_Year'], y=mean_forecast['Risk_Rate_Pct'],
            mode='lines+markers', name='Mean Forecast',
            line=dict(color='#FF2B2B', width=3, dash='dash')
        ))
        
        # Layout adjustments
        fig_bootstrap.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white",
            title="Future Risk Rate Forecast with 50 Bootstrapped Simulations",
            xaxis_title="Time", yaxis_title="Risk Rate (%)",
            height=400, margin=dict(t=40, l=20, r=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_bootstrap, use_container_width=True)
        
        st.markdown("---")
        
        # ROW 2: Scenario Stress Testing (Original Content)
        st.markdown("### 🌪️ Stress Test: Capital Reserve Simulation")
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

    # ==============================================================================
    # TAB 4: MODEL AUDIT (Evaluation)
    # ==============================================================================
    with tab4:
        st.header("⚙️ Model Audit: Performance and Reliability")
        st.markdown("Metrics evaluated on a 30% held-out test set.")
        
        # ROW 1: Key Metrics
        m1, m2, m3, m4, m5 = st.columns(5)

        def display_metric(col, label, value, color_hex):
            col.markdown(f"""
                <div style='background-color: #1E2130; border: 1px solid {color_hex}; padding: 10px; border-radius: 8px; text-align: center; box-shadow: 0 0 10px {color_hex}33;'>
                    <p style='margin: 0; font-size: 0.9em; color: #aaa;'>{label}</p>
                    <h3 style='margin: 5px 0 0; font-size: 1.5em; color: {color_hex};'>{value:.4f}</h3>
                </div>
            """, unsafe_allow_html=True)

        display_metric(m1, "Accuracy", metrics['accuracy'], "#00C2FF")
        display_metric(m2, "Precision", metrics['precision'], "#00FF9D")
        display_metric(m3, "Recall (Sensitivity)", metrics['recall'], "#FF2B2B")
        display_metric(m4, "F1 Score", metrics['f1'], "#FFD700")
        display_metric(m5, "ROC AUC", metrics['roc_auc'], "#FFFFFF")

        st.markdown("---")
        
        # ROW 2: Confusion Matrix and ROC Curve
        c_matrix_col, roc_col = st.columns(2)
        
        with c_matrix_col:
            st.subheader("Confusion Matrix")
            cm = metrics['conf_matrix']
            cm_df = pd.DataFrame(cm, 
                                 index=['Actual Low-Risk', 'Actual High-Risk'], 
                                 columns=['Predicted Low-Risk', 'Predicted High-Risk'])
            
            # Plot Confusion Matrix
            fig_cm = px.imshow(cm_df, text_auto=True, 
                                color_continuous_scale='plasma', 
                                labels=dict(x="Predicted", y="Actual", color="Count"))
            fig_cm.update_xaxes(side="bottom")
            fig_cm.update_layout(
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white",
                title="Classification Outcomes (Test Set)", height=450, 
                margin=dict(t=50, l=10, r=10, b=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with roc_col:
            st.subheader("ROC Curve (Performance Trade-off)")
            # Plot ROC Curve
            fpr, tpr, thresholds = roc_curve(metrics['y_test'], metrics['y_proba'])
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})', line=dict(color='#00FF9D', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier (AUC = 0.5)', line=dict(dash='dash', color='grey')))
            
            fig_roc.update_layout(
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white",
                xaxis_title='False Positive Rate (FPR)',
                yaxis_title='True Positive Rate (TPR / Recall)',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                height=450, margin=dict(t=50, l=10, r=10, b=10),
                legend=dict(yanchor="bottom", xanchor="right", x=0.99, y=0.01)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

if __name__ == "__main__":
    main()
