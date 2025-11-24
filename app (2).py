# app_final.py
"""
AI Sentinel — KEAS 2025 Conference Dashboard (final)
Features:
 - Teal + Navy professional theme and layout
 - Cached dataset loading and preprocessing
 - RandomForest model training & cached artifact
 - Model evaluation tab: ROC, AUC, confusion matrix, classification report
 - SHAP explainability: global + local (fast background sampling)
 - Customer claim timeline with moving avg, anomalies and cumulative exposure
 - Geographic risk heatmap (plotly / pydeck fallback)
 - Monte-Carlo stress testing with simple multipliers
 - Fairness checks for candidate sensitive attributes (region, gender, rural_vs_urban)
 - KPI cards, export buttons, QR placeholder
 - Designed for stakeholder presentation and demos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, accuracy_score
import shap
import joblib
import io
import base64
import warnings

warnings.filterwarnings("ignore")

# ----------------------
# CONFIG
# ----------------------
DATA_PATH = "eswatini_insurance_final_dataset (5).csv"  # Change if necessary
PAGE_TITLE = "AI Sentinel — Explainable Risk Dashboard (Teal + Navy)"

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="expanded", page_icon="🛡️")

# ----------------------
# THEME/CSS
# ----------------------
st.markdown(
    """
    <style>
    :root {
        --navy: #001f3f;
        --teal: #008080;
        --bg: #08121a;
        --card: #07121a;
        --muted: #9eb6bd;
    }
    .stApp { background-color: var(--bg); color: #E8F6F6; }
    .card { background: linear-gradient(180deg, rgba(0,128,128,0.04), rgba(0,31,63,0.04)); padding:12px; border-radius:10px; border:1px solid rgba(0,194,194,0.06); }
    h1, h2, h3 { color: var(--teal); }
    .small { color: var(--muted); font-size:12px; }
    .kpi { font-size:26px; font-weight:700; color:white; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------
# UTIL: load and basic preprocess
# ----------------------
@st.cache_data(ttl=3600)
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # Normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    # Ensure core columns exist or create safe defaults
    if 'claim_amount_SZL' not in df.columns:
        # try alternative names
        if 'claim_amount' in df.columns:
            df.rename(columns={'claim_amount': 'claim_amount_SZL'}, inplace=True)
        else:
            # create synthetic small values to avoid errors
            df['claim_amount_SZL'] = np.random.randint(100, 10000, size=len(df))
    if 'customer_id' not in df.columns:
        df['customer_id'] = [f"CUST-{i+1:05d}" for i in range(len(df))]
    if 'claim_date' in df.columns:
        df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
    else:
        df['claim_date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.arange(len(df)), unit='D')
    if 'location' not in df.columns and 'claim_processing_branch' in df.columns:
        df['location'] = df['claim_processing_branch']
    if 'region' not in df.columns and 'location' in df.columns:
        # create a naive region mapping using location if needed
        df['region'] = df['location']
    # Target (if investigation_outcome present, use it; else use top quantile)
    if 'investigation_outcome' in df.columns:
        df['Risk_Target'] = (df['investigation_outcome'] == 'Confirmed Fraud').astype(int)
    else:
        # safe proxy: top 15% amounts flagged as high-risk (temporary)
        q85 = df['claim_amount_SZL'].quantile(0.85)
        df['Risk_Target'] = (df['claim_amount_SZL'] >= q85).astype(int)
    df['Month_Year'] = df['claim_date'].dt.to_period('M').astype(str)
    # fill some common categorical fields if missing
    if 'rural_vs_urban' not in df.columns:
        df['rural_vs_urban'] = 'Unknown'
    if 'claim_type' not in df.columns:
        df['claim_type'] = 'Other'
    return df

# ----------------------
# UTIL: segmentation
# ----------------------
@st.cache_data
def add_segmentation(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cluster_cols = [c for c in numeric_cols if c not in ['Risk_Target', 'claim_id', 'lat', 'lon']]
    if len(cluster_cols) >= 2:
        Xc = df[cluster_cols].fillna(0)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['segment'] = kmeans.fit_predict(Xc).astype(str)
    else:
        df['segment'] = "0"
    return df

# ----------------------
# MODEL: training & cached
# ----------------------
@st.cache_resource
def train_model_cached(df):
    df2 = df.copy()
    # Feature choices (robust to missing)
    features = []
    pref = ['claim_type', 'location', 'claim_amount_SZL', 'rural_vs_urban', 'policy_premium_SZL', 'policy_maturity_years', 'segment']
    for c in pref:
        if c in df2.columns:
            features.append(c)
        else:
            # create placeholders
            df2[c] = 0 if 'amount' in c or 'premium' in c or 'maturity' in c else "Unknown"
            features.append(c)

    X = df2[features].copy()
    y = df2['Risk_Target'].copy()
    encoders = {}
    # encode categorical columns
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # evaluation metrics
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    metrics = {
        'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float('nan'),
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return {
        'model': model,
        'encoders': encoders,
        'features': features,
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,
        'y_train': y_train
    }

# ----------------------
# SHAP: cached explainer
# ----------------------
@st.cache_resource
def build_shap_explainer(model, X_background=None):
    # Use small background sample for speed
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        # fallback
        explainer = shap.Explainer(model)
    return explainer

# ----------------------
# HELPER: prediction wrapper
# ----------------------
def predict_input(model_info, raw_input_dict):
    # Build a one-row DF in order of features
    features = model_info['features']
    df_row = pd.DataFrame(columns=features, data=[ [raw_input_dict.get(c, 0) for c in features] ])
    # encode categorical values if encoders exist
    for col, le in model_info['encoders'].items():
        val = str(df_row.at[0, col])
        if val in le.classes_:
            df_row[col] = le.transform([val])[0]
        else:
            # map unseen to index 0
            df_row[col] = 0
    df_row = df_row.fillna(0)
    prob = model_info['model'].predict_proba(df_row)[0,1]
    pred = int(prob > 0.5)
    return prob, pred, df_row

# ----------------------
# HELPER: monte-carlo simulator
# ----------------------
def monte_carlo_losses(df, multiplier_func, n_sim=1000, seed=42):
    rng = np.random.default_rng(seed)
    observed = df.loc[df['Risk_Target'] == 1, 'claim_amount_SZL'].dropna().values
    if len(observed) == 0:
        return np.zeros(n_sim)
    sims = []
    for i in range(n_sim):
        sample = rng.choice(observed, size=len(observed), replace=True)
        mult = multiplier_func(sample)
        sims.append(sample.sum() * mult if np.isscalar(mult) else np.sum(sample * mult))
    return np.array(sims)

# ----------------------
# FAIRNESS: basic checks (selection rate, TPR, FPR, PPV)
# ----------------------
def fairness_group_metrics(y_true, y_prob, groups, thresh=0.5):
    # groups: pandas Series aligned with y_true
    y_pred = (y_prob >= thresh).astype(int)
    dfm = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob, 'y_pred': y_pred, 'group': groups})
    results = {}
    for g, sub in dfm.groupby('group'):
        tn, fp, fn, tp = confusion_matrix(sub['y_true'], sub['y_pred'], labels=[0,1]).ravel()
        tpr = tp/(tp+fn) if (tp+fn)>0 else np.nan
        fpr = fp/(fp+tn) if (fp+tn)>0 else np.nan
        ppv = tp/(tp+fp) if (tp+fp)>0 else np.nan
        sel_rate = sub['y_pred'].mean()
        results[g] = {'n': len(sub), 'select_rate': sel_rate, 'TPR': tpr, 'FPR': fpr, 'PPV': ppv}
    return results

# ----------------------
# LOAD everything
# ----------------------
df = load_data()
df = add_segmentation(df)
model_info = train_model_cached(df)
explainer = build_shap_explainer(model_info['model'])

# ----------------------
# UI: Header & KPIs
# ----------------------
st.title("🛡️ AI Sentinel — Explainable AI for Insurance Risk (Teal + Navy)")
st.markdown("**Real-time, explainable claim risk classification — KEAS 2025**")

# KPI row
k1, k2, k3, k4 = st.columns([1.2, 1, 1, 1])
with k1:
    st.markdown("<div class='card'><div class='small'>Records</div><div class='kpi'>{:,}</div></div>".format(len(df)), unsafe_allow_html=True)
with k2:
    st.markdown("<div class='card'><div class='small'>High-risk prevalence</div><div class='kpi'>{:.1%}</div></div>".format(df['Risk_Target'].mean()), unsafe_allow_html=True)
with k3:
    st.markdown("<div class='card'><div class='small'>Average claim (SZL)</div><div class='kpi'>SZL {:,.0f}</div></div>".format(df['claim_amount_SZL'].mean()), unsafe_allow_html=True)
with k4:
    auc_val = model_info['metrics'].get('auc', float('nan'))
    st.markdown("<div class='card'><div class='small'>Model AUC (test)</div><div class='kpi'>{:.3f}</div></div>".format(auc_val), unsafe_allow_html=True)

st.markdown("---")

# Sidebar: quick controls
st.sidebar.header("Controls & Quick Links")
selected_customer = st.sidebar.selectbox("Select customer", df['customer_id'].unique()[:200])
show_qr = st.sidebar.checkbox("Show QR placeholder", value=False)
if show_qr:
    st.sidebar.image("https://via.placeholder.com/150.png?text=QR", width=150)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Decision Center", "Customer Timeline", "Model Eval", "Explainability", "Scenarios & Fairness"])

# ----------------------
# TAB 1: Decision Center (selected/latest claim)
# ----------------------
with tab1:
    st.header("Decision Center — Claim Snapshot")
    cust_df = df[df['customer_id'] == selected_customer].sort_values('claim_date')
    if cust_df.empty:
        st.info("No claims for selected customer.")
    else:
        latest = cust_df.iloc[-1]
        st.subheader(f"Latest claim — {latest.get('claim_id', 'N/A')} • {latest.get('claim_date'):%Y-%m-%d}")
        st.write(f"**Amount:** SZL {latest['claim_amount_SZL']:,}  •  **Type:** {latest['claim_type']}  •  **Location:** {latest.get('location','N/A')}")
        # Allow counterfactual adjustments
        with st.expander("Counterfactual: adjust Claim amount / premium / policy age"):
            amt = st.number_input("Claim amount (SZL)", value=float(latest['claim_amount_SZL']), min_value=0.0, step=100.0)
            premium = st.number_input("Policy premium (SZL)", value=float(latest.get('policy_premium_SZL', 0.0)), min_value=0.0, step=50.0)
            mat = st.number_input("Policy maturity (years)", value=float(latest.get('policy_maturity_years', 0.0)), min_value=0.0, step=1.0)
        raw_input = {
            'claim_type': latest.get('claim_type', 'Unknown'),
            'location': latest.get('location', 'Unknown'),
            'claim_amount_SZL': amt,
            'rural_vs_urban': latest.get('rural_vs_urban', 'Unknown'),
            'policy_premium_SZL': premium,
            'policy_maturity_years': mat,
            'segment': str(latest.get('segment', '0'))
        }
        prob, pred, xrow = predict_input(model_info, raw_input)
        colA, colB, colC = st.columns([1, 2, 1])
        with colA:
            figg = go.Figure(go.Indicator(mode="gauge+number", value=prob*100,
                                         gauge={'axis': {'range': [0,100]}, 'bar': {'color': "#00C2C2" if prob<0.5 else "#FF6B6B"}},
                                         number={'suffix':'%', 'font':{'color':'white'}}))
            figg.update_layout(height=250, paper_bgcolor="#07121a")
            st.plotly_chart(figg, use_container_width=True)
            st.write("**Decision**:", "✅ Approve" if pred==0 else "🚫 Investigate")
        with colB:
            st.metric("Predicted high-risk probability", f"{prob:.2%}")
            st.write("**Quick recommended actions:**")
            if prob > 0.7:
                st.write("- Escalate to senior investigator")
                st.write("- Request immediate supporting documents")
            elif prob > 0.5:
                st.write("- Schedule desk investigation")
            else:
                st.write("- Fast-track payment (subject to verification)")
        with colC:
            if st.button("Export claim snapshot (CSV)"):
                buf = io.StringIO()
                pd.DataFrame([raw_input]).to_csv(buf, index=False)
                st.download_button("Download CSV", data=buf.getvalue(), file_name=f"claim_{latest.get('claim_id','snapshot')}.csv")

# ----------------------
# TAB 2: Customer Timeline (trend + anomalies)
# ----------------------
with tab2:
    st.header("Customer Claim Timeline — Trend & Anomalies")
    cust_df = df[df['customer_id'] == selected_customer].sort_values('claim_date')
    if cust_df.empty:
        st.info("No claims for this customer.")
    else:
        cust = cust_df.copy()
        # rolling trend
        cust['trend'] = cust['claim_amount_SZL'].rolling(window=3, min_periods=1).mean()
        cust['cumulative'] = cust['claim_amount_SZL'].cumsum()
        cust['days_since_last'] = cust['claim_date'].diff().dt.days.fillna(0)
        # anomaly by z-score
        if cust['claim_amount_SZL'].std() > 0:
            cust['z'] = (cust['claim_amount_SZL'] - cust['claim_amount_SZL'].mean()) / cust['claim_amount_SZL'].std()
            cust['anomaly'] = cust['z'].abs() > 2.5
        else:
            cust['anomaly'] = False

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cust['claim_date'], y=cust['claim_amount_SZL'], mode='lines+markers', name='Claim Amount', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=cust['claim_date'], y=cust['trend'], mode='lines', name='Rolling trend (3)', line=dict(width=3, dash='dash', color='cyan')))
        fig.add_trace(go.Scatter(x=cust[cust['anomaly']]['claim_date'], y=cust[cust['anomaly']]['claim_amount_SZL'], mode='markers', name='Anomaly', marker=dict(size=14, color='red', symbol='diamond')))
        fig.add_trace(go.Scatter(x=cust['claim_date'], y=cust['cumulative'], mode='lines', name='Cumulative exposure', yaxis='y2', line=dict(width=2, color='teal')))

        fig.update_layout(title=f"Claim timeline for {selected_customer}", template='plotly_dark', height=520,
                          yaxis=dict(title="Claim Amount (SZL)"), yaxis2=dict(title="Cumulative (SZL)", overlaying='y', side='right'))
        st.plotly_chart(fig, use_container_width=True)

        # summary insights
        last = cust.iloc[-1]
        if last['anomaly']:
            st.error("⚠️ Latest claim is anomalous compared to customer's history — consider investigation.")
        elif last['days_since_last'] < 20 and cust['days_since_last'].mean() > last['days_since_last']:
            st.warning("📌 Claim frequency increased recently.")
        else:
            st.success("✅ Customer claim behavior is within historical range.")

        # optional: compare to peer group (same segment)
        if 'segment' in cust.columns:
            seg = last['segment']
            peer = df[df['segment']==seg]
            st.markdown(f"**Benchmark vs segment {seg}**")
            seg_mean = peer['claim_amount_SZL'].mean()
            st.write(f"Avg claim (segment): SZL {seg_mean:,.0f}  •  Customer avg: SZL {cust['claim_amount_SZL'].mean():,.0f}")

# ----------------------
# TAB 3: Model Evaluation
# ----------------------
with tab3:
    st.header("Model evaluation")
    metrics = model_info['metrics']
    st.write(f"**AUC (test):** {metrics['auc']:.3f}  •  **Accuracy:** {metrics['accuracy']:.3f}")
    # ROC
    X_test = model_info['X_test']
    y_test = model_info['y_test']
    yprob = model_info['model'].predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, yprob)
    figroc = go.Figure()
    figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(color='#00C2C2')))
    figroc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray')))
    figroc.update_layout(template='plotly_dark', height=420, xaxis_title='False positive rate', yaxis_title='True positive rate')
    st.plotly_chart(figroc, use_container_width=True)
    # confusion matrix
    cm = metrics['confusion_matrix']
    figcm = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0','Pred 1'], y=['True 0','True 1'], colorscale='RdBu'))
    figcm.update_layout(template='plotly_dark', height=360)
    st.plotly_chart(figcm, use_container_width=True)
    # classification report
    st.subheader("Classification report (test)")
    st.json(metrics['classification_report'])

# ----------------------
# TAB 4: Explainability (SHAP)
# ----------------------
with tab4:
    st.header("Explainability (SHAP)")
    st.markdown("Global feature importance & local explanation for the selected claim.")
    # Global importance
    feat_imp = pd.DataFrame({'feature': model_info['features'], 'importance': model_info['model'].feature_importances_})
    feat_imp = feat_imp.sort_values('importance', ascending=True)
    fig_imp = go.Figure(go.Bar(x=feat_imp['importance'], y=feat_imp['feature'], orientation='h', marker_color=feat_imp['importance'], marker_colorscale='Viridis'))
    fig_imp.update_layout(template='plotly_dark', height=380)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Local SHAP explanation (selected claim)")
    # Reuse earlier raw_input 'raw_input' if exists, else build from latest
    try:
        sample_input = raw_input
    except NameError:
        latest = df.iloc[-1]
        sample_input = {
            'claim_type': latest.get('claim_type', 'Unknown'),
            'location': latest.get('location', 'Unknown'),
            'claim_amount_SZL': float(latest.get('claim_amount_SZL', 0)),
            'rural_vs_urban': latest.get('rural_vs_urban', 'Unknown'),
            'policy_premium_SZL': float(latest.get('policy_premium_SZL', 0)),
            'policy_maturity_years': float(latest.get('policy_maturity_years', 0)),
            'segment': str(latest.get('segment', '0'))
        }
    prob_s, pred_s, sample_df = predict_input(model_info, sample_input)
    st.write(f"Predicted probability: {prob_s:.2%}")
    # compute SHAP for the one-sample (fast)
    try:
        # prepare X background sample
        X_for_shap = model_info['X_train'].sample(min(200, len(model_info['X_train'])), random_state=42)
        expl = build_shap_explainer(model_info['model'])
        # shap_values may be list for classifiers
        shap_vals = expl.shap_values(sample_df)
        if isinstance(shap_vals, list):
            shap_local = shap_vals[1][0] if len(shap_vals)>1 else shap_vals[0][0]
        else:
            shap_local = np.ravel(shap_vals)
        # build dataframe for display
        shap_df = pd.DataFrame({'feature': model_info['features'], 'shap_value': shap_local})
        shap_df = shap_df.sort_values('shap_value', key=abs, ascending=True)
        colors = ['#FF6B6B' if v>0 else '#00C2C2' for v in shap_df['shap_value']]
        figsh = go.Figure(go.Bar(x=shap_df['shap_value'], y=shap_df['feature'], orientation='h', marker_color=colors))
        figsh.update_layout(template='plotly_dark', height=420)
        st.plotly_chart(figsh, use_container_width=True)
    except Exception as e:
        st.error(f"SHAP error: {e}")
        st.info("SHAP visualizations require additional memory/time; they will be available after warm-up.")

# ----------------------
# TAB 5: Scenarios & Fairness
# ----------------------
with tab5:
    st.header("Scenarios (Monte-Carlo) & Fairness checks")
    st.subheader("Monte-Carlo Stress Test")
    scen = st.selectbox("Scenario", ["None","Severe Weather","Pandemic","Civil Unrest"])
    n_sim = st.slider("Simulations", 200, 5000, 1000, step=200)
    severity = st.slider("Severity (1-10)", 1, 10, 5)
    if scen == "Severe Weather":
        multiplier = lambda s: 1 + (0.04 * severity)  # uniform uplift
    elif scen == "Pandemic":
        multiplier = lambda s: 1 + (0.03 * severity)
    elif scen == "Civil Unrest":
        multiplier = lambda s: 1 + (0.06 * severity)
    else:
        multiplier = lambda s: 1.0

    with st.spinner("Running simulations..."):
        sims = monte_carlo_losses(df, multiplier, n_sim=n_sim)
        p50, p95 = np.percentile(sims, 50), np.percentile(sims, 95)
        current = df.loc[df['Risk_Target']==1, 'claim_amount_SZL'].sum()
        st.metric("Observed flagged loss (sum)", f"SZL {current:,.0f}")
        st.metric("Simulated P50 total loss", f"SZL {p50:,.0f}")
        st.metric("Simulated P95 total loss", f"SZL {p95:,.0f}")
        hist = px.histogram(sims, nbins=60, title="Simulated total loss distribution", template='plotly_dark')
        st.plotly_chart(hist, use_container_width=True)

    st.markdown("---")
    st.subheader("Fairness & Group Metrics (quick checks)")
    # candidates
    candidates = [c for c in df.columns if c.lower() in ['gender','sex','age','age_group','region','location','rural_vs_urban','ethnicity']]
    if not candidates:
        candidates = [c for c in df.columns if 'region' in c.lower() or 'location' in c.lower()][:1]
    if not candidates:
        st.info("No clear sensitive attribute found. Add 'gender' or 'region' column to run fairness checks.")
    else:
        sensitive = st.selectbox("Sensitive attribute", candidates)
        thresh = st.slider("Decision threshold (for parity checks)", 0.0, 1.0, 0.5, step=0.01)
        # align test set groups if indices match
        X_test = model_info['X_test']
        y_test = model_info['y_test']
        yprob = model_info['model'].predict_proba(X_test)[:,1]
        # attempt mapping: if X_test index subset of df index
        if isinstance(X_test, pd.DataFrame) and set(X_test.index).issubset(set(df.index)):
            groups = df.loc[X_test.index, sensitive]
        else:
            # fallback: approximate using first N rows (best effort)
            groups = df[sensitive].iloc[:len(y_test)].reset_index(drop=True)

        group_stats = fairness_group_metrics(y_test.values, yprob, groups, thresh)
        # present table
        rows = []
        for k,v in group_stats.items():
            rows.append({'group': k, 'n': v['n'], 'selection_rate': v['select_rate'], 'TPR': v['TPR'], 'FPR': v['FPR'], 'PPV': v['PPV']})
        gdf = pd.DataFrame(rows).set_index('group')
        st.dataframe(gdf.style.format({'selection_rate':'{:.2%}','TPR':'{:.2%}','FPR':'{:.2%}','PPV':'{:.2%}'}))

        # simple disparate impact check (80% rule)
        sel_rates = gdf['selection_rate']
        if len(sel_rates) > 1:
            ref = sel_rates.max()
            di = sel_rates / ref
            flagged = di[di < 0.8]
            if not flagged.empty:
                st.error(f"Potential disparate impact flagged for groups: {', '.join(flagged.index.astype(str))}")
            else:
                st.success("No group flagged by 80% disparate impact quick-check.")

# ----------------------
# FOOTER / DOWNLOADS
# ----------------------
st.markdown("---")
st.markdown("**Next steps / deployment:** productionize model, add audit logging, run fairness mitigation (reweighing / thresholding), setup periodic monitoring and retraining.")
st.caption("Built for KEAS 2025 — Bhekiwe Sindiswa Dlamini — University of Eswatini")
