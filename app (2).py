import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from io import BytesIO
from datetime import datetime
from PIL import ImageGrab
import base64

# App configuration
st.set_page_config(layout="wide", page_title="Eswatini Insurance Risk Dashboard", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    body {
        background-color: #111;
        color: #EEE;
    }
    .stApp {
        background-color: #111;
    }
    .css-1d391kg, .css-1v3fvcr, .st-cw, .st-bf {
        color: #EEE;
    }
    </style>
""", unsafe_allow_html=True)

# Simple authentication
st.title("🔒 Login to Access Dashboard")
password = st.text_input("Enter Password", type="password")
if password != "eswatini2024":
    st.stop()

st.title("🇸🇿 Eswatini Insurance Risk Intelligence Portal")

uploaded_file = st.file_uploader("📂 Upload insurance dataset (CSV)", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower().str.strip()
    return df

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Dataset loading failed: {str(e)}")
        st.stop()
else:
    st.warning("⚠️ Upload a dataset to view analytics.")
    st.stop()

# Preprocessing
default_cols = ['age', 'income', 'region', 'gender', 'risk']
for col in default_cols:
    if col not in df.columns:
        df[col] = 'Unknown' if col in ['region', 'gender', 'risk'] else 0

# Fill missing values
df['region'] = df['region'].fillna('Unknown')
df['income'] = pd.to_numeric(df['income'], errors='coerce').fillna(df['income'].median())
df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
df['risk'] = df['risk'].fillna("Low")

# Convert dates if any
date_cols = [col for col in df.columns if 'date' in col]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

num_cols = ['age', 'income']
cat_cols = ['region', 'gender']

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', 'passthrough', cat_cols)
])

# Encode categorical variables
encoders = {}
for col in cat_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc

X = df[num_cols + cat_cols]
y = df['risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Region filter
selected_region = st.selectbox("🌍 Filter by Region:", df['region'].unique())
df_filtered = df[df['region'] == selected_region]

# Dashboard layout
row1 = st.columns((1, 2, 1))
row2 = st.columns((2, 2))
row3 = st.columns((1, 1))

with row1[0]:
    st.metric("🧾 Total Policies", len(df_filtered))
    st.metric("⚠️ High-Risk %", f"{(df_filtered['risk'] == 'High').mean() * 100:.1f}%")
with row1[1]:
    st.metric("✅ Model Accuracy", f"{model.score(X_test, y_test) * 100:.2f}%")
    st.metric("🧹 Missing Values Imputed", df.isna().sum().sum())

with row2[0]:
    st.subheader("🗺️ Regional Risk Heatmap - Eswatini")
    @st.cache_data
    def load_regions():
        return gpd.read_file("eswatini_regions.geojson")

    try:
        gdf = load_regions()
        region_risk = df.groupby('region')['risk'].apply(lambda x: (x == 'High').mean()).reset_index()
        region_risk.columns = ['region', 'high_risk_percent']
        gdf['region'] = gdf['region'].str.lower()
        region_risk['region'] = region_risk['region'].str.lower()
        merged = gdf.merge(region_risk, on='region')

        fig = px.choropleth_mapbox(
            merged,
            geojson=merged.geometry,
            locations=merged.index,
            color='high_risk_percent',
            color_continuous_scale="Reds",
            mapbox_style="carto-darkmatter",
            center={"lat": -26.5, "lon": 31.5},
            zoom=6,
            opacity=0.7
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Map load failed: {e}")

if date_cols:
    st.subheader("📆 Time-based Trend of Risk Cases")
    time_col = st.selectbox("Select Date Column", date_cols)
    trend = df.groupby(df[time_col].dt.to_period("M"))['risk'].value_counts().unstack().fillna(0)
    trend.index = trend.index.astype(str)
    fig_trend = px.line(trend, x=trend.index, y=trend.columns, labels={'value': 'Count', 'index': 'Month'})
    st.plotly_chart(fig_trend, use_container_width=True)

with row2[1]:
    st.subheader("📊 Risk Distribution")
    feature = st.radio("Group by:", ['age', 'income'], horizontal=True)
    fig2 = px.histogram(df_filtered, x=feature, color='risk', barmode='group', template='plotly_dark')
    st.plotly_chart(fig2, use_container_width=True)

with row3[0]:
    st.subheader("📌 Risk by Gender")
    pie = df_filtered.groupby('gender')['risk'].value_counts(normalize=True).unstack().fillna(0)
    fig3 = px.pie(values=pie['High'], names=pie.index, title='High Risk % by Gender', template='plotly_dark')
    st.plotly_chart(fig3)

with row3[1]:
    st.subheader("📌 Risk by Region")
    pie2 = df.groupby('region')['risk'].value_counts(normalize=True).unstack().fillna(0)
    fig4 = px.pie(values=pie2['High'], names=pie2.index, title='High Risk % by Region', template='plotly_dark')
    st.plotly_chart(fig4)

st.subheader("📈 SHAP Summary - Feature Importance")
fig_shap, ax = plt.subplots(figsize=(10, 4))
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig_shap)

st.subheader("🧠 Predict Risk for New Customer")
with st.form("predict_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", value=50000.0)
    region = st.selectbox("Region", encoders['region'].classes_)
    gender = st.radio("Gender", encoders['gender'].classes_)
    submit = st.form_submit_button("Predict")

    if submit:
        new_data = pd.DataFrame([{
            'age': age,
            'income': income,
            'region': encoders['region'].transform([region])[0],
            'gender': encoders['gender'].transform([gender])[0]
        }])
        prediction = model.predict(new_data)[0]
        st.success(f"Predicted Risk Level: **{prediction}**")

st.download_button("📥 Download Cleaned Dataset", df.to_csv(index=False), "eswatini_cleaned_data.csv")

st.subheader("📸 Download Dashboard Snapshot")
snapshot_button = st.button("Download Dashboard Screenshot")
if snapshot_button:
    st.warning("📷 Screenshots must be taken from your local browser or operating system.")
    st.info("You can use tools like Windows Snipping Tool, macOS Screenshot, or browser extensions like GoFullPage to save the entire dashboard as a PDF or image.")
