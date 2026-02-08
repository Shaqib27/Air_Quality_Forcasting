import streamlit as st
import pandas as pd
import numpy as np
import joblib

# IMPORT CLEANING FUNCTION
from data_cleaning import clean_single_airdata_df

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Air Quality Forecasting",
    layout="wide"
)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    try:
        return joblib.load("aqi_lgb_model1.pkl")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# ===============================
# FEATURE GENERATION
# ===============================
def create_lag_features(df):
    df = df.copy()

    df["pm25_lag_3"] = df["pm25"].shift(3)
    df["pm25_lag_6"] = df["pm25"].shift(6)
    df["pm25_lag_24"] = df["pm25"].shift(24)

    df["pm25_roll_6"] = df["pm25"].rolling(6).mean()
    df["pm25_roll_24"] = df["pm25"].rolling(24).mean()

    return df
def align_features_with_model(df, model):
    """
    Ensures all features used during training exist at inference time.
    Missing features are filled with 0.
    """
    required_features = model.feature_names_in_.tolist()

    for col in required_features:
        if col not in df.columns:
            df[col] = 0.0

    return df, required_features


# ===============================
# FORECAST FUNCTION
# ===============================
def forecast_next_n_steps(model, history_df, feature_columns, n_steps=24):
    history = history_df.copy()
    forecasts = []

    for step in range(n_steps):
        # Recreate lag features at every step
        history = create_lag_features(history)

        # Align features with model
        history, _ = align_features_with_model(history, model)

        # Take the latest row
        X = history.iloc[-1:][feature_columns]

        prediction = model.predict(X)[0]
        forecasts.append(prediction)

        # Append prediction as next pm25
        next_row = history.iloc[-1:].copy()
        next_row["pm25"] = prediction

        history = pd.concat([history, next_row])

    return forecasts



# ===============================
# UI
# ===============================
st.title("🌬️ PM2.5 Air Quality Forecasting Dashboard")

if model is None:
    st.stop()

st.sidebar.header("Forecast Settings")
n_hours = st.sidebar.slider("Forecast Horizon (Hours)", 1, 48, 24)

uploaded_file = st.file_uploader(
    "Upload CPCB CSV file",
    type="csv"
)

# ===============================
# FILE PROCESSING
# ===============================
if uploaded_file:

    # 1️⃣ LOAD RAW CSV
    raw_df = pd.read_csv(uploaded_file)

    # 2️⃣ CLEAN USING data_cleaning.py
    try:
        df = clean_single_airdata_df(raw_df)
    except Exception as e:
        st.error(f"Data cleaning failed: {e}")
        st.stop()

    # 3️⃣ STANDARDIZE DATE FOR TIME SERIES
    df.rename(columns={"from_date": "date"}, inplace=True)
    df.set_index("date", inplace=True)
    df = df.sort_index()

    # 4️⃣ CREATE LAG & ROLLING FEATURES
    df = create_lag_features(df)
    df.dropna(inplace=True)

    # ===============================
    # PREVIEW
    # ===============================
    st.subheader("Cleaned Data Preview")
    st.dataframe(df.tail())

    # ===============================
    # RUN FORECAST
    # ===============================
    if st.button("Run Forecast"):

        df, trained_features = align_features_with_model(df, model)

        history = df.iloc[-30:].copy()

        with st.spinner("Forecasting air quality..."):
            predictions = forecast_next_n_steps(
                model,
                history,
                trained_features,
                n_steps=n_hours
            )

        # ===============================
        # RESULTS
        # ===============================
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"📈 Next {n_hours} Hours PM2.5 Forecast")
            forecast_df = pd.DataFrame(
                {"Predicted PM2.5": predictions}
            )
            st.line_chart(forecast_df)

        with col2:
            st.subheader("📊 Summary")

            avg_pm = np.mean(predictions)
            st.metric(
                "Average PM2.5",
                f"{avg_pm:.2f} µg/m³"
            )

            if avg_pm < 12:
                quality = "🟢 Good"
            elif avg_pm < 35:
                quality = "🟡 Moderate"
            else:
                quality = "🔴 Unhealthy"

            st.write(f"Estimated Air Quality: **{quality}**")

        st.success("Forecast generated successfully ✅")


