import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report


DATA_PATH_RAW = os.path.join("data", "raw", "synthetic_coffee_health_10000.csv")
DATA_PATH_PROCESSED = os.path.join("data", "processed", "coffee_health_cleaned.csv")

def resolve_data_path() -> str:
    if os.path.exists(DATA_PATH_RAW):
        return DATA_PATH_RAW
    if os.path.exists(DATA_PATH_PROCESSED):
        return DATA_PATH_PROCESSED
    # Fallback to legacy location if present
    legacy = os.path.join("data", "synthetic_coffee_health_10000.csv")
    return legacy


# -------------------------------
# Utilities: Feature Engineering
# -------------------------------
def categorize_bmi(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"


def activity_level(hours: float) -> str:
    if hours < 2:
        return "Low"
    elif hours < 5:
        return "Moderate"
    return "High"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = df.copy()
    # Create engineered features mirroring the notebook
    df_fe["BMI_Category"] = df_fe["BMI"].apply(categorize_bmi)
    df_fe["Sleep_Efficiency"] = df_fe["Sleep_Hours"] / 8.0
    df_fe["Caffeine_per_Cup"] = df_fe["Caffeine_mg"] / df_fe["Coffee_Intake"].replace(0, 1)
    df_fe["Activity_Level"] = df_fe["Physical_Activity_Hours"].apply(activity_level)
    return df_fe


# ------------------------------------
# Data Loading and Preprocessing Cache
# ------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop_duplicates().reset_index(drop=True)


@st.cache_resource(show_spinner=True)
def train_model(df_raw: pd.DataFrame):
    df_clean = df_raw.copy()

    # Basic cleaning: handle missing values
    # Fill categorical NaNs with 'Unknown', numeric NaNs with column medians
    obj_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if obj_cols:
        df_clean[obj_cols] = df_clean[obj_cols].fillna("Unknown")
    if num_cols:
        df_clean[num_cols] = df_clean[num_cols].apply(lambda s: s.fillna(s.median()))

    # Identify categorical columns (object dtype), exclude target
    categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    if "Health_Issues" in categorical_cols:
        categorical_cols.remove("Health_Issues")

    # Encode categoricals with LabelEncoder (fit on full dataset for simplicity)
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[f"{col}_encoded"] = le.fit_transform(df_clean[col])
        label_encoders[col] = le

    # Target encoding
    le_target = LabelEncoder()
    df_clean["Health_Issues_encoded"] = le_target.fit_transform(df_clean["Health_Issues"])

    # Feature engineering
    df_features = engineer_features(df_clean)

    # Encode engineered categoricals
    for col in ["BMI_Category", "Activity_Level"]:
        le = LabelEncoder()
        df_features[f"{col}_encoded"] = le.fit_transform(df_features[col])
        label_encoders[col] = le

    # Risk score (requires Stress_Level encoding). If not present yet, encode Stress_Level
    if "Stress_Level_encoded" not in df_features.columns and "Stress_Level" in df_features.columns:
        le = LabelEncoder()
        df_features["Stress_Level_encoded"] = le.fit_transform(df_features["Stress_Level"])
        label_encoders["Stress_Level"] = le

    df_features["Risk_Score"] = (
        df_features["Stress_Level_encoded"] * 0.3
        + (df_features["BMI"] - 22) * 0.1
        + (df_features["Heart_Rate"] - 70) * 0.05
        + df_features["Smoking"] * 0.3
        + df_features["Alcohol_Consumption"] * 0.1
    )

    # Ensure all feature columns exist to prevent KeyError
    required_feature_cols = [
        "Age",
        "Coffee_Intake",
        "Caffeine_mg",
        "Sleep_Hours",
        "BMI",
        "Heart_Rate",
        "Physical_Activity_Hours",
        "Gender_encoded",
        "Country_encoded",
        "Sleep_Quality_encoded",
        "Stress_Level_encoded",
        "Occupation_encoded",
        "Smoking",
        "Alcohol_Consumption",
        "BMI_Category_encoded",
        "Sleep_Efficiency",
        "Caffeine_per_Cup",
        "Risk_Score",
        "Activity_Level_encoded",
    ]
    for col in required_feature_cols:
        if col not in df_features.columns:
            # default safe values
            df_features[col] = 0

    # Feature set aligned with notebook
    feature_cols = [
        "Age",
        "Coffee_Intake",
        "Caffeine_mg",
        "Sleep_Hours",
        "BMI",
        "Heart_Rate",
        "Physical_Activity_Hours",
        "Gender_encoded",
        "Country_encoded",
        "Sleep_Quality_encoded",
        "Stress_Level_encoded",
        "Occupation_encoded",
        "Smoking",
        "Alcohol_Consumption",
        "BMI_Category_encoded",
        "Sleep_Efficiency",
        "Caffeine_per_Cup",
        "Risk_Score",
        "Activity_Level_encoded",
    ]

    # Train/test split
    # Sanitize any infs/nans before training
    X = df_features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_features["Health_Issues_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # RandomForest (robust without feature scaling)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Fit a scaler for optional use with future models; kept for parity with notebook
    scaler = StandardScaler()
    scaler.fit(X_train)

    return {
        "model": model,
        "scaler": scaler,
        "encoders": label_encoders,
        "target_encoder": le_target,
        "feature_cols": feature_cols,
        "train_metrics": {"accuracy": acc, "f1": f1},
        "df_features": df_features,
    }


def build_sample_row(
    age: int,
    coffee_intake: float,
    sleep_hours: float,
    bmi: float,
    heart_rate: int,
    stress_level: str,
    physical_activity: float,
    smoking: int,
    alcohol: int,
    gender: str,
    country: str,
    sleep_quality: str,
    occupation: str,
    encoders: Dict[str, LabelEncoder],
) -> pd.DataFrame:
    caffeine_mg = coffee_intake * 95.0
    bmi_category = categorize_bmi(bmi)
    sleep_efficiency = sleep_hours / 8.0
    caffeine_per_cup = 95.0 if coffee_intake == 0 else caffeine_mg / coffee_intake
    activity = activity_level(physical_activity)

    row = {
        "Age": age,
        "Coffee_Intake": coffee_intake,
        "Caffeine_mg": caffeine_mg,
        "Sleep_Hours": sleep_hours,
        "BMI": bmi,
        "Heart_Rate": heart_rate,
        "Physical_Activity_Hours": physical_activity,
        "Smoking": smoking,
        "Alcohol_Consumption": alcohol,
        "Sleep_Efficiency": sleep_efficiency,
        "Caffeine_per_Cup": caffeine_per_cup,
    }

    # Encode categorical inputs using fitted encoders. If unseen, fall back gracefully.
    def safe_encode(col: str, value: str) -> int:
        le = encoders.get(col)
        if le is None:
            # Fit a temporary encoder with this single value
            le = LabelEncoder().fit([value])
        try:
            return int(le.transform([value])[0])
        except ValueError:
            # If unseen, extend classes dynamically
            classes = list(le.classes_)
            classes.append(value)
            le.classes_ = np.array(sorted(set(classes)))
            return int(le.transform([value])[0])

    row["Gender_encoded"] = safe_encode("Gender", gender)
    row["Country_encoded"] = safe_encode("Country", country)
    row["Sleep_Quality_encoded"] = safe_encode("Sleep_Quality", sleep_quality)
    row["Stress_Level_encoded"] = safe_encode("Stress_Level", stress_level)
    row["Occupation_encoded"] = safe_encode("Occupation", occupation)
    row["BMI_Category_encoded"] = safe_encode("BMI_Category", bmi_category)
    row["Activity_Level_encoded"] = safe_encode("Activity_Level", activity)

    row["Risk_Score"] = (
        row["Stress_Level_encoded"] * 0.3
        + (bmi - 22) * 0.1
        + (heart_rate - 70) * 0.05
        + smoking * 0.3
        + alcohol * 0.1
    )

    return pd.DataFrame([row])


# ---------------
# Streamlit App
# ---------------
st.set_page_config(page_title="Health Risk Prediction", layout="wide")

st.title("Predictive Health Risk Analysis")
st.caption(
    "Predict health risk levels (None, Mild, Moderate) from lifestyle factors."
)

with st.sidebar:
    st.header("Configuration")
    st.write("Data source:")
    st.code(resolve_data_path())

df = load_data(resolve_data_path())
model_artifacts = train_model(df)

tabs = st.tabs(["Predict", "EDA", "Model"])


# ----------------
# Predict Tab
# ----------------
with tabs[0]:
    st.subheader("Interactive Prediction")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=90, value=35, step=1)
            coffee_intake = st.number_input(
                "Coffee Intake (cups/day)", min_value=0.0, max_value=10.0, value=3.0, step=0.1
            )
            sleep_hours = st.number_input(
                "Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.1
            )
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
        with col2:
            heart_rate = st.number_input(
                "Resting Heart Rate", min_value=40, max_value=120, value=72, step=1
            )
            stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
            sleep_quality = st.selectbox("Sleep Quality", ["Fair", "Good", "Excellent"])
            physical_activity = st.number_input(
                "Physical Activity (hrs/week)", min_value=0.0, max_value=30.0, value=3.0, step=0.5
            )
        with col3:
            smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            alcohol = st.selectbox("Alcohol Consumption", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            gender = st.selectbox("Gender", sorted(df["Gender"].unique().tolist()))
            country = st.selectbox("Country", sorted(df["Country"].unique().tolist()))
            occupation = st.selectbox("Occupation", sorted(df["Occupation"].unique().tolist()))

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        sample = build_sample_row(
            age=age,
            coffee_intake=coffee_intake,
            sleep_hours=sleep_hours,
            bmi=bmi,
            heart_rate=heart_rate,
            stress_level=stress_level,
            physical_activity=physical_activity,
            smoking=int(smoking),
            alcohol=int(alcohol),
            gender=gender,
            country=country,
            sleep_quality=sleep_quality,
            occupation=occupation,
            encoders=model_artifacts["encoders"],
        )

        # Align sample columns order to feature_cols
        X_sample = sample[model_artifacts["feature_cols"]].replace([np.inf, -np.inf], np.nan).fillna(0)
        model: RandomForestClassifier = model_artifacts["model"]
        target_encoder: LabelEncoder = model_artifacts["target_encoder"]
        proba = model.predict_proba(X_sample)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = target_encoder.inverse_transform([pred_idx])[0]

        st.success(f"Predicted Risk: {pred_label}")
        cols = st.columns(3)
        for i, cls in enumerate(target_encoder.classes_):
            with cols[i % 3]:
                st.metric(cls, f"{proba[i]*100:.1f}%")

        with st.expander("Sample features"):
            st.dataframe(sample)


# ----------------
# EDA Tab
# ----------------
with tabs[1]:
    st.subheader("Exploratory Data Analysis")
    df_fe = model_artifacts["df_features"]
    col1, col2 = st.columns(2)
    with col1:
        st.write("Health Issues Distribution")
        st.bar_chart(df_fe["Health_Issues"].value_counts())
    with col2:
        st.write("Average Coffee Intake by Health Issues")
        st.bar_chart(df_fe.groupby("Health_Issues")["Coffee_Intake"].mean())

    st.write("Numerical Summary")
    st.dataframe(df_fe[[
        "Age",
        "Coffee_Intake",
        "Caffeine_mg",
        "Sleep_Hours",
        "BMI",
        "Heart_Rate",
        "Physical_Activity_Hours",
    ]].describe().transpose())


# ----------------
# Model Tab
# ----------------
with tabs[2]:
    st.subheader("Model Performance")
    metrics = model_artifacts["train_metrics"]
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    mcol2.metric("F1-Score (weighted)", f"{metrics['f1']:.3f}")

    st.write("Classes:", ", ".join(model_artifacts["target_encoder"].classes_))


