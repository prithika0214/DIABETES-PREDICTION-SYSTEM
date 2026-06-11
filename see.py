import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------
# Train Model Function
# ---------------------------
@st.cache_resource
def train_model():

    # Load dataset
    df = pd.read_csv("diabetes.csv")

    # Replace medically impossible zeros
    columns = [
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI'
    ]

    for col in columns:
        df[col] = df[col].replace(0, np.nan)

    # Features and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ML Pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42
        ))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


# Train model automatically
model, accuracy = train_model()

# ---------------------------
# Title
# ---------------------------
st.title("🩺 Diabetes Prediction System")
st.write(
    "Predict diabetes risk using patient medical information."
)

st.markdown("---")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Project Info")

st.sidebar.success(
    f"Model Accuracy: {accuracy*100:.2f}%"
)

st.sidebar.info("""
Machine Learning Model:
✔ Random Forest Classifier

Dataset:
✔ PIMA Diabetes Dataset
""")

# ---------------------------
# Input Fields
# ---------------------------
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=79
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Diabetes"):

    input_data = pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]], columns=[
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    confidence = np.max(probability) * 100

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk: Patient is likely Diabetic")
    else:
        st.success("✅ Low Risk: Patient is Not Diabetic")

    st.write(f"### Confidence Score: {confidence:.2f}%")

    # Risk Level
    if confidence < 60:
        risk = "Low"
    elif confidence < 80:
        risk = "Medium"
    else:
        risk = "High"

    st.write(f"### Risk Level: {risk}")

    # Suggestions
    st.subheader("Health Recommendation")

    if prediction == 1:
        st.warning("""
        - Reduce sugar intake
        - Exercise regularly
        - Monitor blood glucose
        - Consult doctor
        - Maintain healthy food habits
        """)
    else:
        st.success("""
        - Continue healthy lifestyle
        - Exercise regularly
        - Balanced diet
        - Regular health checkup
        """)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Built with Streamlit & Machine Learning")