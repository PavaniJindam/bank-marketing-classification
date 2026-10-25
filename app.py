
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Prediction")
st.write(
    "Analyze customer details and predict whether a customer "
    "is likely to subscribe to a bank term deposit."
)

# Load the trained machine-learning pipeline
MODEL_PATH = Path(__file__).parent / "model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Please add model.pkl "
        "to the same GitHub repository as app.py."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# Navigation
page = st.sidebar.radio(
    "Choose a page",
    ["Home", "Customer Prediction", "Model Comparison"]
)

if page == "Home":
    st.header("Project Overview")

    st.write("""
    This project compares machine-learning classification algorithms
    for predicting bank term-deposit subscriptions.

    The dataset contains customer information such as age, job,
    account balance, housing loan status, and previous campaign
    outcomes.

    The project uses Logistic Regression, Decision Tree, KNN,
    and Random Forest for classification.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Dataset Records", "4,521")
    col2.metric("Subscribed Customers", "521")
    col3.metric("Subscription Rate", "11.52%")

    st.info(
        "This application supports exploratory analysis and "
        "prediction. Predictions are estimates, not guarantees."
    )

elif page == "Customer Prediction":
    st.header("Predict Customer Subscription")
    st.write("Enter the customer's details below.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age", min_value=18, max_value=100, value=35
            )
            job = st.selectbox("Job", [
                "admin.", "blue-collar", "entrepreneur",
                "housemaid", "management", "retired",
                "self-employed", "services", "student",
                "technician", "unemployed", "unknown"
            ])
            marital = st.selectbox(
                "Marital Status",
                ["married", "single", "divorced"]
            )
            education = st.selectbox("Education", [
                "primary", "secondary", "tertiary", "unknown"
            ])
            balance = st.number_input(
                "Account Balance", value=1000
            )
            default = st.selectbox("Credit in Default", ["no", "yes"])
            housing = st.selectbox("Housing Loan", ["no", "yes"])
            loan = st.selectbox("Personal Loan", ["no", "yes"])

        with col2:
            contact = st.selectbox(
                "Contact Type", ["cellular", "telephone", "unknown"]
            )
            day = st.number_input(
                "Last Contact Day", min_value=1,
                max_value=31, value=15
            )
            month = st.selectbox("Last Contact Month", [
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"
            ])
            campaign = st.number_input(
                "Contacts in Current Campaign",
                min_value=1, value=1
            )
            pdays = st.number_input(
                "Days Since Previous Contact (-1 = Never Contacted)",
                min_value=-1, value=-1
            )
            previous = st.number_input(
                "Contacts Before This Campaign",
                min_value=0, value=0
            )
            poutcome = st.selectbox("Previous Campaign Outcome", [
                "unknown", "other", "failure", "success"
            ])

        submitted = st.form_submit_button("Predict Subscription")

    if submitted:
        # Keep these columns consistent with notebook training.
        customer = pd.DataFrame([{
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }])

        try:
            prediction = model.predict(customer)[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(customer)[0]
                probability_yes = float(probabilities[1])

            if prediction == 1:
                st.success("Prediction: Likely to subscribe (YES)")
            else:
                st.warning("Prediction: Unlikely to subscribe (NO)")

            if hasattr(model, "predict_proba"):
                st.metric(
                    "Estimated Subscription Probability",
                    f"{probability_yes:.1%}"
                )
                st.caption(
                    "This is the model's estimated probability, "
                    "not a guarantee of customer behavior."
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write(
                "Check that model.pkl contains the fitted pipeline "
                "and that its input columns match the notebook."
            )

elif page == "Model Comparison":
    st.header("Classification Model Comparison")
    st.write(
        "These results come from the model evaluation "
        "performed in the project notebook."
    )

    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Random Forest"
        ],
        "Accuracy": [0.705, 0.799, 0.876, 0.893],
        "Precision": [0.224, 0.221, 0.278, 0.684],
        "Recall": [0.635, 0.298, 0.048, 0.125],
        "F1-score": [0.331, 0.254, 0.082, 0.211],
        "ROC-AUC": [0.735, 0.581, 0.611, 0.728]
    })

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1-score": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Compare Evaluation Metrics")
    chart_data = results.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    ]
    st.bar_chart(chart_data)

    st.subheader("Interpretation")
    st.write("""
    - Random Forest achieved the highest accuracy (89.3%) and
      precision (68.4%).
    - Logistic Regression achieved the highest recall (63.5%)
      and F1-score (0.331) among the four tested models.
    - The dataset is imbalanced, so accuracy alone does not
      show how well the model identifies subscribers.
    """)
