import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="wide",
)

# Keep the model file in the same folder as this app.py file.
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "breast_cancer_model.pkl",
)


# --------------------------------------------------
# 2. LOAD THE TRAINED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)


st.title("🎗️ Breast Cancer Classification")
st.subheader("PySpark + Machine Learning + Streamlit")

st.markdown(
    """
    This application demonstrates how a Logistic Regression model trained
    with PySpark can be used to classify the UCI Breast Cancer Wisconsin
    (Diagnostic) dataset.

    **Target classes:** Benign and Malignant.
    """
)

st.warning(
    "Educational demonstration only. This model is not clinically validated "
    "and must not be used to make medical decisions."
)

try:
    model = load_model(MODEL_PATH)
except Exception:
    st.error(
        "Could not load the model. Ensure that `breast_cancer_model.pkl` "
        "exists in the same folder as `app.py` and is a trusted model file "
        "exported by your training notebook."
    )
    st.stop()


# --------------------------------------------------
# 3. EXTRACT MODEL PARAMETERS
# --------------------------------------------------
try:
    feature_names = model["feature_names"]
    scaler_mean = np.asarray(model["scaler_mean"], dtype=float)
    scaler_std = np.asarray(model["scaler_std"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    intercept = float(model["intercept"])
    threshold = float(model["threshold"])
except (KeyError, TypeError, ValueError) as exc:
    st.error(
        "The model file does not contain the expected exported parameters. "
        "Expected keys: feature_names, scaler_mean, scaler_std, "
        "coefficients, intercept, threshold."
    )
    st.stop()

# Avoid division by zero when scaling.
scaler_std = np.where(scaler_std == 0, 1.0, scaler_std)


# --------------------------------------------------
# 4. PREDICTION FUNCTION
# --------------------------------------------------
def predict_breast_cancer(input_df):
    """Predict from a DataFrame containing the 30 expected features."""
    # Preserve the feature order used when the model was trained.
    X = input_df[feature_names].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError("Input contains missing or non-finite values.")

    # Apply the scaling parameters learned during training.
    X_scaled = (X - scaler_mean) / scaler_std

    # Logistic Regression linear score.
    scores = X_scaled @ coefficients + intercept

    # Convert scores to probabilities, clipping to avoid overflow.
    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(scores, -500, 500))
    )

    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities


# --------------------------------------------------
# 5. INPUT INTERFACE
# --------------------------------------------------
st.divider()
st.header("Make a prediction")

manual_tab, csv_tab = st.tabs(
    ["Enter features manually", "Upload a CSV file"]
)


# --------------------------------------------------
# 6. MANUAL INPUT
# --------------------------------------------------
with manual_tab:
    st.write(
        "Enter the 30 numeric features produced by the UCI dataset's "
        "image-analysis process. Use values from a valid dataset record "
        "for a meaningful demonstration."
    )

    with st.form("prediction_form"):
        st.caption(
            "Features are grouped by statistic: mean, standard error, and "
            "worst. All 30 fields are required."
        )
        input_values = {}

        for group_name, start_index in [
            ("Mean features", 0),
            ("Standard error features", 10),
            ("Worst features", 20),
        ]:
            st.markdown(f"#### {group_name}")
            cols = st.columns(3)

            for i in range(start_index, min(start_index + 10, len(feature_names))):
                feature = feature_names[i]
                with cols[(i - start_index) % 3]:
                    input_values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        value=0.0,
                        format="%.6f",
                        key=f"manual_{feature}",
                    )

        submitted = st.form_submit_button(
            "Predict",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame([input_values], columns=feature_names)

        try:
            predictions, probabilities = predict_breast_cancer(input_df)
            predicted_class = int(predictions[0])
            malignant_probability = float(probabilities[0])

            st.divider()
            st.subheader("Prediction result")
            col1, col2 = st.columns(2)

            with col1:
                if predicted_class == 1:
                    st.error("Model output: Malignant")
                else:
                    st.success("Model output: Benign")

            with col2:
                st.metric(
                    "Estimated malignant probability",
                    f"{malignant_probability:.2%}",
                )

            st.progress(malignant_probability)
            st.caption(
                "This is the model's estimated output, not a clinically "
                "validated estimate of an individual's actual cancer risk."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# --------------------------------------------------
# 7. CSV BATCH PREDICTIONS
# --------------------------------------------------
with csv_tab:
    st.write(
        "Upload a CSV containing the 30 feature columns. Column names must "
        "match the feature names used during training."
    )

    template = pd.DataFrame(columns=feature_names)
    st.download_button(
        label="Download CSV template",
        data=template.to_csv(index=False),
        file_name="breast_cancer_input_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload feature data", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            missing_columns = [
                col for col in feature_names if col not in input_df.columns
            ]

            if missing_columns:
                st.error(
                    "Missing required feature columns: "
                    + ", ".join(missing_columns)
                )
            elif input_df.empty:
                st.error("The uploaded CSV contains no records.")
            else:
                input_df = input_df[feature_names].apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                if input_df.isna().any().any():
                    st.error(
                        "The CSV contains missing or non-numeric feature "
                        "values. Please correct the file."
                    )
                elif not np.isfinite(input_df.to_numpy(dtype=float)).all():
                    st.error("The CSV contains infinite or invalid values.")
                else:
                    st.write("Input preview:")
                    st.dataframe(input_df.head(), use_container_width=True)

                    if st.button("Run batch predictions", type="primary"):
                        predictions, probabilities = predict_breast_cancer(
                            input_df
                        )
                        results_df = input_df.copy()
                        results_df["predicted_label"] = predictions
                        results_df["predicted_class"] = np.where(
                            predictions == 1, "Malignant", "Benign"
                        )
                        results_df["estimated_malignant_probability"] = (
                            probabilities
                        )

                        st.subheader("Batch prediction results")
                        st.dataframe(results_df, use_container_width=True)
                        st.download_button(
                            label="Download predictions as CSV",
                            data=results_df.to_csv(index=False),
                            file_name="breast_cancer_predictions.csv",
                            mime="text/csv",
                        )
                        st.caption(
                            "Outputs are experimental model predictions and "
                            "are not medical diagnoses."
                        )
        except Exception as exc:
            st.error(f"Could not process the uploaded CSV: {exc}")


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Creator: Ibikunle Gabriel. \n"
    "Dataset: UCI Breast Cancer Wisconsin (Diagnostic). \n"
    "Model: PySpark Logistic Regression. \n"
    "For educational and demonstration purposes only."
)
