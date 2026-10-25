"""Streamlit UI: collects the selected features and calls the FastAPI service.

The model is never loaded or fitted here - the API is the only prediction service.
"""

import os

import requests
import streamlit as st

from production_app.schemas import FEATURES

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 10

# Median measurements of benign / malignant tumours in the training data.
SAMPLES = {
    "benign": {
        "concave points_mean": 0.02344,
        "radius_se": 0.2575,
        "radius_worst": 13.35,
        "texture_worst": 22.82,
        "smoothness_worst": 0.1254,
        "concavity_worst": 0.1412,
        "concave points_worst": 0.07431,
        "symmetry_worst": 0.2687,
    },
    "malignant": {
        "concave points_mean": 0.08628,
        "radius_se": 0.5472,
        "radius_worst": 20.59,
        "texture_worst": 28.945,
        "smoothness_worst": 0.14345,
        "concavity_worst": 0.4049,
        "concave points_worst": 0.182,
        "symmetry_worst": 0.3103,
    },
}


def load_sample(name: str) -> None:
    for feature, value in SAMPLES[name].items():
        st.session_state[feature] = value


def describe_error(response: requests.Response) -> str:
    if response.status_code == 422:
        try:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in err['loc'][1:])}: {err['msg']}"
                for err in response.json()["detail"]
            )
        except ValueError, KeyError, TypeError:
            problems = response.text
        return f"The API rejected the input - {problems}"
    if response.status_code == 503:
        return "The API is running but the model artifact is unavailable."
    return f"Unexpected response from the API ({response.status_code})."


st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺")
st.title("Breast Cancer Classifier")
st.caption(
    "Enter the 8 selected cell-nucleus measurements, then predict. "
    f"Requests go to the FastAPI service at {API_URL}."
)

if "loaded" not in st.session_state:
    for name in FEATURES:
        st.session_state.setdefault(name, SAMPLES["benign"][name])
    st.session_state["loaded"] = True

left, right = st.columns(2)
left.button("Load benign example", on_click=load_sample, args=("benign",))
right.button("Load malignant example", on_click=load_sample, args=("malignant",))

with st.form("predict"):
    columns = st.columns(2)
    for index, feature in enumerate(FEATURES):
        columns[index % 2].number_input(
            feature,
            key=feature,
            min_value=0.0,
            step=0.001,
            format="%.5f",
        )
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    payload = {feature: st.session_state[feature] for feature in FEATURES}
    try:
        response = requests.post(
            f"{API_URL}/predict", json=payload, timeout=TIMEOUT_SECONDS
        )
    except requests.Timeout:
        st.error(f"The API did not answer within {TIMEOUT_SECONDS} seconds.")
    except requests.RequestException:
        st.error(f"Cannot reach the API at {API_URL}. Is the container running?")
    else:
        if response.status_code == 200:
            result = response.json()
            st.success(
                f"Prediction: **{result['label']}** (class {result['prediction']})"
            )
            if result["probability"] is not None:
                st.metric("Malignant probability", f"{result['probability']:.1%}")
        else:
            st.error(describe_error(response))
