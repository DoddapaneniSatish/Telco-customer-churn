import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# MUST be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📞",
    layout="centered",
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "data_schema.json"

# Keep the existing hosted API as the fallback; local/Compose runs set API_URL.
API_BASE_URL = os.environ.get("API_URL", "https://telco-customer-churn-ml.onrender.com").rstrip("/")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# -----------------------------------------------------------------------------
# Load schema from JSON file
# -----------------------------------------------------------------------------
@st.cache_resource
def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


schema = load_schema(SCHEMA_PATH)

numerical_features = schema.get("numerical", {})
categorical_features = schema.get("categorical", {})

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("📞 Telco Customer Churn Prediction App")
st.write(
    "Explore how a customer profile changes a model’s churn estimate. "
    "This educational demo supports investigation; it does not establish why a customer leaves."
)

st.header("Input Customer Features")

user_input: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Numerical Features
# -----------------------------------------------------------------------------
st.subheader("Numerical Features")

for feature_name, stats in numerical_features.items():
    min_val = float(stats.get("min", 0.0))
    max_val = float(stats.get("max", 1000.0))
    mean_val = float(stats.get("mean", (min_val + max_val) / 2))
    median_val = float(stats.get("median", mean_val))

    default_val = median_val

    label = feature_name.replace("_", " ").title()
    help_text = (
        f"Min: {min_val:.2f}, Max: {max_val:.2f}, "
        f"Mean: {mean_val:.2f}, Median: {median_val:.2f}"
    )

    range_val = max_val - min_val
    if range_val > 1000:
        step = 10.0
    elif range_val > 100:
        step = 1.0
    elif range_val > 10:
        step = 0.1
    else:
        step = 0.01

    user_input[feature_name] = st.number_input(
        label,
        min_value=min_val,
        max_value=max_val,
        value=float(default_val),
        step=step,
        help=help_text,
        key=feature_name,
    )

# -----------------------------------------------------------------------------
# Categorical Features
# -----------------------------------------------------------------------------
st.subheader("Categorical Features")

for feature_name, info in categorical_features.items():
    unique_values = info.get("unique_values", [])
    value_counts = info.get("value_counts", {})

    if not unique_values:
        continue

    default_value = (
        max(value_counts, key=value_counts.get)
        if value_counts
        else unique_values[0]
    )

    try:
        default_idx = unique_values.index(default_value)
    except ValueError:
        default_idx = 0

    label = feature_name.replace("_", " ").title()

    user_input[feature_name] = st.selectbox(
        label,
        options=unique_values,
        index=default_idx,
        key=feature_name,
        help=f"Distribution: {value_counts}",
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# Predict Button
# -----------------------------------------------------------------------------
if st.button("🔮 Predict Churn", type="primary"):
    payload = {"instances": [user_input]}

    with st.spinner("Estimating churn..."):
        try:
            resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request to API failed: {e}")
        else:
            if resp.status_code != 200:
                st.error(f"❌ API error: HTTP {resp.status_code} - {resp.text}")
            else:
                data = resp.json()

                churn_preds = data.get("churn_prediction", [])
                churn_probs = data.get("churn_probability", [])

                if not churn_preds:
                    st.warning("⚠️ No prediction returned from API.")
                else:
                    churn = churn_preds[0]
                    prob = churn_probs[0]

                    st.success("✅ Prediction successful!")

                    st.subheader("Prediction Result")

                    if churn == 1:
                        st.metric(
                            label="Predicted churn class",
                            value="YES ❌",
                            delta=None,
                        )
                    else:
                        st.metric(
                            label="Predicted churn class",
                            value="NO ✅",
                            delta=None,
                        )

                    st.metric("Model churn score", f"{prob:.2%}")
                    st.caption("Model scores are not calibrated guarantees or causal explanations.")

                    with st.expander("📋 View Input Summary"):
                        st.json(user_input)

st.markdown("---")
with st.expander("Connection details"):
    st.code(API_BASE_URL)
