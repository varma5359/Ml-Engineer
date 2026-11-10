# Day 66: Streamlit or Equivalent UI for Model Demonstrations

**Pathway:** Machine Learning Engineer  
**Phase:** 7 — Deployment and MLOps  
**Level:** Beginner to Intermediate  
**Estimated time:** 2–3 hours  
**Approved mini-project:** Interactive ML Application

## 1. Learning objectives
- Explain why an ML model needs a user interface.
- Build a Streamlit app with input widgets.
- Load a trusted, serialized model pipeline.
- Display predictions and probability estimates.
- Validate inputs and communicate model limitations.

## 2. What is an ML application UI?
A user interface allows a person to interact with a model without writing Python code.

**Flow:** User input → UI validation → model pipeline → prediction → result and explanation.

Streamlit is a Python framework for building data and ML applications. It is useful for prototypes, internal tools and demonstrations. It does not automatically provide production authentication, governance or high availability.

## 3. Install Streamlit
```bash
python -m pip install streamlit pandas scikit-learn joblib
```

Check the installation:
```bash
streamlit --version
```

## 4. Prepare a model artifact
Use a fitted scikit-learn pipeline and save it as `artifacts/model.joblib`. The feature names, order and types in the UI must match the model's training schema. Only load trusted serialized artifacts.

## 5. Create a simple Streamlit interface
Create `app.py`:

```python
import streamlit as st

st.set_page_config(page_title="Customer Risk Demo", page_icon="📊")
st.title("Customer Risk Prediction")
st.write("Enter customer information to request a model prediction.")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
monthly_charge = st.number_input(
    "Monthly charge", min_value=0.0, value=75.0, step=5.0
)
support_calls = st.number_input(
    "Support calls", min_value=0, max_value=50, value=2
)

st.write("Review your inputs before requesting a prediction.")
st.write({"age": age, "monthly_charge": monthly_charge,
          "support_calls": support_calls})
```

Run:
```bash
streamlit run app.py
```

Streamlit opens a local browser tab, normally at `http://localhost:8501`.

## 6. Load a model and predict
Add this to the app after the imports:

```python
import joblib
import pandas as pd
from pathlib import Path

@st.cache_resource
def load_model():
    return joblib.load(Path("artifacts/model.joblib"))

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model artifact not found. Train and save the model first.")
    st.stop()
```

Add a prediction button after the input widgets:

```python
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "age": age,
        "monthly_charge": monthly_charge,
        "support_calls": support_calls
    }])

    prediction = model.predict(input_data)[0]
    st.subheader("Prediction")
    st.write(int(prediction))

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0].max()
        st.metric("Highest class probability", f"{probability:.1%}")
```

**Important:** The example assumes the model expects exactly these three columns. Replace them with the real feature schema. The maximum class probability is not necessarily a calibrated confidence score.

## 7. Improve the user experience
- Use labels that describe units and allowed values.
- Display validation errors near the relevant input.
- Show the model version and prediction timestamp.
- Explain what the output means.
- Provide a warning that predictions are estimates, not guarantees.
- Avoid displaying personal or sensitive training records.

## 8. Approved mini-project — Interactive ML Application

**Scenario:** A business team wants to test a trained prediction model through a simple browser interface.

**Requirements**
1. Reuse a trained and serialized pipeline.
2. Build an intuitive Streamlit form.
3. Add suitable widgets for all model features.
4. Validate ranges and required fields.
5. Generate a prediction on button click.
6. Display output clearly and show probability only if supported.
7. Display model version and limitations.
8. Handle a missing or incompatible artifact gracefully.

**Suggested structure**
```text
interactive_ml_application/
├── app.py
├── artifacts/model.joblib
├── src/train.py
├── requirements.txt
└── README.md
```

**Validation:** Test valid inputs, boundary values, invalid inputs, missing artifact and feature-schema mismatch. Confirm the UI does not silently reorder or rename features incorrectly.

## 9. Common mistakes
| Mistake | Better approach |
|---|---|
| UI features do not match training schema | Reuse documented feature contract |
| Model reloads on every rerun | Use `st.cache_resource` appropriately |
| Display probability as certainty | Explain probability and calibration limits |
| Expose raw customer data | Minimize and protect displayed data |
| Assume local app is production-ready | Add access, deployment and monitoring controls |

## 10. Completion checklist
- [ ] Explain UI-to-model flow.
- [ ] Build a Streamlit input form.
- [ ] Load a fitted pipeline.
- [ ] Display prediction safely.
- [ ] Complete Interactive ML Application.

**Next lesson:** Day 67 — Docker and containerizing ML applications.
