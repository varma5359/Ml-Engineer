# Day 86: Capstone — Interactive Application

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved mini-project / deliverable:** Prediction interface

## 1. Learning objectives
- Design a user interface for the churn prediction service.
- Collect customer features using appropriate input widgets.
- Call the prediction API and display results.
- Handle API errors and invalid inputs.
- Communicate model limitations and human review requirements.

## 2. Why build a prediction interface?
A trained model and API are difficult for many business users to operate directly. An interface provides a guided way to submit inputs and view results.

Architecture:
Browser UI → API request → FastAPI validation → fitted pipeline → prediction response → UI result.

Keep the UI separate from model logic. The API remains the authoritative prediction service.

## 3. Suggested application structure
```text
customer_churn_capstone/
├── app/
│   ├── main.py
│   └── ui.py
├── api/
│   └── main.py
├── artifacts/
├── tests/
└── requirements.txt
```

## 4. Install UI dependencies
```bash
python -m pip install streamlit requests
```

Keep API and UI dependencies in the project's dependency file and pin tested versions before release.

## 5. Create a Streamlit form
Create `app/ui.py`. The fields below are illustrative; match the exact request schema from Day 85.

```python
import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Assistant", page_icon="📊")
st.title("Customer Churn Prediction")
st.caption("Decision-support estimate — not a guarantee of future behavior.")

with st.form("customer_form"):
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    monthly_charges = st.number_input(
        "Monthly charges", min_value=0.0, value=75.0
    )
    contract = st.selectbox(
        "Contract type",
        ["Month-to-month", "One year", "Two year"]
    )
    internet_service = st.selectbox(
        "Internet service",
        ["DSL", "Fiber optic", "No"]
    )
    submitted = st.form_submit_button("Estimate churn risk")
```

The form collects a coherent set of feature values and submits them together.

## 6. Call the prediction API
Continue in `app/ui.py`:
```python
API_URL = "http://127.0.0.1:8000/predict"

if submitted:
    payload = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "contract": contract,
        "internet_service": internet_service
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction result")
        st.write("Prediction:", result["prediction"])
        st.metric("Estimated churn probability",
                  f'{result["churn_probability"]:.1%}')
        st.caption(f'Model version: {result["model_version"]}')
    except requests.exceptions.Timeout:
        st.error("The prediction service took too long to respond.")
    except requests.exceptions.RequestException:
        st.error("The prediction service is unavailable. Try again later.")
```

Do not expose internal server details or credentials in the UI. In a deployed setup, API base URL and secrets should be configured securely rather than hardcoded.

## 7. Run both services
Terminal 1 — API:
```bash
uvicorn api.main:app --reload --port 8000
```

Terminal 2 — UI:
```bash
streamlit run app/ui.py
```

The exact import path depends on your project structure. Confirm the API is reachable before starting UI testing.

## 8. Improve the interface
- Use labels and units that match the feature contract.
- Display the model version.
- Show errors without stack traces.
- Explain probability as an estimate, not certainty.
- Add a clear human-review message.
- Avoid showing unnecessary personal information.
- Consider accessibility and responsive layout.

## 9. Test the integration
Test:
1. Valid input returns a prediction.
2. Invalid values are blocked by the UI or API.
3. API unavailable produces a clear message.
4. Timeout does not freeze the user workflow indefinitely.
5. Response model version is visible.
6. UI payload matches the API schema exactly.

## 10. Approved deliverable — Prediction interface

**Objective:** Create an interactive interface that calls the capstone inference API.

### Tasks
1. Build a Streamlit form using the approved feature schema.
2. Validate user input and units.
3. Call the FastAPI prediction endpoint.
4. Display predicted class, churn probability and model version.
5. Handle timeout, connection and validation errors.
6. Explain limitations and human decision responsibility.
7. Test the UI/API integration using valid and invalid examples.

**Suggested structure**
```text
customer_churn_capstone/
├── app/ui.py
├── api/main.py
├── tests/test_ui_api.py
└── README.md
```

**Validation:** Verify API payload names and types, test unavailable service behavior, and confirm no sensitive information is exposed in messages or logs.

## 11. AI-assisted implementation guide
| Level | Prompt |
|---|---|
| Hint | "Give me one hint for connecting Streamlit to my FastAPI endpoint." |
| Plan | "Break the prediction interface into form, API call and result sections." |
| Example | "Show a small requests.post example with a timeout." |
| Guided implementation | "Help me build the Streamlit form first; wait before adding API logic." |
| Explanation | "Explain raise_for_status and request timeouts." |
| Debugging | "Help me debug this 422 response from my prediction API." |
| Review | "Review my interface for schema compatibility and safe error messages." |

## 12. Daily quiz

**Q1. What should the UI call to obtain a prediction?**  
A. The training notebook  
B. The inference API  
C. The raw dataset folder  
D. The Git history  
**Correct: B.** The UI should use the serving interface.

**Q2. Why set a request timeout?**  
A. To change the prediction label  
B. To prevent indefinite waiting for a response  
C. To retrain the model  
D. To remove API validation  
**Correct: B.** Timeouts bound how long the client waits.

**Q3. What should the UI show with a prediction?**  
A. Only an unexplained number  
B. Prediction context, probability where supported and model version  
C. Secret keys  
D. Training customer records  
**Correct: B.** Users need context and traceability.

**Q4. What is the purpose of raise_for_status()?**  
A. Raise an exception for unsuccessful HTTP responses  
B. Train the model  
C. Save the dataset  
D. Start Docker  
**Correct: A.** It helps route HTTP errors through error handling.

**Q5. Which design is preferable?**  
A. UI duplicates model preprocessing independently  
B. UI calls the API and the API owns inference logic  
C. UI trains a model on each click  
D. UI stores credentials in source code  
**Correct: B.** A single serving contract reduces logic duplication.

## 13. Interview preparation

### Q1. Why separate the UI and prediction API?
**Answer:** It separates presentation from model serving. The API can be tested and reused by other clients, while the UI focuses on collecting inputs and displaying results. It also centralizes validation and model-version handling.

**Follow-up:** How could another application reuse the same API?

### Q2. How should an interface handle an API outage?
**Answer:** Set bounded timeouts, catch connection and HTTP errors, show a safe message and avoid displaying stale or fabricated predictions. Log diagnostic information securely and provide a retry path where appropriate.

**Follow-up:** Would you cache predictions during an outage?

## 14. Completion checklist
- [ ] Build the Streamlit input form.
- [ ] Call the FastAPI endpoint.
- [ ] Display result and model version.
- [ ] Handle timeout and API errors.
- [ ] Explain limitations and human review.
- [ ] Complete Prediction interface.

**Next lesson:** Day 87 — Capstone: Docker and deployment preparation.
