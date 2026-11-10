# Day 64: Model Deployment Concepts and Inference APIs

**Pathway:** Machine Learning Engineer  
**Phase:** 6 — Advanced ML and Practical Applications  
**Level:** Intermediate  
**Estimated learning time:** 2–3 hours  
**Approved mini-project:** Prediction API

## 1. Learning objectives
- Explain model deployment and inference.
- Distinguish batch and online prediction.
- Describe a basic inference API architecture.
- Build a small prediction endpoint.
- Validate requests and handle errors.
- Identify deployment, monitoring and security requirements.

## 2. What is model deployment?
Deployment makes a trained model available to an application or user for predictions.

Training workflow:
Data → preprocessing → training → evaluation → artifact.

Inference workflow:
Request data → validation → preprocessing/model → prediction → response.

A deployed model needs more than a saved file: it needs input contracts, runtime dependencies, access controls, monitoring and an update strategy.

## 3. Batch versus online inference

| Type | Description | Example |
|---|---|---|
| Batch | Predict many records together on a schedule | Daily customer risk scores |
| Online | Predict when a request arrives | Estimate churn risk for one customer |
| Streaming | Process continuous event flow | Real-time anomaly scoring |

Choose based on latency, volume, freshness and infrastructure needs.

## 4. Basic API architecture
Client → HTTP request → API validation → model artifact → prediction → JSON response.

The API should:
- Validate required fields and data types.
- Apply the exact training preprocessing.
- Return a defined response schema.
- Handle invalid input and model errors.
- Avoid exposing sensitive internal details.
- Record operational metrics safely.

## 5. Prepare a reusable model artifact
Use the pipeline from Day 63 or create a small trained pipeline and save it as `artifacts/model.joblib`. Keep the feature names and order documented.

## 6. Create a basic FastAPI service

### 6.1 Install packages
```bash
python -m pip install fastapi uvicorn scikit-learn pandas joblib
```

### 6.2 Create app.py
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="ML Prediction API", version="1.0.0")
model = joblib.load("artifacts/model.joblib")

class PredictionInput(BaseModel):
    age: float = Field(gt=0, le=120)
    monthly_charge: float = Field(ge=0)
    support_calls: int = Field(ge=0)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictionInput):
    try:
        frame = pd.DataFrame([payload.model_dump()])
        prediction = model.predict(frame)[0]
        return {"prediction": int(prediction)}
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )
```

**Important:** The model artifact used here must have been trained with exactly these feature names and compatible input types. Replace the example schema to match your own model. Pydantic's `model_dump()` is used in Pydantic v2; use the appropriate method for your installed version.

### 6.3 Start the server
```bash
uvicorn app:app --reload
```

Expected terminal output indicates Uvicorn is serving the application locally, typically at `http://127.0.0.1:8000`.

### 6.4 Open API documentation
Visit `http://127.0.0.1:8000/docs` in a browser to inspect and test the interactive API documentation.

## 7. Test the endpoint
Example JSON body (field values are illustrative):
```json
{
  "age": 35,
  "monthly_charge": 75.5,
  "support_calls": 2
}
```

A successful response follows the declared schema, for example:
```json
{
  "prediction": 0
}
```

The numeric prediction depends on the loaded model and training data.

## 8. HTTP status codes
| Code | Meaning | Example |
|---:|---|---|
| 200 | Success | Prediction returned |
| 400 | Bad request | Application-level invalid request |
| 422 | Validation error | Missing or invalid Pydantic field |
| 500 | Server error | Unexpected inference failure |
| 503 | Service unavailable | Model or dependency unavailable |

Do not return raw stack traces or secrets to clients.

## 9. Deployment considerations
- Store model artifact in a controlled location.
- Pin dependencies and record versions.
- Load model once at startup where appropriate.
- Validate schema and ranges.
- Add authentication and authorization for non-public services.
- Use HTTPS in production.
- Set request size and rate limits.
- Monitor latency, errors and prediction distribution.
- Plan safe model updates and rollback.

## 10. Approved mini-project — Prediction API

**Scenario:** A business application needs to request predictions from a trained ML model.

**Objective:** Build a local REST API that accepts validated input and returns model predictions.

### Requirements
1. Train or reuse a serialized model pipeline.
2. Define an input schema that matches training features.
3. Implement a health endpoint.
4. Implement a prediction endpoint.
5. Validate types, required fields and reasonable ranges.
6. Return a stable JSON response.
7. Handle invalid requests and inference failures.
8. Test normal, missing, out-of-range and malformed inputs.
9. Document startup and API usage.

### Suggested structure
```text
prediction_api/
├── artifacts/
│   └── model.joblib
├── app/
│   ├── main.py
│   └── schemas.py
├── tests/
├── requirements.txt
└── README.md
```

### Validation checklist
- API starts successfully.
- Health endpoint responds.
- Valid request returns expected response shape.
- Invalid request returns a validation error.
- Model feature schema matches API schema.
- Sensitive input values are not written to logs unnecessarily.

## 11. Common mistakes
- Loading a model trained on different columns.
- Accepting unchecked user input.
- Returning internal exception details.
- Loading model on every request unnecessarily.
- Exposing an unauthenticated public endpoint.
- Forgetting dependency and artifact versioning.

## 12. Lesson completion
- [ ] Explain batch, online and streaming inference.
- [ ] Describe API request-to-prediction flow.
- [ ] Build a FastAPI health and predict endpoint.
- [ ] Validate input and test errors.
- [ ] Complete Prediction API.

**Next lesson:** Day 65 — FastAPI model serving and input validation.
