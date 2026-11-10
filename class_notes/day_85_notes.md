# Day 85: Capstone — Prediction API

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved mini-project / deliverable:** Inference service

## 1. Learning objectives
- Explain the role of an inference API.
- Load a trusted, versioned preprocessing/model pipeline.
- Define request and response schemas with Pydantic.
- Validate customer input and return predictions.
- Test success, invalid input and model-readiness behavior.

## 2. What is an inference API?
An inference API exposes a trained model through a service endpoint. A client sends feature values and receives a prediction.

Flow:
Client → HTTP request → schema validation → model pipeline → prediction → response.

The API must use the same feature contract and preprocessing behavior as training. Do not retrain the model inside the request handler.

## 3. Project structure
```text
customer_churn_capstone/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── artifacts/
│   └── churn_pipeline.joblib
├── tests/
├── requirements.txt
└── README.md
```

## 4. Install dependencies
```bash
python -m pip install fastapi uvicorn pydantic pandas joblib scikit-learn
```

Record tested versions in requirements.txt. Ensure the artifact is created by the approved training workflow and loaded only from a trusted location.

## 5. Define the input and output schema
The schema below is illustrative. Replace fields with the exact model features identified in the capstone. Use consistent types, units and allowed values.

```python
from pydantic import BaseModel, Field

class ChurnRequest(BaseModel):
    tenure: float = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    contract: str
    internet_service: str

class ChurnResponse(BaseModel):
    prediction: int
    churn_probability: float
    model_version: str
```

## 6. Load the fitted pipeline once
Create `app/main.py`:
```python
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schemas import ChurnRequest, ChurnResponse

MODEL_PATH = Path("artifacts/churn_pipeline.joblib")
MODEL_VERSION = "churn-model-v1"

app = FastAPI(title="Customer Churn Prediction API")
model = None

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError("Required model artifact is unavailable")
    model = joblib.load(MODEL_PATH)
```

Startup event APIs may differ across FastAPI versions; lifespan handlers are the current recommended pattern in newer applications. The important design is to load a trusted artifact once and fail readiness if it is missing.

## 7. Create the prediction endpoint
Continue in `app/main.py`:
```python
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    input_frame = pd.DataFrame([request.model_dump()])
    try:
        prediction = int(model.predict(input_frame)[0])
        classes = list(model.classes_)
        positive_index = classes.index(1)
        probability = float(model.predict_proba(input_frame)[0][positive_index])
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed"
        )

    return ChurnResponse(
        prediction=prediction,
        churn_probability=probability,
        model_version=MODEL_VERSION
    )
```

This assumes the positive class is encoded as integer 1 and the fitted pipeline exposes `classes_` and `predict_proba`. Adapt the schema and class mapping to the actual artifact. Avoid returning internal exception details.

## 8. Run the API
```bash
uvicorn app.main:app --reload
```

Open:
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

FastAPI provides an interactive OpenAPI documentation page.

## 9. Example request
```json
{
  "tenure": 12,
  "monthly_charges": 75.5,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic"
}
```

The exact payload must match the actual trained feature schema. A response may contain a class, probability and model version. Do not invent a prediction result before running the real artifact.

## 10. Test the endpoint
Use FastAPI TestClient:
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_invalid_request():
    response = client.post("/predict", json={"tenure": -1})
    assert response.status_code == 422
```

Add a valid prediction test using an available test artifact and a request matching the schema. Do not make tests depend on a production model download.

## 11. Approved deliverable — Inference service

**Objective:** Expose the approved churn pipeline through a tested FastAPI service.

### Tasks
1. Define the feature request schema.
2. Load the trusted fitted pipeline at startup.
3. Add health and prediction endpoints.
4. Return prediction, positive-class probability and model version.
5. Validate invalid and missing fields.
6. Add automated API tests.
7. Handle missing model and prediction failures safely.
8. Document how the artifact and feature schema are versioned.

**Validation:** Test valid request, invalid type/range, missing fields, unavailable artifact and positive-class mapping. Confirm API feature names exactly match the training pipeline.

## 12. AI-assisted implementation guide
| Level | Prompt |
|---|---|
| Hint | "Give me one hint for designing a request schema from my feature list." |
| Plan | "Break the inference service into schema, model loading, route and tests." |
| Example | "Show a small FastAPI endpoint that returns a model version." |
| Guided implementation | "Help me implement the request schema first and wait for my test." |
| Explanation | "Explain why the API should load the model once at startup." |
| Debugging | "Help me debug this feature-name mismatch between API and pipeline." |
| Review | "Review my API for validation, error handling and safe artifact loading." |

## 13. Daily quiz

**Q1. What is the role of an inference API?**  
A. Train the model for every request  
B. Expose model predictions to clients  
C. Replace the dataset  
D. Create Git commits  
**Correct: B.** It provides a service interface for predictions.

**Q2. Why load the model once at startup?**  
A. To avoid repeated artifact loading per request  
B. To change the target  
C. To remove validation  
D. To fit on client input  
**Correct: A.** It reduces repeated loading overhead.

**Q3. What does Pydantic validate?**  
A. Docker image layers  
B. Request data structure and constraints  
C. Model accuracy  
D. Dataset license  
**Correct: B.** Schemas validate incoming request fields and values.

**Q4. Why return model_version?**  
A. For traceability of the prediction source  
B. To reveal private training records  
C. To improve precision automatically  
D. To skip testing  
**Correct: A.** It helps connect predictions to the deployed artifact.

**Q5. What should happen if the model artifact is missing?**  
A. Return a random prediction  
B. Fail readiness or return a safe unavailable response  
C. Train a new model on request data  
D. Ignore the problem  
**Correct: B.** The service must not pretend it can serve valid predictions.

## 14. Interview preparation

### Q1. How do you deploy a scikit-learn model through FastAPI?
**Answer:** Save the complete fitted pipeline, define Pydantic request/response schemas, load the trusted artifact at startup, validate inputs, convert them to the expected DataFrame structure, call predict/predict_proba and return a versioned response. Add tests and operational controls.

**Follow-up:** How do you prevent a mismatch between training and API features?

### Q2. Why should prediction errors not expose full exception details?
**Answer:** Internal tracebacks can reveal filesystem paths, package details or sensitive implementation information. Log diagnostic details securely on the server and return a safe message and appropriate status code to the client.

**Follow-up:** How would you correlate a client error with a server log safely?

## 15. Completion checklist
- [ ] Define request and response schemas.
- [ ] Load the fitted pipeline.
- [ ] Implement health and prediction endpoints.
- [ ] Validate positive-class probability.
- [ ] Test valid and invalid requests.
- [ ] Complete Inference service.

**Next lesson:** Day 86 — Capstone: Interactive application.
