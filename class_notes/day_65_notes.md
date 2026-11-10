# Day 65: FastAPI Model Serving and Input Validation

**Pathway:** Machine Learning Engineer  
**Phase:** 6 — Advanced ML and Practical Applications  
**Level:** Intermediate  
**Estimated learning time:** 2–3 hours  
**Approved mini-project:** ML Prediction Service

## 1. Learning objectives
- Build a FastAPI service around a trained model.
- Define request and response schemas with Pydantic.
- Validate input types, ranges and required fields.
- Separate API schemas, model loading and prediction logic.
- Test successful and invalid requests.
- Explain safe error handling and service readiness.

## 2. What is model serving?
Model serving is the process of making a trained model available for inference through a service.

Typical request flow:
Client → API route → request validation → feature preparation → model prediction → response validation → client.

FastAPI uses Python type annotations and Pydantic models to validate incoming request data.

## 3. Project structure
```text
ml_prediction_service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── model_service.py
├── artifacts/
│   └── model.joblib
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
```

Keep model artifacts trusted and versioned. The schema in this lesson is illustrative; align it with the actual model features.

## 4. Install dependencies
```bash
python -m pip install fastapi uvicorn pydantic pandas joblib scikit-learn
```

Record tested versions in the project's dependency file.

## 5. Define request and response schemas
Create `app/schemas.py`:

```python
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    age: float = Field(gt=0, le=120)
    monthly_charge: float = Field(ge=0)
    support_calls: int = Field(ge=0)

class PredictionResponse(BaseModel):
    prediction: int
    model_version: str
```

**Explanation:**
- BaseModel defines the request/response structure.
- Field constraints validate values.
- Invalid requests are rejected before prediction.
- Add or remove fields to match the trained model's exact feature schema.

## 6. Load and call the model
Create `app/model_service.py`:

```python
import joblib
import pandas as pd

MODEL_PATH = "artifacts/model.joblib"
MODEL_VERSION = "churn-model-v1"

model = joblib.load(MODEL_PATH)

def predict_one(features: dict) -> int:
    frame = pd.DataFrame([features])
    prediction = model.predict(frame)[0]
    return int(prediction)
```

**Explanation:** The model is loaded when the module is imported. In a more advanced service, use FastAPI lifespan management to load resources at startup and expose readiness state. Ensure the artifact is trusted.

## 7. Create API routes
Create `app/main.py`:

```python
from fastapi import FastAPI, HTTPException
from app.schemas import PredictionRequest, PredictionResponse
from app.model_service import predict_one, MODEL_VERSION

app = FastAPI(title="ML Prediction Service", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = request.model_dump()
        prediction = predict_one(features)
        return PredictionResponse(
            prediction=prediction,
            model_version=MODEL_VERSION
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate prediction"
        )
```

**Pydantic v2 note:** `model_dump()` is the current method. In Pydantic v1, `dict()` is commonly used.

## 8. Run the service
From the project root:
```bash
uvicorn app.main:app --reload
```

Open:
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

The service can only predict if the artifact exists and its expected features match the request.

## 9. Test request validation
Valid request:
```json
{
  "age": 35,
  "monthly_charge": 72.5,
  "support_calls": 2
}
```

Invalid request:
```json
{
  "age": -5,
  "monthly_charge": "high",
  "support_calls": -1
}
```

The invalid request should fail validation and normally return HTTP 422. The endpoint should not call the model with invalid data.

## 10. Automated API test
Create `tests/test_api.py`:

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_invalid_prediction():
    response = client.post("/predict", json={
        "age": -1,
        "monthly_charge": 50,
        "support_calls": 1
    })
    assert response.status_code == 422
```

Install test dependencies:
```bash
python -m pip install pytest httpx
pytest -q
```

**Note:** The valid-prediction test requires a compatible trusted model artifact. Tests should use a controlled test artifact rather than a real production model.

## 11. Input validation beyond basic types
Consider:
- Required feature presence.
- Numeric ranges and units.
- Category allowlists.
- Null and missing-value policy.
- Unexpected fields.
- Batch size limits.
- Feature schema version.
- Whether the feature can be known at prediction time.

Validation should reflect the real business domain. Do not invent arbitrary constraints that reject valid records.

## 12. Error handling and logging
Good practice:
- Return safe messages to clients.
- Log diagnostic context without logging sensitive raw payloads.
- Distinguish validation errors from internal failures.
- Add request IDs for tracing.
- Monitor latency, request count and failure rate.
- Avoid returning stack traces or filesystem paths.

## 13. Approved mini-project — ML Prediction Service

**Scenario:** A customer application needs a reusable service to request churn predictions.

**Objective:** Build a FastAPI service with explicit schemas, model loading, prediction and tests.

### Requirements
1. Create a trained model pipeline and save a trusted artifact.
2. Create a Pydantic request schema matching model features.
3. Create a response schema with prediction and model version.
4. Implement health and prediction routes.
5. Load the model once per application process or with lifespan management.
6. Validate missing, invalid and out-of-range inputs.
7. Add tests for health, valid request and invalid request.
8. Document setup, artifact requirements and API examples.
9. Explain production requirements such as authentication, HTTPS, monitoring and rollback.

### Suggested submission evidence
- Project folder structure.
- API documentation screenshot or request/response examples.
- Test output.
- Model metadata and feature schema.
- README with setup and limitations.

## 14. Common mistakes
| Mistake | Better approach |
|---|---|
| Request schema differs from model features | Keep a single documented feature contract |
| Load model on every request | Load at startup/lifespan |
| Return raw exception details | Return safe errors and log securely |
| Skip input range checks | Use Pydantic constraints and domain validation |
| No tests for invalid payloads | Test validation failures |
| Public API has no access control | Add authentication and rate limiting where needed |
| Assume health means model is ready | Add readiness checks for model and dependencies |

## 15. Phase 6 completion checklist
- [ ] Explain gradient boosting and compare boosting libraries.
- [ ] Build and evaluate a churn boosting model.
- [ ] Explain forecasting and create time-series features.
- [ ] Build a recommendation prototype.
- [ ] Integrate preprocessing with model pipelines.
- [ ] Track model experiments.
- [ ] Package reproducible model artifacts.
- [ ] Serialize and validate a pipeline.
- [ ] Explain deployment and inference API concepts.
- [ ] Build a FastAPI prediction service with input validation.

**Phase complete:** Advanced ML and Practical Applications (Days 54–65).  
**Next phase:** Phase 7 — Deployment and MLOps (Days 66–76).
