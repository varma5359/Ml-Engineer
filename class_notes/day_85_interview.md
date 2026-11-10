# Day 85: Interview Questions — Capstone Prediction API

**Phase:** 8 — End-to-End ML Capstone

## 1. Core questions

### Q1. What is model inference?
**Answer:**
- Using a trained model to generate predictions for new inputs.
- It does not normally retrain the model.
- Inputs must follow the model's feature contract.
- The output may be a class, score or probability.

**Follow-up:** What is the difference between batch and online inference?

### Q2. Why use FastAPI for model serving?
**Answer:**
- It provides Python API routes and request validation.
- Pydantic schemas define request and response contracts.
- It generates OpenAPI documentation.
- It supports asynchronous service patterns where needed.

**Follow-up:** When might a batch scoring job be preferable to an API?

### Q3. Why load the model at startup?
**Answer:**
- Avoid repeated disk loading for each request.
- Reduce prediction latency.
- Detect missing or invalid artifacts early.
- Keep readiness false when the required model cannot load.

**Follow-up:** How do you safely update the model version?

### Q4. What is a request schema?
**Answer:**
- A definition of expected input fields and types.
- It can enforce ranges and required values.
- It acts as an API contract.
- It should match the model's actual training features.

**Follow-up:** How would you handle optional features?

### Q5. Why include model version in the response or logs?
**Answer:**
- It identifies which artifact generated the prediction.
- It supports debugging and performance analysis.
- It enables rollback investigation.
- It improves traceability.

**Follow-up:** What other metadata should be recorded with a prediction?

## 2. Scenario questions

### Q6. The API returns a 422 response. What does it usually indicate?
**Answer:**
- Request validation failed.
- A required field may be missing or have an invalid type/value.
- Inspect the schema and response details.
- Correct the client payload rather than bypassing validation.

**Follow-up:** How do you distinguish validation failure from model failure?

### Q7. API prediction differs from notebook prediction for the same customer. What do you inspect?
**Answer:**
- Feature names, order, types and units.
- Target class and probability mapping.
- Model artifact version.
- Preprocessing pipeline and missing-value handling.
- Input serialization and category spelling.

**Follow-up:** How can a full fitted pipeline reduce this risk?

### Q8. The model file is missing during application startup. What should happen?
**Answer:**
- Startup/readiness should fail for a service that requires the artifact.
- Return a safe unavailable status if the service is already running but not ready.
- Log the artifact issue securely.
- Restore the correct trusted artifact and verify its version.

**Follow-up:** Why is returning a default prediction unsafe?

## 3. Project questions

### Q9. Describe your Inference service.
**Answer:**
- I created request and response schemas.
- I loaded the trusted fitted pipeline once.
- I implemented health and prediction endpoints.
- I validated the request and returned prediction, probability and model version.
- I tested invalid input and artifact failure.

**Follow-up:** How would you secure the API before external exposure?

### Q10. How do you test an ML prediction API?
**Answer:**
- Test valid input and expected response shape.
- Test missing fields and invalid types/ranges.
- Test feature schema and class mapping.
- Compare API predictions with direct pipeline predictions.
- Test artifact availability and safe error behavior.

**Follow-up:** How would you test compatibility when deploying a new model version?

## 4. Rapid revision
- Inference: prediction using a trained model.
- FastAPI: Python API framework.
- Pydantic: request/response validation.
- Startup: load required model resources.
- 422: request validation error commonly returned by FastAPI.
- Model version: prediction traceability.
- Pipeline: consistent preprocessing and prediction.
