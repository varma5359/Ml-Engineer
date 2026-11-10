# Day 65: Interview Questions — FastAPI Model Serving and Input Validation

**Phase:** 6 — Advanced ML and Practical Applications

## 1. Core questions

### Q1. What is model serving?
**Answer:**
- Making a trained model available for inference.
- A service receives input, validates it, calls the model and returns predictions.
- It requires an input contract and a compatible model artifact.
- It also needs monitoring, security and version management.

### Q2. What is Pydantic?
**Answer:**
- A Python library for data validation and settings management.
- FastAPI uses Pydantic models to validate request bodies.
- It supports types and constraints.
- It helps create consistent request and response schemas.

### Q3. What is a request schema?
**Answer:**
- A definition of the fields an API accepts.
- It specifies names, data types and validation rules.
- It documents the expected input contract.
- It should match the model's feature requirements.

### Q4. What is response_model in FastAPI?
**Answer:**
- It defines the expected response structure.
- FastAPI validates and documents the response against the schema.
- It helps keep API output consistent.
- Avoid including internal or sensitive fields in responses.

### Q5. Why load the model at startup?
**Answer:**
- Avoids repeated disk loading for every request.
- Reduces request latency.
- Allows the application to detect artifact loading problems early.
- Readiness should remain false if required model resources are unavailable.

### Q6. What is the difference between health and readiness?
**Answer:**
- Health or liveness checks whether the service process is alive.
- Readiness checks whether it can serve requests.
- Readiness may verify model loading and dependencies.
- Platform-specific probe design should match operational needs.

## 2. Validation and errors

### Q7. What happens when a required field is missing?
**Answer:**
- Pydantic validation rejects the request.
- FastAPI commonly returns HTTP 422.
- The model should not be called with invalid input.
- The client should correct the payload according to the schema.

### Q8. Why validate numeric ranges?
**Answer:**
- Type correctness alone does not guarantee realistic values.
- A negative age or impossible count may be invalid for the business domain.
- Constraints reject invalid data early.
- Domain constraints must be based on actual requirements.

### Q9. How should internal prediction errors be returned?
**Answer:**
- Return a safe, non-sensitive error message.
- Log diagnostic details securely on the server.
- Avoid exposing stack traces, file paths or secrets.
- Use appropriate status codes.

## 3. Practical scenarios

### Q10. API returns 200 but prediction is wrong because feature order changed. How do you prevent this?
**Answer:**
- Validate feature names and schema.
- Build a DataFrame using the exact expected column names.
- Use a saved preprocessing pipeline.
- Add tests for input order and schema.
- Version the input contract.

### Q11. The service works locally but fails in deployment. What do you investigate?
**Answer:**
- Dependency versions and Python runtime.
- Artifact path and permissions.
- Environment configuration.
- Model input schema.
- Network and port configuration.
- Startup logs and readiness checks.

### Q12. How do you handle model updates without downtime?
**Answer:**
- Version artifacts.
- Validate the new model before release.
- Deploy using a controlled rollout strategy.
- Monitor the new version.
- Keep a rollback path.
- Ensure old and new versions have compatible request contracts or version the API.

### Q13. What should you monitor for a prediction service?
**Answer:**
- Request count and latency.
- Error and timeout rates.
- Resource usage.
- Input schema failures.
- Prediction distribution.
- Data drift and delayed model performance.
- Model and API version.

## 4. Project discussion

### Q14. Explain your ML Prediction Service.
**Answer:**
1. I trained and serialized a model pipeline.
2. I defined a Pydantic request schema.
3. I loaded the model at application startup.
4. I implemented health and prediction endpoints.
5. I validated input fields and constraints.
6. I returned a consistent response with model version.
7. I tested valid and invalid requests.

### Q15. Why separate schemas, routes and model logic into different files?
**Answer:**
- Schemas define API contracts.
- Routes handle HTTP behavior.
- Model service handles loading and prediction.
- Separation improves testing and maintenance.
- It reduces coupling as the service grows.

### Q16. What is the difference between 422 and 500?
**Answer:**
- 422 indicates request data failed validation.
- 500 indicates an unexpected server-side error.
- Clients should fix invalid request data.
- Server failures require investigation by the service team.

### Q17. Is a local FastAPI server production-ready?
**Answer:**
- No. A local development server is for development and testing.
- Production needs a suitable server process and deployment setup.
- Add authentication, HTTPS, monitoring, logging, resource limits and rollback.
- Validate scaling and concurrency behavior.

## 5. Rapid revision
- FastAPI: Python API framework.
- Pydantic: request/response validation.
- Schema: input/output contract.
- 422: validation failure.
- 500: internal server error.
- Startup model loading: avoids repeated load per request.
- Readiness: service is prepared to serve traffic.
- Production: security, monitoring, versioning and rollback.
