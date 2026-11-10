# Day 64: Interview Questions — Model Deployment Concepts and Inference APIs

**Phase:** 6 — Advanced ML and Practical Applications

## 1. Core questions

### Q1. What is model deployment?
**Answer:**
- Making a trained model available to applications or users.
- It may run batch jobs, online APIs or streaming inference.
- Deployment includes dependencies, input validation and monitoring.
- A model file alone is not a complete deployment.

### Q2. What is inference?
**Answer:**
- Inference is using a trained model to generate predictions on new data.
- It should apply the same preprocessing used during training.
- It may run in batch or respond to individual requests.
- Input schema and model version must be controlled.

### Q3. Compare batch and online inference.
**Answer:**
- Batch processes many records at scheduled times.
- Online inference responds to requests in near real time.
- Batch is suitable for periodic scoring.
- Online is suitable for interactive applications with latency requirements.

### Q4. What is a REST API?
**Answer:**
- An interface that allows systems to communicate over HTTP.
- It commonly uses methods such as GET and POST.
- It exchanges structured data, often JSON.
- An ML API can accept features and return a prediction.

### Q5. What is FastAPI?
**Answer:**
- A Python framework for building APIs.
- It uses Python type hints for request validation and documentation.
- It supports asynchronous endpoints.
- It provides interactive API documentation.

### Q6. What is the purpose of a health endpoint?
**Answer:**
- It gives an operational signal that the service is responding.
- It can support monitoring and orchestration checks.
- A basic health response does not necessarily prove the model artifact is valid.
- Deeper readiness checks may verify dependencies and model loading.

## 2. Practical scenarios

### Q7. The API returns 422. What does it usually indicate?
**Answer:**
- Request validation failed.
- A required field may be missing.
- A field type or constraint may be invalid.
- Inspect the validation response and correct the request schema.

### Q8. The API starts, but predictions fail. What do you check?
**Answer:**
- Artifact path and load status.
- Expected feature names and types.
- Preprocessing pipeline compatibility.
- Package versions.
- Model input shape.
- Application logs without exposing secrets.

### Q9. How do you protect a public prediction API?
**Answer:**
- Use authentication and authorization.
- Enforce HTTPS.
- Validate payloads and limit request sizes.
- Apply rate limits.
- Avoid logging sensitive data.
- Monitor errors and access patterns.
- Store artifacts and secrets securely.

### Q10. Why should the model be loaded once rather than on every request?
**Answer:**
- Repeated loading adds latency and I/O overhead.
- Startup loading can make requests faster.
- The service should fail readiness if required artifacts cannot load.
- Model update strategy must account for worker processes and version consistency.

## 3. Project questions

### Q11. Describe the Prediction API architecture.
**Answer:**
1. Client sends a JSON request.
2. API validates fields and ranges.
3. Data is converted into the model's expected input format.
4. Fitted pipeline generates prediction.
5. API returns a stable JSON response.
6. Errors and operational metrics are handled safely.

### Q12. What is the difference between 400, 422 and 500?
**Answer:**
- 400: request is invalid at the application level.
- 422: request structure or values fail validation in common FastAPI workflows.
- 500: unexpected server-side failure.
- Use clear, safe error messages.

### Q13. How do you deploy a model update safely?
**Answer:**
- Validate the new artifact and schema.
- Run automated tests.
- Version the model.
- Deploy to a controlled environment.
- Monitor health and performance.
- Keep rollback capability.

### Q14. What is the difference between liveness and readiness?
**Answer:**
- Liveness indicates whether the process is alive and should continue running.
- Readiness indicates whether it is ready to serve traffic.
- Readiness may check model loading and required dependencies.
- Exact endpoint design depends on deployment platform.

## 4. Rapid revision
- Deployment: make model usable by an application.
- Inference: generate prediction from new input.
- REST: HTTP-based service interface.
- FastAPI: typed Python API framework.
- 422: request validation issue.
- Health: service status.
- Production: auth, HTTPS, monitoring, versioning and rollback.
