# Day 76: Production ML Architecture and Reliability

**Pathway:** Machine Learning Engineer  
**Phase:** 7 — Deployment and MLOps  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Approved mini-project:** Production ML Architecture

## 1. Learning objectives
- Explain the main components of a production ML system.
- Trace offline training and online inference workflows.
- Identify reliability, scalability and failure-handling requirements.
- Design health, readiness, monitoring and rollback mechanisms.
- Create a production architecture diagram and operational checklist.

## 2. What is production ML architecture?
Production ML architecture describes how data, training, artifacts, serving applications and monitoring components work together reliably.

A model is only one part of the system. Production readiness also depends on data quality, deployment, security, observability, ownership and recovery.

## 3. Reference architecture

```text
                OFFLINE / TRAINING
Data Sources → Data Validation → Feature Preparation
                                  ↓
                         Model Training
                                  ↓
                     Evaluation / Approval
                                  ↓
                      Model Registry
                                  ↓
                       Versioned Artifact
                                  |
                                  v
                ONLINE / INFERENCE
Client → API Gateway → Prediction Service → Model Artifact
                            ↓                    ↓
                      Input Validation       Prediction
                            ↓                    ↓
                         Logs / Metrics ← Response
                            ↓
                 Monitoring and Alerting
                            ↓
                  Feedback / Ground Truth
                            ↓
                    Retraining Workflow
```

The diagram is a conceptual reference. Actual services and deployment boundaries depend on workload, risk and infrastructure.

## 4. Offline training path
1. Ingest approved data.
2. Validate schema, quality and lineage.
3. Prepare features using reproducible code.
4. Train candidate models.
5. Evaluate against baseline and business criteria.
6. Register the approved artifact with metadata.
7. Package and release a specific model version.

## 5. Online inference path
1. Client sends a request.
2. Gateway applies routing and access controls.
3. Service validates the input contract.
4. The model pipeline produces a prediction.
5. API returns a stable response.
6. Operational and model-monitoring events are recorded safely.

Online services need predictable latency and graceful error handling. Batch scoring may be more appropriate when real-time response is not required.

## 6. Reliability concepts

| Concept | Meaning |
|---|---|
| Availability | Service is accessible when required |
| Latency | Time taken to respond |
| Throughput | Requests processed per unit time |
| Scalability | Ability to handle increased load |
| Fault tolerance | Continue or degrade safely when components fail |
| Recovery | Restore service after failure |
| SLO | Target for a service-level indicator |
| SLA | Formal service commitment where applicable |

## 7. Health and readiness
- **Liveness:** Is the process alive?
- **Readiness:** Can it safely serve requests?
- Readiness may check model loading and critical dependencies.
- A health endpoint should not expose sensitive internal details.

A service should fail readiness when a required model artifact cannot be loaded.

## 8. Failure handling
Plan for:
- Missing or corrupt model artifact.
- Invalid request payload.
- Feature store or database unavailable.
- Registry or storage unavailable during startup.
- High latency or resource exhaustion.
- Model version/schema mismatch.
- Monitoring pipeline failure.

Use timeouts, bounded retries where safe, clear error responses, circuit breakers where appropriate and documented rollback procedures. Do not retry non-idempotent operations blindly.

## 9. Capacity and deployment choices
Choose based on:
- Online versus batch inference.
- Expected request volume and peak load.
- Model size and memory requirements.
- Latency target.
- CPU/GPU needs.
- Cost and operational skill.
- Data residency and security constraints.

Possible deployment patterns include a single service, multiple API replicas behind a load balancer, batch jobs, streaming consumers or managed ML platforms.

## 10. Example reliability checklist
```python
production_checks = {
    "model_version_pinned": True,
    "input_schema_validated": True,
    "readiness_check": True,
    "authentication_configured": True,
    "latency_monitored": True,
    "error_rate_monitored": True,
    "rollback_documented": True
}

for check, passed in production_checks.items():
    print(check, "PASS" if passed else "REVIEW")
```

This is a checklist demonstration, not proof that the controls are implemented or effective.

## 11. Approved mini-project — Production ML Architecture

**Scenario:** A business plans to expose a customer prediction model to an internal application and needs a reliable end-to-end architecture.

**Requirements**
1. Define business use case, users and inference mode.
2. Draw offline training and online serving flows.
3. Identify data validation, feature preparation and model registry components.
4. Define API gateway, authentication and model serving responsibilities.
5. Define health/readiness and operational monitoring.
6. Identify at least six failure scenarios and mitigations.
7. Define model promotion, rollback and retraining feedback.
8. Document reliability targets and assumptions.
9. Explain security, privacy and governance boundaries.

**Suggested structure**
```text
production_ml_architecture/
├── docs/architecture.md
├── docs/failure_modes.md
├── docs/operations_checklist.md
└── README.md
```

**Submission evidence:** Architecture diagram, component responsibilities, failure-mode table, monitoring plan, deployment/rollback flow and assumptions.

## 12. Phase 7 cumulative assessment — Deployment and MLOps

**Scenario:** A subscription company has a validated churn model and wants to make it available to internal customer-success staff. The service must be repeatable, versioned, monitored and safe to update.

### Tasks
1. Build a simple Streamlit interface for a prediction pipeline.
2. Containerize the API or application with Docker.
3. Track training experiments in MLflow.
4. Register at least two model versions and identify the approved version.
5. Version the dataset or pipeline inputs with DVC.
6. Create a CI workflow that runs tests and data/model validation.
7. Define data-drift and performance-monitoring reports.
8. Design a retraining trigger and approval workflow.
9. Create an ML risk register.
10. Draw the end-to-end production architecture and rollback path.

### Assessment rubric

| Area | Weight |
|---|---:|
| Architecture and component understanding | 20% |
| Practical correctness and integration | 25% |
| Versioning and reproducibility | 15% |
| CI, testing and validation | 15% |
| Monitoring and retraining reasoning | 10% |
| Security, privacy and governance | 10% |
| Documentation and evidence | 5% |
| **Total** | **100%** |

### Validation and common failure points
- Model and feature schema do not match.
- Container works locally but not through its mapped port.
- MLflow run is missing dataset/code lineage.
- DVC pointer exists but remote data is unavailable.
- CI tests pass but model-quality gate is absent.
- Drift alert is treated as automatic retraining.
- New model is deployed without approval or rollback.
- Sensitive data or secrets are included in logs/images.

### Interview assessment
1. Explain the difference between CI/CD and ML model monitoring.
2. How do you deploy a specific approved model version?
3. How do you investigate drift when labels are delayed?
4. What is the difference between liveness and readiness?
5. How do you roll back a model without changing the API contract?
6. How do you secure model artifacts and inference endpoints?
7. Why is a model registry not a substitute for model evaluation?

## 13. Common mistakes
- Designing only the model and ignoring the service around it.
- No version link between training and deployed artifact.
- No readiness check or rollback plan.
- Treating monitoring as dashboard-only work.
- No capacity or latency assumptions.
- Ignoring operational ownership and incident response.

## 14. Phase 7 completion checklist
- [ ] Build an interactive model UI.
- [ ] Containerize an ML application.
- [ ] Track experiments with MLflow.
- [ ] Register and version models.
- [ ] Version data and pipelines with DVC.
- [ ] Add CI checks and ML validation gates.
- [ ] Monitor data drift and model performance.
- [ ] Design retraining and feedback loops.
- [ ] Complete an ML risk assessment.
- [ ] Design production architecture and reliability controls.

**Phase complete:** Deployment and MLOps (Days 66–76).  
**Next phase:** Phase 8 — End-to-End ML Capstone (Days 77–90).
