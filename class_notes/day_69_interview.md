# Day 69: Interview Questions — Model Registry and Version Management

**Phase:** 7 — Deployment and MLOps

## 1. Core questions

### Q1. What is a model registry?
**Answer:**
- A managed catalogue of model versions.
- It links versions to training runs and artifacts.
- It stores descriptions, tags and lifecycle information.
- It supports controlled model review and deployment.

### Q2. What is the difference between an MLflow run and a registered model version?
**Answer:**
- A run records one execution of training or evaluation.
- A registered version is a model artifact added to a named registry entry.
- A version can reference its source run.
- One model name can have multiple versions.

### Q3. Why use model versioning?
**Answer:**
- To know exactly which model is deployed.
- To compare new and previous releases.
- To reproduce results and investigate incidents.
- To support controlled rollback.

### Q4. What metadata should be stored?
**Answer:**
- Model name and version.
- Source run ID.
- Dataset and code revision.
- Validation metrics and test evidence.
- Feature schema and dependency versions.
- Approval and deployment information.

### Q5. What is model promotion?
**Answer:**
- Moving a candidate through review toward an approved deployment.
- It should require defined validation criteria.
- Authorization and audit history are important.
- Promotion is not simply selecting the newest version.

### Q6. Why avoid deploying "latest" directly?
**Answer:**
- The reference may change when a new version is registered.
- The deployed artifact can become ambiguous.
- Pin a specific approved version.
- Record the deployed version for rollback and audits.

## 2. Scenario questions

### Q7. Version 4 performs worse than production version 3. What do you do?
**Answer:**
- Do not promote version 4.
- Compare data, code and validation results.
- Keep version 3 active.
- Investigate the regression and record the decision.

### Q8. A registry contains a model but no dataset version. What is the risk?
**Answer:**
- Training lineage is incomplete.
- The result may be difficult to reproduce.
- Add dataset revision/checksum and code revision.
- Do not claim full traceability until evidence is available.

### Q9. Production predictions change unexpectedly after a new model is registered. What do you inspect?
**Answer:**
- Whether deployment uses a moving alias or version reference.
- Registry and deployment event history.
- Artifact checksum and model version.
- Feature schema and preprocessing.
- Configuration changes and rollout logs.

### Q10. Who should approve production promotion?
**Answer:**
- It depends on organizational governance.
- Usually designated model owners, engineering and risk/business reviewers have defined responsibilities.
- Approval criteria should be documented.
- A developer should not silently bypass required controls.

## 3. Project questions

### Q11. Describe your Model Version Registry.
**Answer:**
- I logged multiple training runs.
- I registered model artifacts under one model name.
- I attached run and dataset lineage.
- I added validation metadata.
- I loaded a specific version and documented rollback.

### Q12. Does a model registry guarantee model quality?
**Answer:**
- No. It manages artifacts and lifecycle metadata.
- Quality depends on data, validation and review.
- Automated tests and approval policies are still required.
- Production monitoring continues after deployment.

### Q13. How do you roll back?
**Answer:**
1. Identify the last known-good version.
2. Verify artifact and schema compatibility.
3. Change deployment configuration to that exact version.
4. Restore traffic safely.
5. Monitor service and prediction behavior.
6. Record the rollback event.

## 4. Rapid revision
- Run: one experiment execution.
- Registry: managed model catalogue.
- Version: immutable model release reference.
- Promotion: controlled lifecycle decision.
- Pin deployed version.
- Keep previous approved version for rollback.
