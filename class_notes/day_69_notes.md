# Day 69: Model Registry and Version Management

**Pathway:** Machine Learning Engineer  
**Phase:** 7 — Deployment and MLOps  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Approved mini-project:** Model Version Registry

## 1. Learning objectives
- Explain why models need a registry.
- Distinguish experiment runs from registered model versions.
- Register a model in MLflow.
- Add descriptions, tags and version metadata.
- Explain promotion, rollback and approval controls.

## 2. What is a model registry?
A model registry is a managed catalogue of model versions and their lifecycle information.

A registered model can include:
- Model name and version.
- Source training run.
- Model artifact location.
- Description and tags.
- Validation evidence.
- Approval or deployment status.

**Flow:** Training run → logged model → registry version → validation/approval → deployment.

A registry does not automatically prove a model is safe or suitable for production.

## 3. Run MLflow locally
Install MLflow:
```bash
python -m pip install mlflow scikit-learn
```

For this lesson, use the same local tracking store and artifact configuration for logging and registry operations. Some registry capabilities require a database-backed tracking server and a suitable artifact store; verify requirements for your MLflow version.

## 4. Train and log a model
```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

mlflow.set_experiment("model_registry_learning")

with mlflow.start_run(run_name="iris_rf_v1") as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("n_estimators", 100)
    mlflow.sklearn.log_model(
        model,
        name="iris_model",
        registered_model_name="IrisClassifier"
    )

    print("Run ID:", run.info.run_id)
```

**Note:** API signatures and registry behavior may vary by MLflow version. The example demonstrates registration through model logging. Use a validation set for model selection; the test metric here is only a compact API demonstration.

## 5. Inspect registered model versions
```python
from mlflow import MlflowClient

client = MlflowClient()
versions = client.search_model_versions("name='IrisClassifier'")

for version in versions:
    print(version.name, version.version, version.status)
```

The registry records versions and their source. The exact status fields available depend on the backend and version.

## 6. Add model metadata
```python
latest = max(versions, key=lambda item: int(item.version))

client.update_model_version(
    name="IrisClassifier",
    version=latest.version,
    description="Baseline iris classifier for registry practice"
)

client.set_model_version_tag(
    name="IrisClassifier",
    version=latest.version,
    key="validation_status",
    value="pending_review"
)
```

Metadata should be factual and traceable. Do not label a model production-ready before the required reviews.

## 7. Load a specific version
```python
model_uri = f"models:/IrisClassifier/{latest.version}"
registered_model = mlflow.sklearn.load_model(model_uri)
print(registered_model.predict(X_test.iloc[:3]))
```

Pinning a version makes the requested artifact explicit. Avoid relying on an ambiguous "latest" version in production.

## 8. Version lifecycle and promotion
A practical governance flow might be:
- Candidate: newly registered.
- Validation: automated and manual checks.
- Approved: authorized for a defined use.
- Production: deployed and monitored.
- Archived: no longer active but retained for traceability.

MLflow's available stage/alias mechanisms differ by version. Use the lifecycle feature supported by your installed version and organization policy.

## 9. Rollback strategy
- Keep the previous approved version.
- Record which version is deployed in each environment.
- Validate the rollback artifact and schema.
- Switch traffic/configuration to the known-good version.
- Monitor after rollback and record the incident.

## 10. Approved mini-project — Model Version Registry

**Scenario:** A team retrains a churn model frequently and needs to identify, review and deploy a specific approved version.

**Requirements**
1. Log at least two model runs.
2. Register model versions.
3. Attach source run, dataset version and code revision.
4. Add description and validation tags.
5. Define candidate/approved/production policy.
6. Load a specific version for inference.
7. Demonstrate how a previous version could be restored.
8. Document access and approval responsibilities.

**Suggested structure**
```text
model_version_registry/
├── src/train_register.py
├── reports/validation.md
├── config/
└── README.md
```

**Validation:** Confirm each version maps to the intended run and artifact. Check version-specific predictions. Do not promote based only on a single metric.

## 11. Common mistakes
- Confusing a run ID with a model version.
- Registering an unvalidated artifact as approved.
- Deploying a moving "latest" reference.
- Missing dataset and code lineage.
- Overwriting the previous production version.
- Assuming registry metadata replaces model testing.

## 12. Completion checklist
- [ ] Explain registry and version.
- [ ] Register a model.
- [ ] Add metadata and tags.
- [ ] Load a specific version.
- [ ] Explain promotion and rollback.
- [ ] Complete Model Version Registry.

**Next lesson:** Day 70 — DVC and dataset/model versioning.
