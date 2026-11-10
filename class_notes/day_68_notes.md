# Day 68: MLflow Experiment Tracking Fundamentals

**Pathway:** Machine Learning Engineer  
**Phase:** 7 — Deployment and MLOps  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Approved mini-project:** Experiment Tracking Workspace

## 1. Learning objectives
- Explain experiment tracking and MLflow Tracking.
- Create an experiment and start runs.
- Log parameters, metrics, tags and artifacts.
- Search and compare recorded runs.
- Explain tracking server and artifact storage at a high level.

## 2. Why experiment tracking?
Training runs can differ by dataset, model, parameters, code and environment. Without records, teams cannot reliably answer which run produced a result or why a model was selected.

An experiment record typically includes:
- Parameters: settings supplied to a run.
- Metrics: measured results.
- Tags: searchable metadata.
- Artifacts: files such as plots, reports and model files.
- Source information: code and dataset version.

## 3. What is MLflow Tracking?
MLflow Tracking records experiment runs and their metadata. It can use a local tracking store for learning and a configured tracking server and artifact store for team environments.

**Flow:** Python training script → MLflow client → tracking store + artifact store → UI/search for comparison.

## 4. Install and launch the local UI
```bash
python -m pip install mlflow scikit-learn pandas matplotlib
mlflow ui --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. The default local setup is intended for learning, not as an internet-exposed production tracking server.

## 5. Prepare a model experiment
```python
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

## 6. Create experiment and log a run
```python
mlflow.set_experiment("phase7_tracking_workspace")

with mlflow.start_run(run_name="rf_baseline"):
    n_estimators = 120
    max_depth = 5

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
    mlflow.log_metric("f1", f1_score(y_test, predictions))
    mlflow.set_tag("model_family", "random_forest")
    mlflow.log_dict({"features": list(X.columns)}, "feature_schema.json")
    mlflow.sklearn.log_model(model, name="model")

print("Run logged")
```

**Explanation:**
- `set_experiment` selects or creates the experiment.
- `start_run` opens a run context.
- `log_param` records a training setting.
- `log_metric` records a numeric result.
- `set_tag` records searchable metadata.
- `log_dict` stores a JSON artifact.
- `log_model` logs a model artifact; exact model logging APIs can vary across MLflow versions.

For model selection, use validation metrics during tuning and reserve an untouched test set for final evaluation.

## 7. Compare runs programmatically
```python
runs = mlflow.search_runs(
    experiment_names=["phase7_tracking_workspace"],
    order_by=["metrics.f1 DESC"]
)
print(runs[["run_name", "params.n_estimators",
            "params.max_depth", "metrics.f1"]].head())
```

The result columns depend on which values were logged. The MLflow UI also supports run comparison.

## 8. Log a plot as an artifact
```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
mlflow.log_artifact("confusion_matrix.png")
plt.close()
```

Artifacts preserve useful evidence with the run. Avoid logging sensitive raw records or credentials.

## 9. Tracking URI and team setup
- A local file-based tracking store is useful for a single-user exercise.
- A shared tracking server centralizes run metadata.
- Artifact storage may use local or remote object storage.
- Configure authentication, permissions, backups and network controls for team usage.
- Keep credentials outside source code.

## 10. Approved mini-project — Experiment Tracking Workspace

**Scenario:** A team trains multiple candidate models and needs a searchable record of experiments.

**Requirements**
1. Create an MLflow experiment.
2. Train at least three model configurations.
3. Log model parameters and validation metrics.
4. Add tags for model family, dataset version and code revision.
5. Log a plot and feature schema as artifacts.
6. Compare runs in the UI or with `search_runs`.
7. Write a short candidate-selection summary.

**Suggested structure**
```text
experiment_tracking_workspace/
├── src/train.py
├── reports/
├── artifacts/
├── requirements.txt
└── README.md
```

**Validation:** Confirm each run has the expected parameters, metrics and artifacts. Verify that run metadata identifies the data and code version. Do not tune against the final test set.

## 11. Common mistakes
- Logging a metric without identifying the dataset.
- Using test metrics repeatedly for tuning.
- Logging secrets or sensitive data as artifacts.
- Mixing experiments with different target definitions.
- Assuming local MLflow UI is a secure shared server.
- Not recording library versions and code revision.

## 12. Completion checklist
- [ ] Explain experiment, run, parameter, metric and artifact.
- [ ] Log a training run.
- [ ] Search and compare runs.
- [ ] Log an artifact.
- [ ] Complete Experiment Tracking Workspace.

**Next lesson:** Day 69 — Model registry and version management.
