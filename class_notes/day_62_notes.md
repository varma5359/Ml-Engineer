# Day 62: ML Project Reproducibility and Artifact Management

**Pathway:** Machine Learning Engineer  
**Phase:** 6 — Advanced ML and Practical Applications  
**Level:** Intermediate  
**Estimated learning time:** 2–3 hours  
**Approved mini-project:** Reproducible Model Package

## 1. Learning objectives
- Explain reproducibility in ML projects.
- Identify what must be versioned and recorded.
- Save and load model artifacts safely.
- Record environment and configuration details.
- Create a package that another developer can reproduce.

## 2. What is reproducibility?
A reproducible ML workflow allows another person to understand and rerun the process using the same data, code, configuration and environment.

A model file alone is not enough. Reproducibility may require:
- Source code and Git revision.
- Dataset version or immutable reference.
- Python and library versions.
- Preprocessing steps.
- Random seeds and split configuration.
- Hyperparameters and evaluation metrics.
- Model artifact and schema.
- Instructions to run and validate.

## 3. Artifact types
| Artifact | Example |
|---|---|
| Data reference | DVC-tracked dataset version |
| Code | Git commit |
| Configuration | YAML/JSON parameters |
| Model | Serialized estimator or pipeline |
| Metrics | JSON/CSV evaluation report |
| Environment | requirements.txt or lock file |
| Evidence | plots, logs and README |

## 4. Set a random seed
```python
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
print("Seeds configured")
```

A seed helps control randomness in libraries that use these generators. It does not guarantee identical results across every library version, hardware setup or parallel execution mode.

## 5. Train and save a complete pipeline
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

dataset = load_breast_cancer(as_frame=True)
X, y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])
pipeline.fit(X_train, y_train)
print("Pipeline trained")
```

## 6. Save artifact with joblib
```python
from pathlib import Path
import joblib

artifact_dir = Path("artifacts")
artifact_dir.mkdir(exist_ok=True)

model_path = artifact_dir / "classifier.joblib"
joblib.dump(pipeline, model_path)
print("Saved:", model_path)
```

**Explanation:** The complete pipeline is saved, including scaler and estimator.

### 6.1 Load and test
```python
loaded_pipeline = joblib.load(model_path)
predictions = loaded_pipeline.predict(X_test)
print(predictions[:10])
```

Only load serialized artifacts from trusted sources. Pickle/joblib-based formats can execute code during loading and should not be treated as safe for untrusted files.

## 7. Save metadata
```python
import json
import sklearn
import joblib

metadata = {
    "model_name": "breast_cancer_logistic_pipeline",
    "random_state": 42,
    "python_libraries": {
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__
    },
    "features": list(X.columns),
    "target": "target"
}

with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print("Metadata saved")
```

For a real project, record Python version, dataset version, code revision, split details, training parameters and metrics as well.

## 8. Example package structure
```text
reproducible_model/
├── data/
│   └── README.md
├── artifacts/
│   ├── model.joblib
│   └── metadata.json
├── src/
│   ├── train.py
│   └── predict.py
├── reports/
│   └── metrics.json
├── requirements.txt
├── config.yaml
└── README.md
```

Do not commit sensitive data, credentials or large artifacts without considering repository policy and storage needs.

## 9. Environment capture
A simple environment file can list dependencies:
```text
pandas
numpy
scikit-learn
joblib
```

For a reliable release, pin tested versions or use a lock file. A loose requirements file is convenient for a demo but does not fully lock the environment.

## 10. Artifact integrity
A checksum can help identify whether an artifact changed.

```python
import hashlib

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()

print(sha256_file(model_path))
```

Store the checksum in a trusted manifest or artifact registry. A checksum detects changes; it does not by itself prove who created the file.

## 11. Approved mini-project — Reproducible Model Package

**Scenario:** Another engineer must be able to reproduce a model result and load the same preprocessing and estimator.

**Objective:** Package a trained model with metadata, environment instructions and validation evidence.

### Requirements
1. Train a pipeline with a fixed split and seed.
2. Save the complete fitted pipeline.
3. Save feature schema and target definition.
4. Record metrics and training parameters.
5. Record library versions and code revision.
6. Save a checksum for the artifact.
7. Provide a prediction script.
8. Document how to recreate the environment and verify predictions.

### Validation
- Reload artifact in a fresh Python process.
- Confirm expected input columns and order.
- Confirm prediction output shape.
- Compare predictions before and after serialization.
- Confirm metadata matches the actual training run.
- Never load untrusted serialized artifacts.

## 12. Common mistakes
- Saving model without preprocessing.
- Not recording dataset revision.
- Assuming random seed guarantees full reproducibility.
- Using unpinned dependencies in a production release.
- Loading untrusted pickle/joblib files.
- Storing secrets in metadata or Git.
- Overwriting model artifacts without version history.

## 13. Lesson completion
- [ ] Explain reproducibility.
- [ ] Save and reload a complete pipeline.
- [ ] Record model metadata and dependencies.
- [ ] Generate an artifact checksum.
- [ ] Complete Reproducible Model Package.

**Next lesson:** Day 63 — Scikit-learn pipelines and serialization.
