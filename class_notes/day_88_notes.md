# Day 88: Capstone — Experiment Tracking and Versioning

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved mini-project / deliverable:** Reproducibility package

## 1. Learning objectives
- Connect a selected model to its MLflow run.
- Record parameters, metrics and model artifacts.
- Link the run to dataset and source-code versions.
- Track large data and pipeline files with DVC.
- Create a reproducibility manifest for the deployed model.

## 2. What is reproducibility?
A reproducible ML result should identify:
- Source code revision.
- Dataset version or checksum.
- Feature and target definitions.
- Training configuration and random seeds.
- Dependency/environment versions.
- Evaluation results.
- Model artifact and registry version.

MLflow and DVC complement each other: MLflow records experiment information and artifacts, while DVC can version data and pipeline inputs/outputs.

## 3. Record a run with MLflow
Install and configure MLflow using the tracking setup from Day 68:
```bash
python -m pip install mlflow
```

Example:
```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("customer_churn_capstone")

with mlflow.start_run(run_name="optimized_churn_model") as run:
    mlflow.log_param("model_family", "random_forest")
    mlflow.log_param("threshold", 0.35)
    mlflow.log_param("dataset_version", "replace-with-real-dvc-revision")
    mlflow.log_metric("validation_pr_auc", validation_pr_auc)
    mlflow.log_metric("validation_recall", validation_recall)
    mlflow.set_tag("code_revision", "replace-with-real-git-commit")
    mlflow.sklearn.log_model(best_model, name="churn_pipeline")
    print("Run ID:", run.info.run_id)
```

Replace all placeholder values with actual metadata. Use validation metrics for selection. Do not log a metric name that does not match the data used to calculate it.

## 4. Record source and dataset revisions
Useful commands:
```bash
git rev-parse HEAD
dvc status
dvc list . data
```

Use the actual DVC metadata and repository history to identify the dataset version. A command's output depends on repository setup; record the relevant commit and DVC pointer/checksum rather than inventing a version label.

## 5. Track the dataset with DVC
If not already tracked:
```bash
dvc add data/raw/customer_churn.csv
git add data/raw/customer_churn.csv.dvc data/raw/.gitignore
git commit -m "Track churn dataset version"
dvc push
```

Use an approved remote. Never commit cloud access credentials.

## 6. Create a reproducibility manifest
```python
import json
import platform
import sklearn
import pandas
import mlflow

manifest = {
    "project": "Customer Churn Prediction and ML Deployment System",
    "python_version": platform.python_version(),
    "scikit_learn_version": sklearn.__version__,
    "pandas_version": pandas.__version__,
    "mlflow_version": mlflow.__version__,
    "dataset_revision": "INSERT_REAL_DVC_REVISION",
    "git_commit": "INSERT_REAL_GIT_COMMIT",
    "mlflow_run_id": "INSERT_REAL_RUN_ID",
    "model_version": "INSERT_REGISTERED_MODEL_VERSION"
}

with open("reports/reproducibility_manifest.json", "w") as file:
    json.dump(manifest, file, indent=2)
```

A manifest is useful only when the fields contain verified values and refer to accessible artifacts.

## 7. Verify the artifact and run linkage
Check:
- Model artifact is accessible from its logged run.
- Run identifies dataset and code revision.
- Feature schema is stored.
- Dependency versions are recorded.
- Registered version points to the intended run.
- Deployment image and model versions are known.

## 8. Approved deliverable — Reproducibility package

**Objective:** Assemble the evidence needed to identify and reproduce the selected capstone model.

### Tasks
1. Log the selected training run in MLflow.
2. Record validation metrics and configuration.
3. Link the run to a real Git commit.
4. Track the dataset or preprocessing output with DVC.
5. Record dataset revision/checksum.
6. Store the fitted pipeline and feature schema.
7. Record Python and library versions.
8. Link the registered model version and container image tag.
9. Create a reproducibility manifest and README instructions.

**Suggested structure**
```text
customer_churn_capstone/
├── reports/reproducibility_manifest.json
├── reports/feature_schema.json
├── dvc.yaml
├── dvc.lock
├── artifacts/
└── README.md
```

**Validation:** A second environment should be able to identify the same source code, dataset and model artifact. Record unavailable dependencies or remote access limitations instead of claiming complete reproduction.

## 9. AI-assisted implementation guide
| Level | Prompt |
|---|---|
| Hint | "Give me one hint for linking an MLflow run to a DVC dataset revision." |
| Plan | "Create a checklist for a reproducibility manifest." |
| Example | "Show a compact JSON manifest with code, data and model identifiers." |
| Guided implementation | "Help me log the dataset revision and code commit to my MLflow run." |
| Explanation | "Explain how MLflow and DVC complement each other." |
| Debugging | "Help me diagnose why my model artifact cannot be loaded in a clean environment." |
| Review | "Review my reproducibility package for missing lineage or placeholder values." |

## 10. Daily quiz

**Q1. What does MLflow primarily record in experiment tracking?**  
A. Only Git branches  
B. Runs, parameters, metrics, tags and artifacts  
C. Only raw customer data  
D. Only Docker networks  
**Correct: B.** These are core tracking records.

**Q2. What is DVC commonly used to version?**  
A. Large datasets and pipeline outputs  
B. UI colors  
C. API requests only  
D. Interview answers  
**Correct: A.** DVC complements Git for data and pipeline versioning.

**Q3. Which value identifies a source-code revision?**  
A. Git commit SHA  
B. Customer tenure  
C. Model probability  
D. HTTP status  
**Correct: A.** A commit SHA identifies a repository revision.

**Q4. Why record dependency versions?**  
A. To make reports longer  
B. To support environment reconstruction and compatibility  
C. To remove the need for tests  
D. To alter the target  
**Correct: B.** Dependencies can change model loading and behavior.

**Q5. What should you do with a placeholder revision in a manifest?**  
A. Claim it is verified  
B. Replace it with the actual verified revision before submission  
C. Delete all metadata  
D. Use a random string  
**Correct: B.** Reproducibility requires real traceable identifiers.

## 11. Interview preparation

### Q1. How do MLflow and DVC work together?
**Answer:** MLflow tracks experiment runs, parameters, metrics and model artifacts. DVC tracks large datasets and pipeline inputs/outputs alongside Git. Recording the DVC data revision in the MLflow run links the experiment to the exact data used.

**Follow-up:** What would you do if the DVC remote data was unavailable?

### Q2. What should a reproducibility manifest contain?
**Answer:** It should identify source code commit, dataset revision, feature schema, training configuration, dependency versions, MLflow run, registered model version and deployed image tag. Values must be real and traceable.

**Follow-up:** Does a manifest alone guarantee bit-for-bit reproducibility?

## 12. Completion checklist
- [ ] Log the selected run in MLflow.
- [ ] Record dataset and code versions.
- [ ] Track data with DVC.
- [ ] Save dependency and feature schema metadata.
- [ ] Link model and container versions.
- [ ] Complete Reproducibility package.

**Next lesson:** Day 89 — Capstone: Monitoring and documentation.
