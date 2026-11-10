# Day 68: Interview Questions — MLflow Experiment Tracking Fundamentals

**Phase:** 7 — Deployment and MLOps

## 1. Core questions

### Q1. What is experiment tracking?
**Answer:**
- Recording the details and results of model training runs.
- It captures parameters, metrics, tags and artifacts.
- It supports comparisons and reproducibility.
- Dataset and code versions should also be recorded.

### Q2. What is MLflow?
**Answer:**
- An open-source platform for managing parts of the ML lifecycle.
- MLflow Tracking records experiment runs.
- It can log models and artifacts.
- Other MLflow components support model packaging and registry workflows.

### Q3. What is the difference between an experiment and a run?
**Answer:**
- An experiment groups related training runs.
- A run represents one execution of a workflow.
- Each run can have its own parameters, metrics, tags and artifacts.
- Multiple runs can be compared within an experiment.

### Q4. Compare parameters and metrics.
**Answer:**
- Parameters are settings such as max_depth or learning_rate.
- Metrics are measured results such as F1 or RMSE.
- Parameters describe configuration.
- Metrics describe observed performance.

### Q5. What is an artifact?
**Answer:**
- A file produced or stored by a run.
- Examples include model files, plots, reports and schemas.
- Artifacts help preserve evidence.
- Sensitive data should not be logged casually.

### Q6. What is a tracking URI?
**Answer:**
- It identifies the MLflow tracking store or server.
- It determines where run metadata is recorded.
- A local URI is useful for learning.
- Team environments need controlled server and artifact storage configuration.

## 2. Scenario questions

### Q7. Two runs have different F1 scores. Can you choose the higher one immediately?
**Answer:**
- Check that data, target, split and metric definitions match.
- Compare validation results and stability.
- Consider runtime and complexity.
- Keep the test set for final evaluation.

### Q8. A run has metrics but no dataset version. What is missing?
**Answer:**
- Data lineage is incomplete.
- Record immutable dataset version or checksum.
- Record code revision and preprocessing configuration.
- Re-run if the original data cannot be identified.

### Q9. Why should a local MLflow UI not be exposed publicly without controls?
**Answer:**
- It may expose experiment metadata and artifacts.
- It may permit unauthorized access or modification depending on configuration.
- Use network restrictions, authentication and authorization.
- Protect artifact storage and credentials.

### Q10. What would you log for a model run?
**Answer:**
- Model family and hyperparameters.
- Dataset and code version.
- Validation metrics and split strategy.
- Environment versions.
- Model and evaluation artifacts.
- Notes explaining the run.

## 3. Project questions

### Q11. Describe your Experiment Tracking Workspace.
**Answer:**
- I created a named MLflow experiment.
- I trained multiple model configurations.
- I logged parameters, validation metrics and tags.
- I stored the feature schema and evaluation plot.
- I compared runs and documented the selected candidate.

### Q12. Does experiment tracking guarantee reproducibility?
**Answer:**
- No. It records run information.
- Exact reproducibility also needs data, code, dependencies and environment details.
- Randomness and hardware can affect results.
- A rerun should be validated.

### Q13. How do you avoid leaking information into experiment artifacts?
**Answer:**
- Review artifacts before logging.
- Do not log credentials or unnecessary personal data.
- Restrict artifact access.
- Apply retention and deletion policies.
- Use synthetic or aggregated examples where possible.

## 4. Rapid revision
- Experiment: group of related runs.
- Run: one training execution.
- Parameter: model setting.
- Metric: measured result.
- Artifact: output file.
- Tracking URI: metadata store location.
- Record dataset and code versions.
