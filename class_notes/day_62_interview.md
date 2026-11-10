# Day 62: Interview Questions — ML Project Reproducibility and Artifact Management

**Phase:** 6 — Advanced ML and Practical Applications

## 1. Core questions

### Q1. What does reproducibility mean in machine learning?
**Answer:**
- Another person can understand and rerun the workflow.
- Code, data, configuration and environment must be traceable.
- Model and preprocessing artifacts must be preserved.
- Metrics and validation strategy should be recorded.

### Q2. What should be versioned in an ML project?
**Answer:**
- Source code and configuration.
- Dataset version or reference.
- Dependencies and environment.
- Model and preprocessing artifacts.
- Evaluation reports and metadata.
- Random seeds and split configuration.

### Q3. Why is a model file alone insufficient?
**Answer:**
- It may not include preprocessing.
- It does not identify the training dataset or code.
- It may require specific dependency versions.
- It may not contain feature schema or target definition.

### Q4. What is an ML artifact?
**Answer:**
- A file or output produced or used by an ML workflow.
- Examples include models, tokenizers, scalers, metrics, plots and metadata.
- Artifacts should be versioned and associated with an experiment.
- Access and integrity should be controlled.

### Q5. Why use a random seed?
**Answer:**
- It controls some random operations.
- It makes data splitting and stochastic training more repeatable.
- It helps compare experiments.
- It does not guarantee bit-for-bit reproducibility across all environments.

### Q6. Why should we save a full preprocessing pipeline?
**Answer:**
- Inference needs the same transformation as training.
- Saving only the estimator can cause different feature representations.
- A full pipeline reduces training-serving mismatch.
- Input schema still needs validation.

## 2. Tools and safety

### Q7. What is joblib used for?
**Answer:**
- It can serialize Python objects such as scikit-learn estimators.
- It is often used to save fitted pipelines.
- The artifact can be loaded later for inference.
- Only load files from trusted sources because pickle-based formats can execute code.

### Q8. What is a checksum?
**Answer:**
- A checksum such as SHA-256 is a digest computed from file contents.
- It can detect whether a file has changed.
- It helps verify artifact integrity.
- It does not independently authenticate the source.

### Q9. Why pin dependency versions?
**Answer:**
- Library changes can affect behavior or compatibility.
- Pinned versions make the environment more predictable.
- Lock files can capture transitive dependencies.
- Validate the environment before release.

## 3. Practical scenarios

### Q10. The model works on the trainer's laptop but fails on another machine. What do you check?
**Answer:**
- Python and library versions.
- Missing dependencies.
- Artifact path and permissions.
- Input schema and feature order.
- Environment variables and configuration.
- Recreate the documented environment and rerun validation.

### Q11. A model was saved but its scaler was not. What is the risk?
**Answer:**
- New data may be passed to the estimator in a different scale.
- Predictions can become unreliable.
- Save and load the full fitted pipeline.
- Revalidate predictions after reload.

### Q12. Why should serialized models from unknown sources not be loaded?
**Answer:**
- Pickle/joblib deserialization can execute code.
- An untrusted file can compromise the environment.
- Verify source and integrity.
- Use safer, constrained formats where appropriate and follow organizational controls.

### Q13. How would you create a reproducible model package?
**Answer:**
1. Save training code and configuration.
2. Record dataset version and split strategy.
3. Record dependency versions and code revision.
4. Save the fitted pipeline.
5. Save metrics and feature metadata.
6. Generate artifact checksum.
7. Provide setup and inference instructions.
8. Test reload in a clean environment.

## 4. Rapid revision
- Reproducibility requires code, data, environment and configuration.
- Artifact: model, preprocessing, metrics or metadata.
- Save full pipeline.
- Record dependencies and data version.
- Seed improves repeatability but does not guarantee identical results.
- joblib/pickle files must be trusted before loading.
- Checksum detects content changes.
