# Day 88: Interview Questions — Capstone Experiment Tracking and Versioning

**Phase:** 8 — End-to-End ML Capstone

## 1. Core questions

### Q1. What makes an ML experiment reproducible?
**Answer:**
- Code and configuration are versioned.
- Dataset and preprocessing inputs are identified.
- Dependencies and environment are recorded.
- Model artifacts and evaluation evidence are preserved.
- Randomness and hardware-related limitations are documented.

**Follow-up:** What can prevent bit-for-bit reproduction?

### Q2. How does MLflow help?
**Answer:**
- Records training runs.
- Logs parameters, metrics, tags and artifacts.
- Links a model artifact to its source run.
- Supports searching and comparing experiments.

**Follow-up:** What information would you log for the final churn candidate?

### Q3. How does DVC help?
**Answer:**
- Tracks large data and pipeline artifacts.
- Stores metadata in files committed to Git.
- Uses a cache and configured remote for data contents.
- Supports reproducing data pipeline stages.

**Follow-up:** What happens if the Git pointer is available but the remote is not?

### Q4. Why record the Git commit SHA?
**Answer:**
- It identifies the exact source revision.
- It supports code review and reproduction.
- It links a model run to the implementation.
- It helps investigate changes between releases.

**Follow-up:** Why is a branch name alone less precise?

### Q5. What should a model artifact be linked to?
**Answer:**
- Training run and model configuration.
- Dataset version and feature schema.
- Code revision and dependencies.
- Evaluation report and registered model version.
- Deployment image and environment.

**Follow-up:** How do you verify the registry version points to the correct run?

## 2. Scenario questions

### Q6. A teammate loads the model but gets a scikit-learn version warning or error. What do you inspect?
**Answer:**
- Training and serving library versions.
- Artifact provenance and compatibility.
- Dependency lock or environment specification.
- Model serialization and feature schema.
- Rebuild and validate in a controlled environment if needed.

**Follow-up:** Why is serializing a model not the same as versioning its environment?

### Q7. An MLflow run has a good score but no dataset revision. Can it be reproduced?
**Answer:**
- Not reliably from that record alone.
- Identify the exact dataset from available evidence.
- Record DVC revision or checksum.
- Link it to the run and document any uncertainty.
- Do not claim full reproducibility without traceable inputs.

**Follow-up:** What other lineage metadata should be added?

### Q8. DVC data is tracked but not pushed to remote. What is the impact?
**Answer:**
- The local user may have cached data.
- Teammates or clean environments may not be able to retrieve it.
- Configure and verify the remote, then push.
- Test retrieval from a clean environment.

**Follow-up:** Why should remote credentials not be committed?

## 3. Project questions

### Q9. Describe your Reproducibility package.
**Answer:**
- I logged the selected run in MLflow.
- I linked dataset and code revisions.
- I tracked data with DVC.
- I recorded dependencies, feature schema and model version.
- I created a manifest and documented reproduction steps.

**Follow-up:** How would you test the package on another machine?

### Q10. Does a Docker image alone make an ML project reproducible?
**Answer:**
- No.
- It packages an application environment but does not necessarily identify data, training configuration or source lineage.
- Link the image to code, model artifact and dataset versions.
- Preserve training and evaluation metadata.

**Follow-up:** What if the image contains a model but the model's source run is unknown?

## 4. Rapid revision
- MLflow: experiment and model tracking.
- DVC: data and pipeline versioning.
- Git SHA: source revision.
- Dataset checksum/revision: data identity.
- Manifest: traceability record.
- Model registry: managed model versions.
- Container tag: deployable application version.
