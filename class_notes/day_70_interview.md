# Day 70: Interview Questions — DVC and Dataset/Model Versioning

**Phase:** 7 — Deployment and MLOps

## 1. Core questions

### Q1. What is DVC?
**Answer:**
- Data Version Control is a tool for managing data and ML pipeline versions.
- It works alongside Git.
- Git stores small metadata/pointer files.
- DVC manages large data and artifact contents through cache and remote storage.

### Q2. Why not store every dataset directly in Git?
**Answer:**
- Large datasets make repositories heavy.
- Binary data changes are not handled efficiently as source-code diffs.
- DVC separates metadata from large content.
- Git remains the source of code and version references.

### Q3. What is a .dvc file?
**Answer:**
- A small metadata file created when DVC tracks a file or directory.
- It identifies the content version and relevant output information.
- It is committed to Git.
- DVC uses it to locate the corresponding content.

### Q4. What is a DVC remote?
**Answer:**
- Storage for DVC-managed data and artifacts.
- It may be local or a supported remote backend.
- Teams use shared remote storage to exchange data versions.
- Access and credentials must be secured.

### Q5. Explain dvc add, push and pull.
**Answer:**
- `dvc add`: starts tracking data and creates metadata.
- `dvc push`: uploads cached data to remote storage.
- `dvc pull`: retrieves data referenced by the current project revision.
- Git commits the pointer files and pipeline definitions.

### Q6. What is dvc repro?
**Answer:**
- It runs pipeline stages when dependencies or outputs require updating.
- It uses the pipeline definition.
- It helps reproduce data processing and training steps.
- Reproducibility still depends on code, environment and external inputs.

## 2. Scenario questions

### Q7. A teammate checks out an older Git commit but sees no dataset. What should they do?
**Answer:**
- Confirm the commit contains the correct .dvc pointer.
- Check remote configuration and access.
- Run `dvc pull`.
- Verify the retrieved file against the expected version.

### Q8. dvc repro says everything is up to date, but the result seems wrong. What do you inspect?
**Answer:**
- Pipeline dependencies and declared outputs.
- Whether all relevant source files are tracked as dependencies.
- Parameters and environment.
- Whether an input changed outside the declared pipeline.
- Validate the output independently.

### Q9. How do you link a model to its dataset?
**Answer:**
- Record the Git commit containing the DVC metadata.
- Record the DVC data revision or checksum.
- Record pipeline and training configuration.
- Attach the lineage to the experiment and model registry version.

### Q10. Can a local DVC remote be used by a team?
**Answer:**
- It can work if the path is shared and accessible to everyone.
- A local path on one person's laptop is not a shared team remote.
- Use approved shared or object storage.
- Configure permissions, backup and credentials.

## 3. Project questions

### Q11. Describe your Versioned ML Dataset project.
**Answer:**
- I initialized Git and DVC.
- I tracked the raw dataset using DVC.
- I committed pointer metadata to Git.
- I configured a remote and pushed data.
- I created a preprocessing stage and reproduced it.
- I documented how to retrieve a selected version.

### Q12. What is the difference between dvc.yaml and dvc.lock?
**Answer:**
- dvc.yaml describes pipeline stages, commands, dependencies, parameters and outputs.
- dvc.lock records resolved state and hashes for a pipeline execution.
- Both support reproducibility.
- They should be versioned in Git.

### Q13. Does DVC replace MLflow?
**Answer:**
- No. They address related but different workflow needs.
- DVC is commonly used for data and pipeline versioning.
- MLflow tracks experiments and model lifecycle information.
- They can be used together with linked version metadata.

## 4. Rapid revision
- Git: source code and metadata history.
- DVC: large data and pipeline versioning.
- .dvc file: data pointer metadata.
- dvc push: upload data.
- dvc pull: retrieve data.
- dvc repro: reproduce pipeline.
- Link data revision to model run.
