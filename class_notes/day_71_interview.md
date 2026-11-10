# Day 71: Interview Questions — CI/CD for ML Workflows

**Phase:** 7 — Deployment and MLOps

## 1. Core questions

### Q1. What is CI/CD?
**Answer:**
- CI integrates changes frequently and runs automated checks.
- Continuous Delivery keeps validated changes ready for release.
- Continuous Deployment automatically releases changes after required checks.
- Pipelines reduce manual errors and provide traceable evidence.

### Q2. Why is CI/CD for ML different from normal software CI/CD?
**Answer:**
- ML depends on code, data, features and model artifacts.
- A code change can alter model behavior.
- Data schema and quality must be checked.
- Model quality gates are needed in addition to software tests.

### Q3. What is GitHub Actions?
**Answer:**
- A workflow automation platform integrated with GitHub.
- Workflows are defined in YAML files.
- Events such as push and pull request can trigger jobs.
- Jobs run steps on configured runners.

### Q4. What is a runner?
**Answer:**
- A machine or environment that executes a workflow job.
- It can be hosted by GitHub or self-hosted.
- It runs commands and actions.
- Runner permissions and secrets need careful control.

### Q5. What are quality gates in an ML pipeline?
**Answer:**
- Checks that must pass before a workflow proceeds.
- Examples include tests, schema validation and minimum validation performance.
- Thresholds should be defined from business and risk requirements.
- A failed gate should stop unsafe promotion.

### Q6. Why separate CI from model training?
**Answer:**
- Training may be expensive and slow.
- Fast tests should provide quick feedback.
- Training can run on approved changes or scheduled triggers.
- Separate workflows improve cost and operational control.

## 2. Scenario questions

### Q7. All unit tests pass, but model performance drops. What is missing?
**Answer:**
- Model evaluation and baseline comparison.
- Data validation and feature checks.
- A controlled validation dataset.
- A performance gate before release.

### Q8. A pull request contains a changed data schema. What checks should run?
**Answer:**
- Required columns and types.
- Missing and unexpected columns.
- Target validity and data quality.
- Feature pipeline compatibility.
- Training/evaluation checks when appropriate.

### Q9. CI needs an API token. Where should it be stored?
**Answer:**
- In the platform's protected secrets mechanism or an approved secret manager.
- Never hardcode it in source or workflow files.
- Use least privilege and rotate credentials.
- Avoid printing secrets in logs.

### Q10. A deployment job runs even when tests fail. What do you investigate?
**Answer:**
- Job dependencies and conditions.
- Whether deployment is gated on successful validation.
- Workflow triggers and branch protections.
- Permissions and environment approval settings.

## 3. Project questions

### Q11. Describe your Automated ML Validation project.
**Answer:**
- I added tests for feature logic and data schema.
- I created a GitHub Actions workflow for pull requests and pushes.
- The workflow installs dependencies and runs tests.
- I added model validation gates and recorded pass/fail evidence.

### Q12. What is the difference between a unit test and a model quality check?
**Answer:**
- A unit test verifies a small code component.
- A model quality check evaluates model behavior on controlled data.
- Both are needed.
- A passing unit test does not prove acceptable generalization.

### Q13. How do you prevent a worse model from being deployed?
**Answer:**
- Evaluate candidate and baseline on consistent validation data.
- Define acceptable metric and safety thresholds.
- Require approval for promotion.
- Preserve the current production version and rollback option.

## 4. Rapid revision
- CI: integrate and test.
- Delivery: release-ready.
- Deployment: automatic release.
- Runner: executes workflow.
- Quality gate: blocks unsafe progression.
- ML CI/CD checks code, data, model and serving contracts.
