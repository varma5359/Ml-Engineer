# Day 71: CI/CD for ML Workflows

**Pathway:** Machine Learning Engineer  
**Phase:** 7 — Deployment and MLOps  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Approved mini-project:** Automated ML Validation

## 1. Learning objectives
- Explain continuous integration and continuous delivery/deployment.
- Identify checks required before an ML change is accepted.
- Create a GitHub Actions workflow for Python tests.
- Add data/model validation steps to a pipeline.
- Explain why CI/CD for ML includes more than application code.

## 2. What is CI/CD?
- **Continuous Integration (CI):** frequently integrate code changes and run automated checks.
- **Continuous Delivery:** keep validated changes ready for release.
- **Continuous Deployment:** automatically release changes when required checks pass.

Typical ML flow:
Code/data change → lint and unit tests → data/schema checks → training or evaluation checks → build artifact/image → approval/deployment.

Not every commit should train a large model. CI can run fast checks, while expensive training and deployment may be separate workflows.

## 3. Why ML CI/CD is different
ML changes can affect:
- Application code.
- Data schema and data quality.
- Feature engineering.
- Model performance.
- Dependencies and model artifacts.
- API contract and serving behavior.

A code test passing does not prove that a model is still useful. Define model-specific quality gates.

## 4. Example project structure
```text
ml_project/
├── app/
├── src/
│   ├── train.py
│   └── validate_data.py
├── tests/
│   ├── test_features.py
│   └── test_api.py
├── requirements.txt
└── .github/workflows/ml-ci.yml
```

## 5. Create a small test
Example `src/features.py`:
```python
def create_features(age, monthly_charge):
    if age <= 0:
        raise ValueError("age must be positive")
    if monthly_charge < 0:
        raise ValueError("monthly_charge cannot be negative")
    return [age, monthly_charge]
```

Example `tests/test_features.py`:
```python
import pytest
from src.features import create_features

def test_create_features():
    assert create_features(30, 50) == [30, 50]

def test_invalid_age():
    with pytest.raises(ValueError):
        create_features(0, 50)
```

Run locally:
```bash
python -m pip install pytest
pytest -q
```

## 6. Add a GitHub Actions workflow
Create `.github/workflows/ml-ci.yml`:

```yaml
name: ML CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest

      - name: Run tests
        run: pytest -q
```

**Explanation:**
- Workflow triggers on pushes and pull requests to main.
- A job runs on a hosted Ubuntu runner.
- Actions check out the repository and configure Python.
- Dependencies are installed.
- Pytest runs and the workflow reports pass/fail.

For production, review and pin third-party actions to trusted immutable commit SHAs according to organizational policy.

## 7. Add ML-specific quality gates
Possible checks:
- Required columns and data types exist.
- Missing-value rate is below an agreed threshold.
- Target labels are valid.
- No duplicate primary keys where uniqueness is required.
- Feature pipeline tests pass.
- Model evaluation metric meets an agreed baseline.
- API contract tests pass.
- Model artifact and schema are compatible.

Do not use arbitrary metric thresholds. Define thresholds with business owners and evaluate on a controlled validation dataset.

## 8. Separate CI from expensive training
A practical workflow can have:
1. Fast CI on each pull request: lint, unit tests, schema tests.
2. Training workflow on approved data/code changes.
3. Evaluation workflow: compare candidate against baseline.
4. Release workflow: publish a versioned artifact/image after approval.

## 9. Approved mini-project — Automated ML Validation

**Scenario:** A team wants to prevent code and data changes from breaking an ML workflow.

**Requirements**
1. Create a Git repository with a small ML project.
2. Add unit tests for feature logic.
3. Add a data-schema validation script.
4. Create a GitHub Actions workflow.
5. Run tests on push and pull request.
6. Add a validation gate for a model metric or baseline comparison.
7. Save workflow evidence and document what each gate protects.

**Suggested structure**
```text
automated_ml_validation/
├── src/
├── tests/
├── reports/
├── requirements.txt
└── .github/workflows/ml-ci.yml
```

**Validation:** Intentionally break one test and confirm CI fails. Restore it and confirm CI passes. Test invalid schema and document the failure. Do not expose secrets in workflow logs.

## 10. Common mistakes
- Running only unit tests and calling the model production-validated.
- Training expensive models on every trivial commit.
- Using the test set as a repeated release-tuning target.
- Storing API keys in workflow YAML.
- Ignoring dependency and action versions.
- Allowing deployment without model and data checks.

## 11. Completion checklist
- [ ] Explain CI, continuous delivery and deployment.
- [ ] Create unit tests.
- [ ] Create a GitHub Actions workflow.
- [ ] Add ML quality gates.
- [ ] Complete Automated ML Validation.

**Next lesson:** Day 72 — Model monitoring and data drift.
