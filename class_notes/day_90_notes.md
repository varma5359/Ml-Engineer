# Day 90: Capstone — Final Presentation and Portfolio

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Advanced — Integrated project  
**Estimated time:** 2–3 hours for final review and presentation; implementation spans Days 77–89  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved final deliverable:** End-to-end ML portfolio project

## 1. Learning objectives
- Present the business problem and technical solution clearly.
- Demonstrate the end-to-end ML workflow and deployed interface.
- Explain model evaluation, threshold and limitations.
- Show reproducibility, monitoring and operational controls.
- Prepare a portfolio README and interview walkthrough.

## 2. What should the final project demonstrate?
The capstone is not only a high-scoring model. It should demonstrate that you can move from a business requirement to a controlled ML application.

**End-to-end flow:**

Business problem → data profiling → preprocessing → baseline → candidate models → evaluation → tuning → explainability → API → UI → Docker → experiment/data versioning → monitoring → documentation → presentation.

## 3. Final repository structure
Use a clean structure that reflects your actual implementation:
```text
customer_churn_capstone/
├── app/
│   └── ui.py
├── api/
│   ├── main.py
│   └── schemas.py
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── monitor.py
├── data/
│   └── README.md
├── artifacts/
├── reports/
│   ├── dataset_profile.md
│   ├── baseline_report.md
│   ├── evaluation_report.md
│   ├── model_interpretation.md
│   └── production_readiness.md
├── tests/
├── docs/
│   ├── project_requirements.md
│   ├── architecture.md
│   ├── monitoring_plan.md
│   └── runbook.md
├── Dockerfile.api
├── Dockerfile.ui
├── compose.yaml
├── requirements-api.txt
├── requirements-ui.txt
├── dvc.yaml
├── dvc.lock
└── README.md
```

Do not commit private customer data, secrets or unnecessary large artifacts. Include instructions for obtaining approved data and model artifacts.

## 4. Final presentation structure
A concise 8–12 slide presentation can follow this order:

1. **Title and objective:** project name, author and problem.
2. **Business context:** churn definition, users and decision.
3. **Data:** source, schema, quality and limitations.
4. **Preparation:** split strategy, preprocessing and leakage controls.
5. **Experiments:** baseline and candidate comparison.
6. **Evaluation:** metrics, confusion matrix and threshold.
7. **Explainability:** global and local interpretation.
8. **System architecture:** API, UI and containers.
9. **MLOps:** MLflow, DVC, model version and CI checks.
10. **Monitoring and governance:** drift, delayed labels, risks and rollback.
11. **Demo:** show a valid prediction and safe error handling.
12. **Conclusion:** limitations, next improvements and learning.

## 5. Prepare a reproducible demonstration
Before presenting:
- Start the API and confirm readiness.
- Start the UI and verify API connectivity.
- Submit a valid example input.
- Show prediction, probability and model version.
- Demonstrate one invalid input or safe error.
- Show model/run/data version evidence.
- Show monitoring and runbook documents.

Use synthetic or approved demonstration data. Do not display real customer personal information in a public portfolio.

## 6. Portfolio README checklist
The README should include:
- Project summary and business problem.
- Target definition and prediction horizon.
- Dataset source and usage conditions.
- Installation and run instructions.
- Architecture diagram.
- Model and evaluation summary.
- API endpoints and example request.
- UI screenshot or demonstration link, if available.
- Docker instructions.
- Experiment/data versioning approach.
- Monitoring and governance approach.
- Limitations and future work.

Only claim features that are implemented and tested. Mark planned improvements as future work.

## 7. Final technical walkthrough
Prepare to explain:
- Why the churn target and horizon were selected.
- How data leakage was prevented.
- Why the selected model outperformed or was preferred to the baseline.
- Why the chosen threshold matches review capacity.
- How the API guarantees feature schema compatibility.
- How the model artifact is versioned.
- How the service can be monitored and rolled back.

## 8. Approved final deliverable — End-to-end ML portfolio project

**Objective:** Submit a complete, explainable and reproducible Customer Churn Prediction and ML Deployment System.

### Required evidence
1. ML project requirements document.
2. Dataset profiling report and data dictionary.
3. Reusable preprocessing pipeline.
4. Baseline model and comparison.
5. Candidate model training suite.
6. Evaluation report and threshold rationale.
7. Optimized model and tuning evidence.
8. Model interpretation report.
9. Tested FastAPI inference service.
10. Interactive prediction interface.
11. Containerized API and UI.
12. MLflow experiment and DVC data lineage.
13. Reproducibility manifest.
14. Monitoring plan and production-readiness report.
15. Final presentation and portfolio README.

### Final validation
- Run the documented setup from a clean environment as far as dependencies and approved data access allow.
- Execute automated tests.
- Verify API and UI integration.
- Confirm model and dataset versions.
- Review privacy, security and intended-use statements.
- Demonstrate rollback and incident response steps conceptually or in a controlled test.
- Identify any remaining gaps honestly.

## 9. Final phase assessment — End-to-End ML Capstone

**Scenario:** You are presenting the churn prediction system to a technical panel and a business stakeholder. They want to understand whether the solution is useful, technically sound, maintainable and safe to pilot.

### Assessment tasks
1. Explain the business problem, target and prediction horizon.
2. Demonstrate data profiling and preprocessing.
3. Explain the baseline and candidate model selection.
4. Interpret final evaluation metrics and threshold.
5. Explain at least one global and one local model explanation.
6. Demonstrate the API and UI.
7. Explain Docker deployment and service communication.
8. Show MLflow/DVC/model version lineage.
9. Explain monitoring, delayed labels and retraining decisions.
10. Present security, privacy, governance and rollback controls.
11. Answer technical and business questions.
12. Identify limitations and next steps.

### Assessment rubric

| Area | Weight |
|---|---:|
| Problem definition and business understanding | 10% |
| Data quality and preprocessing | 10% |
| Modeling, evaluation and validation | 20% |
| Explainability and interpretation | 10% |
| API, UI and container integration | 15% |
| Reproducibility and versioning | 10% |
| Monitoring, reliability and governance | 10% |
| Testing, documentation and portfolio quality | 10% |
| Presentation and interview communication | 5% |
| **Total** | **100%** |

### Common failure points
- Target or prediction horizon is ambiguous.
- Features contain future information.
- Preprocessing is fitted before data splitting.
- Model selection uses final test metrics repeatedly.
- Threshold does not match business review capacity.
- API feature schema differs from training.
- Model artifact or dependency versions are not recorded.
- UI shows stale or fabricated results after an API error.
- Monitoring has no owner or response action.
- Documentation claims controls that are not implemented.

## 10. Final interview questions

### Q1. Walk me through your end-to-end ML project.
**Answer structure:**
- Business problem and churn definition.
- Data source, profiling and leakage controls.
- Preprocessing and baseline.
- Candidate training, evaluation and tuning.
- Explainability and threshold selection.
- API, UI and containerization.
- Tracking, versioning, monitoring and governance.
- Limitations and next steps.

**Follow-up:** What was the most important engineering decision you made and why?

### Q2. How would you decide whether this system is ready for a pilot?
**Answer structure:**
- Check model quality against agreed requirements.
- Verify API/UI behavior and automated tests.
- Confirm data/model/code lineage.
- Verify authentication, privacy and security controls.
- Confirm monitoring, support owner and rollback.
- Record open risks and obtain required approval.

**Follow-up:** What conditions would make you stop the pilot?

### Q3. What would you improve if you had another month?
**Answer structure:**
- Improve data representativeness and label quality.
- Validate performance across time and relevant segments.
- Improve probability calibration and threshold policy.
- Strengthen automated testing and CI/CD.
- Improve monitoring and retraining evidence.
- Add controlled deployment and rollback automation.

**Follow-up:** Which improvement would you prioritize first and how would you measure its impact?

## 11. Final completion checklist
- [ ] All Day 77–89 deliverables are integrated.
- [ ] Final project runs using documented steps.
- [ ] Tests and validation evidence are available.
- [ ] API and UI demo works.
- [ ] Data, code, model and container versions are traceable.
- [ ] Monitoring, security and limitations are documented.
- [ ] Portfolio README is complete.
- [ ] Final presentation is delivered.
- [ ] Learner can explain the full project without relying on copied code.

<badge color="success">90-day pathway capstone</badge>

**Pathway outcome:** A portfolio-ready end-to-end ML engineering project, with documented evidence of problem solving, model development, deployment preparation, reproducibility, monitoring and communication.
