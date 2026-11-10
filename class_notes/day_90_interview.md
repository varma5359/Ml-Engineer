# Day 90: Interview Questions — Final Capstone Presentation and Portfolio

**Phase:** 8 — End-to-End ML Capstone

## 1. Project walkthrough questions

### Q1. Walk me through your Customer Churn Prediction and ML Deployment System.
**Answer:**
- I started by defining the churn event, prediction horizon and business action.
- I profiled the dataset and checked missingness, duplicates, target balance and leakage risks.
- I created a reusable preprocessing pipeline and baseline.
- I trained candidate models, compared validation performance and selected a model.
- I evaluated the final candidate, selected a threshold and documented explanations.
- I exposed the model through FastAPI and built a Streamlit interface.
- I containerized the services and recorded experiment/data versions.
- I documented monitoring, limitations, security and rollback.

**Follow-up:** Which stage had the greatest effect on the final result?

### Q2. How did you prevent data leakage?
**Answer:**
- I defined which information was available at prediction time.
- I excluded post-outcome fields.
- I split data before fitting preprocessing.
- I placed transformations inside the model pipeline and cross-validation.
- I reserved the final test set for final evaluation.

**Follow-up:** How would you split monthly snapshots from the same customer?

### Q3. Why did you select your final model?
**Answer:**
- I compared it against a simple baseline.
- I used validation metrics aligned with the churn problem.
- I considered precision, recall, PR-AUC, threshold and review capacity.
- I checked segment behavior and operational complexity.
- I then evaluated the selected model on the untouched test set.

**Follow-up:** Would you select a model with the highest PR-AUC if it violated a key segment requirement?

### Q4. How did you choose the prediction threshold?
**Answer:**
- I evaluated threshold behavior using validation scores.
- I considered false-positive and false-negative costs.
- I considered how many customers the retention team can review.
- I selected and documented a threshold that matches the approved operating objective.
- I kept final test data out of threshold tuning.

**Follow-up:** What happens if review capacity changes?

### Q5. How do you explain the model to a business stakeholder?
**Answer:**
- State the prediction target and time horizon.
- Explain what the risk score means and what it does not mean.
- Show relevant model evaluation and limitations.
- Explain important contributing features without claiming causation.
- Describe the human decision that follows the score.

**Follow-up:** How would you measure whether the retention action actually reduces churn?

## 2. System design questions

### Q6. Explain your API and UI architecture.
**Answer:**
- Streamlit collects input fields matching the feature contract.
- It sends a request to the FastAPI endpoint.
- FastAPI validates the request and calls the fitted pipeline.
- The API returns prediction, probability and model version.
- The UI displays the result and handles errors safely.

**Follow-up:** How could another application use the same API?

### Q7. Why did you use Docker Compose?
**Answer:**
- It defines the API and UI services together.
- It configures ports, environment variables and networking.
- The UI can reach the API using service-name DNS.
- It provides a repeatable local integration setup.

**Follow-up:** What changes would be needed for a production orchestrator?

### Q8. How do you reproduce a selected model?
**Answer:**
- MLflow records run parameters, metrics and artifacts.
- DVC identifies data and pipeline versions.
- Git identifies source code.
- A manifest records dependencies, model version and container tag.
- The artifact is linked to its source run and data revision.

**Follow-up:** What if a remote dataset is no longer accessible?

## 3. Monitoring and governance questions

### Q9. What would you monitor after release?
**Answer:**
- API availability, latency and error rate.
- Input schema, missingness and drift.
- Prediction distribution and model version.
- Recall, precision or other metrics when labels mature.
- Segment performance and alert response.

**Follow-up:** How would you monitor the model before churn labels arrive?

### Q10. When would you retrain the model?
**Answer:**
- After investigating a validated data or performance signal.
- When sufficient mature and approved labels are available.
- When the data and target definitions remain consistent.
- After a candidate passes data, quality, segment and operational checks.
- Only after required approval and deployment planning.

**Follow-up:** Why should a drift alert not automatically trigger deployment?

### Q11. How did you address privacy and security?
**Answer:**
- Minimized sensitive fields and controlled logging.
- Kept credentials out of source code and container images.
- Restricted access to data, artifacts and endpoints.
- Used trusted model artifacts and versioned dependencies.
- Documented intended use, limitations and human oversight.

**Follow-up:** What would you do if a secret were committed to Git?

## 4. Behavioral and reflection questions

### Q12. What was the most difficult issue in the project?
**Answer structure:**
- Describe the specific technical issue.
- Explain how you reproduced and investigated it.
- State the evidence that identified the root cause.
- Explain the fix and how you tested it.
- Describe what you would do differently next time.

**Follow-up:** What did you learn from the issue?

### Q13. What are the current limitations of your project?
**Answer:**
- State limitations in dataset coverage and representativeness.
- Explain label and prediction-horizon assumptions.
- Report any weak segments or calibration concerns.
- Identify deployment controls that are still planned rather than implemented.
- Describe next validation steps.

**Follow-up:** Which limitation creates the highest risk?

### Q14. How would you improve the system next?
**Answer:**
- Improve data quality and temporal validation.
- Add more robust monitoring and alert automation.
- Strengthen CI/CD and controlled model promotion.
- Evaluate calibration and intervention impact.
- Improve operational documentation and incident exercises.

**Follow-up:** Which improvement would you prioritize and how would you measure success?

## 5. Rapid revision
- Problem definition: target, prediction unit and horizon.
- Leakage: information unavailable at prediction time.
- Pipeline: consistent transformations and estimator.
- Baseline: reference performance.
- Validation: model and threshold selection.
- Test: final estimate.
- Explainability: model behavior, not causation.
- API: controlled inference interface.
- Docker: packaged application environment.
- MLflow: experiment and artifact tracking.
- DVC: data and pipeline versioning.
- Monitoring: service, data and model behavior.
- Retraining: evidence-based controlled process.
- Portfolio: verifiable evidence and clear communication.
