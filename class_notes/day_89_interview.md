# Day 89: Interview Questions — Capstone Monitoring and Documentation

**Phase:** 8 — End-to-End ML Capstone

## 1. Core questions

### Q1. What should be monitored after an ML model is deployed?
**Answer:**
- API availability, latency, throughput and errors.
- Input schema, missingness and data drift.
- Prediction distribution and model version.
- Predictive metrics after ground truth is available.
- Segment performance and incident signals.

**Follow-up:** What would you monitor first if no labels are available yet?

### Q2. Why are delayed labels important?
**Answer:**
- The actual churn outcome may occur after the prediction.
- Accuracy, recall and similar supervised metrics require true labels.
- Store prediction identity and timestamp so outcomes can be joined later.
- Do not treat unlabeled predictions as correct or incorrect.

**Follow-up:** How do you handle labels that arrive late or are revised?

### Q3. What makes an alert actionable?
**Answer:**
- A defined metric and calculation window.
- A threshold based on evidence.
- A minimum sample size where relevant.
- Severity, owner and response action.
- Deduplication and escalation rules.

**Follow-up:** How can alert fatigue be reduced?

### Q4. What is a runbook?
**Answer:**
- An operational document for managing a service.
- It includes architecture, owners, checks and diagnostics.
- It describes escalation and recovery steps.
- It should be tested and updated.

**Follow-up:** How would you validate a rollback procedure?

### Q5. What should model limitations documentation include?
**Answer:**
- Intended and prohibited use.
- Training population and data period.
- Target definition and horizon.
- Data and segment limitations.
- Threshold and calibration caveats.
- Human review and reassessment conditions.

**Follow-up:** Why should limitations be visible to end users?

## 2. Scenario questions

### Q6. API latency is normal but churn recall is falling. What do you do?
**Answer:**
- Confirm labels are mature and valid.
- Check feature quality, drift and serving schema.
- Inspect model version and threshold.
- Analyze segment-level performance.
- Follow the model incident and retraining policy.

**Follow-up:** Would you immediately retrain?

### Q7. A drift alert fires on a small sample. What should happen?
**Answer:**
- Check sample size and window.
- Verify data collection and schema.
- Compare with historical variability.
- Avoid treating the score as definitive.
- Reassess after enough data is available.

**Follow-up:** How could minimum sample size be selected?

### Q8. A readiness report says "authentication planned." Can the service be called production-ready?
**Answer:**
- Not if authentication is a required control for the intended release.
- Planned work is not implemented evidence.
- Record the gap and release condition.
- Restrict deployment until required controls pass review.

**Follow-up:** How would you document a temporary pilot exception?

## 3. Project questions

### Q9. Describe your Production-readiness report.
**Answer:**
- I documented the deployed architecture and model version.
- I defined service, data and model monitoring.
- I specified alert windows, thresholds and owners.
- I created a runbook and rollback procedure.
- I listed implemented controls, gaps and open risks.

**Follow-up:** Which evidence would you present to a release reviewer?

### Q10. How do you keep documentation current?
**Answer:**
- Store it with the source repository.
- Link it to model and application versions.
- Update it as part of release changes.
- Assign document owners and review dates.
- Validate operational instructions through exercises.

**Follow-up:** What should trigger a documentation review?

## 4. Rapid revision
- Operational monitoring: service health and performance.
- Model monitoring: drift and predictive quality.
- Delayed labels: outcomes available after prediction.
- Alert: metric, window, threshold, owner and action.
- Runbook: operational diagnosis and recovery.
- Readiness: evidence-based release decision.
