# Day 73: Interview Questions — Model Performance Monitoring

**Phase:** 7 — Deployment and MLOps

## 1. Core questions

### Q1. What is model performance monitoring?
**Answer:**
- It measures how the deployed model behaves over time.
- It includes prediction quality when ground truth is available.
- It may include segment metrics and calibration.
- It complements operational monitoring.

### Q2. Compare operational monitoring and model monitoring.
**Answer:**
- Operational monitoring tracks latency, throughput, errors and resources.
- Model monitoring tracks drift, prediction behavior and predictive quality.
- A service can be online but produce poor predictions.
- Both views are required.

### Q3. Why can model metrics be delayed?
**Answer:**
- The actual outcome may occur after the prediction.
- Churn, fraud investigation and loan repayment can take time.
- Predictions must be linked to later outcomes.
- Metrics should only use labels that are mature and correctly defined.

### Q4. Why record model version with each prediction?
**Answer:**
- Different versions may produce different outputs.
- It enables version-specific performance analysis.
- It supports incident investigation and rollback.
- It creates deployment lineage.

### Q5. What is P95 latency?
**Answer:**
- The 95th percentile of observed response time.
- About 95% of measured requests are at or below this value.
- It reveals slow-tail behavior that an average can hide.
- Interpret it over an adequate time window and sample size.

### Q6. Why monitor segment performance?
**Answer:**
- Overall metrics can hide poor performance for a subgroup.
- Segment analysis can identify uneven errors.
- Groups need sufficient sample sizes.
- Use lawful, justified and privacy-protective segmentation.

## 2. Scenario questions

### Q7. API latency is normal but F1 falls. What do you investigate?
**Answer:**
- Data and concept drift.
- Label definition and label quality.
- Feature pipeline changes.
- Model version and threshold.
- Performance by time and segment.
- Whether ground-truth labels are mature.

### Q8. Overall accuracy is stable but recall for a key segment drops. What do you do?
**Answer:**
- Confirm sample size and metric reliability.
- Investigate data and error patterns in that segment.
- Assess business and fairness impact.
- Review threshold and model behavior.
- Escalate according to the model risk policy.

### Q9. Predictions are stored but actual labels cannot be joined. What is missing?
**Answer:**
- A stable prediction or outcome linkage key.
- Prediction timestamp and target window.
- Label source and definition.
- Data lineage and privacy controls.

### Q10. How would you reduce alert noise?
**Answer:**
- Set minimum sample sizes.
- Use rolling windows and validated thresholds.
- Deduplicate repeated alerts.
- Separate warning and critical levels.
- Assign clear ownership and response procedures.

## 3. Project questions

### Q11. Describe your Prediction Performance Monitor.
**Answer:**
- I designed a prediction log with model version and timestamp.
- I joined delayed ground truth.
- I calculated task-specific metrics.
- I reported results by time and segment.
- I also monitored request volume and latency and documented alert thresholds.

### Q12. Why should predictions without labels not enter accuracy calculations?
**Answer:**
- Accuracy requires known actual outcomes.
- Treating missing labels as a class creates false results.
- Keep unlabeled predictions in operational and drift monitoring.
- Add them to supervised metrics only when valid labels arrive.

### Q13. What is the difference between drift and performance degradation?
**Answer:**
- Drift is a change in data or output distribution.
- Performance degradation is worse predictive quality against ground truth.
- Drift can occur without degradation.
- Performance can degrade even when simple feature drift is not detected.

## 4. Rapid revision
- Operational: latency, errors, traffic and resources.
- Model: predictions, drift and quality.
- Delayed labels: evaluate when outcomes mature.
- Model version: trace every prediction.
- P95: tail latency.
- Segment metrics: identify hidden weaknesses.
