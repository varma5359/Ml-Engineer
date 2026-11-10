# Day 73: Model Performance Monitoring

**Pathway:** Machine Learning Engineer  
**Phase:** 7 — Deployment and MLOps  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Approved mini-project:** Prediction Performance Monitor

## 1. Learning objectives
- Explain the difference between operational and predictive monitoring.
- Track model metrics when ground truth becomes available.
- Compare performance by time window and segment.
- Monitor prediction volume, latency and errors.
- Build a simple performance monitoring report.

## 2. What should be monitored?
Two broad groups:

**Operational monitoring**
- Request count and throughput.
- Latency percentiles.
- Error and timeout rates.
- CPU, memory and availability.
- Model/API version.

**Model monitoring**
- Prediction distribution.
- Input quality and drift.
- Ground-truth metrics when labels arrive.
- Performance by segment.
- Calibration and threshold behavior where relevant.

A service can be operationally healthy while its predictions are poor.

## 3. Delayed ground truth
For many use cases, the true outcome is not immediately available. For example, a churn label may only be known after a defined future period.

Flow:
Prediction at time t → store prediction ID and model version → outcome becomes available later → join outcome to prediction → calculate metrics.

Use a stable, privacy-aware identifier and record the prediction timestamp, feature window and target window.

## 4. Create an example prediction log
```python
import pandas as pd

logs = pd.DataFrame({
    "prediction_id": [101, 102, 103, 104, 105, 106],
    "model_version": ["v1", "v1", "v1", "v1", "v1", "v1"],
    "segment": ["A", "A", "B", "B", "A", "B"],
    "prediction": [1, 0, 1, 0, 1, 0],
    "actual": [1, 0, 0, 0, 1, 1],
    "latency_ms": [35, 42, 55, 40, 38, 70]
})
print(logs)
```

This is illustrative monitoring data.

## 5. Calculate model metrics
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = logs["actual"]
y_pred = logs["prediction"]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred, zero_division=0))
print("F1:", f1_score(y_true, y_pred, zero_division=0))
```

Metrics must match the task and business cost. For regression, use measures such as MAE/RMSE. For ranking, use ranking metrics.

## 6. Monitor by segment
```python
segment_metrics = logs.groupby("segment").apply(
    lambda group: pd.Series({
        "rows": len(group),
        "accuracy": accuracy_score(group["actual"], group["prediction"]),
        "recall": recall_score(group["actual"], group["prediction"], zero_division=0)
    }),
    include_groups=False
)
print(segment_metrics)
```

If the installed pandas version does not support `include_groups`, omit it or select columns before groupby.apply. Always check segment sample sizes; tiny groups produce unstable metrics.

## 7. Operational metrics
```python
print("Request count:", len(logs))
print("Mean latency:", logs["latency_ms"].mean())
print("P95 latency:", logs["latency_ms"].quantile(0.95))
print("Max latency:", logs["latency_ms"].max())
```

For real services, collect latency at the API layer and report percentiles over meaningful time windows. Do not use a tiny example sample to set production SLOs.

## 8. Monitor prediction distribution
```python
print(logs["prediction"].value_counts(normalize=True))
print(logs.groupby("model_version")["prediction"].mean())
```

A sudden shift may reflect new inputs, a model change or a serving bug. It needs investigation.

## 9. Alert design
Define:
- Metric and calculation window.
- Minimum sample count.
- Threshold and severity.
- Alert owner and response time.
- Suppression/deduplication rules.
- Investigation and rollback procedure.

Avoid alerts that fire on every small random fluctuation. Validate thresholds against historical behavior.

## 10. Approved mini-project — Prediction Performance Monitor

**Scenario:** A deployed churn model needs a recurring report showing how well predictions perform after outcomes arrive.

**Requirements**
1. Define a prediction log schema.
2. Store model version, timestamp, prediction and traceable outcome reference.
3. Join delayed labels to predictions.
4. Calculate suitable model metrics.
5. Report metrics by time window and segment.
6. Track latency, request volume and errors.
7. Define sample-size-aware alerts.
8. Document label delay and monitoring limitations.

**Suggested structure**
```text
prediction_performance_monitor/
├── data/prediction_log.csv
├── src/metrics.py
├── src/report.py
├── reports/
└── README.md
```

**Validation:** Test missing labels, duplicate IDs, mismatched model versions and small segments. Confirm only predictions with available ground truth enter supervised metrics.

## 11. Common mistakes
- Calculating accuracy before labels arrive.
- Mixing predictions from different model versions.
- Ignoring segment-level performance.
- Alerting on tiny sample sizes.
- Monitoring only average latency.
- Logging personal information without need.

## 12. Completion checklist
- [ ] Explain operational versus model monitoring.
- [ ] Join delayed labels correctly.
- [ ] Calculate and segment performance metrics.
- [ ] Monitor latency and prediction distribution.
- [ ] Complete Prediction Performance Monitor.

**Next lesson:** Day 74 — Retraining strategies and feedback loops.
