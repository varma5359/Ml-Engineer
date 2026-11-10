# Day 74: Retraining Strategies and Feedback Loops

**Pathway:** Machine Learning Engineer  
**Phase:** 7 — Deployment and MLOps  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Approved mini-project:** Retraining Workflow Design

## 1. Learning objectives
- Explain when retraining may be required.
- Compare scheduled, event-triggered and manual retraining.
- Design a safe feedback loop.
- Define candidate evaluation and promotion gates.
- Explain rollback and post-release monitoring.

## 2. What is model retraining?
Retraining means fitting a model again using updated or newly approved data. It may be needed because the population, business process, data pipeline or relationship between features and outcomes has changed.

Retraining is not an automatic solution to every alert. First investigate data defects, label changes and serving mismatches.

## 3. Common retraining strategies

| Strategy | Trigger | Example |
|---|---|---|
| Scheduled | Fixed cadence | Monthly retraining |
| Drift-triggered | Validated data/performance alert | Feature distribution changes materially |
| Performance-triggered | Ground-truth metric falls | Recall below approved threshold |
| Event-triggered | Known business/data event | New product or policy introduced |
| Manual | Owner decision | Investigation confirms need |

Choose a cadence based on data freshness, label delay, risk and retraining cost.

## 4. Feedback loop architecture
Prediction → outcome collection → data validation → monitoring → retraining decision → candidate training → evaluation → approval → deployment → continued monitoring.

Important: Feedback data can be biased. For example, outcomes are only observed for customers who received an intervention. Record how decisions and outcomes were generated.

## 5. Define a retraining policy
A policy should state:
- Trigger and monitoring window.
- Minimum data volume and label maturity.
- Eligible training population.
- Data quality and leakage checks.
- Candidate evaluation metrics.
- Baseline comparison and acceptance criteria.
- Approval owner and deployment plan.
- Rollback condition.

## 6. Example trigger evaluation
```python
def should_retrain(
    current_recall,
    minimum_recall,
    drift_alert,
    enough_labeled_rows
):
    if not enough_labeled_rows:
        return False, "Not enough mature labels"
    if current_recall < minimum_recall:
        return True, "Recall below approved threshold"
    if drift_alert:
        return True, "Drift requires investigation"
    return False, "No retraining trigger"
```

**Explanation:** This is a simple policy example. In a real system, a drift alert may trigger investigation rather than immediate training. Thresholds must be approved for the use case.

```python
decision, reason = should_retrain(
    current_recall=0.71,
    minimum_recall=0.75,
    drift_alert=False,
    enough_labeled_rows=True
)
print(decision, reason)
```

## 7. Candidate-versus-production comparison
A retrained candidate should be evaluated against the current production model on the same appropriate validation data.

Consider:
- Primary metric and confidence/variation.
- Segment performance and fairness.
- Calibration and threshold.
- Latency, memory and cost.
- Data quality and feature compatibility.
- Rollback readiness.

Do not replace a production model just because the new model is newer.

## 8. Feedback loop risks
- Delayed or incorrect labels.
- Selection bias from who receives interventions.
- Target leakage from post-outcome features.
- Data poisoning or malicious inputs.
- Repeated training on model-generated labels.
- Population shifts and policy changes.

Use data validation, provenance, access controls and human review where appropriate.

## 9. Approved mini-project — Retraining Workflow Design

**Scenario:** A churn model is monitored continuously, and the team needs a controlled process for deciding when and how to retrain.

**Requirements**
1. Define retraining triggers and monitoring windows.
2. Specify minimum mature-label volume.
3. Define data validation and leakage checks.
4. Design a candidate training workflow.
5. Define baseline comparison and acceptance gates.
6. Document approval and deployment responsibilities.
7. Define rollback and post-release monitoring.
8. Draw a feedback-loop architecture and explain failure paths.

**Suggested structure**
```text
retraining_workflow_design/
├── docs/retraining_policy.md
├── docs/architecture.md
├── config/thresholds.yaml
└── README.md
```

**Validation:** Walk through at least three scenarios: no trigger, drift alert with stable performance, and confirmed performance degradation. Explain why each leads to a different decision.

## 10. Common mistakes
- Retraining immediately on every drift alert.
- Training before labels are mature.
- Comparing candidate and production on different data.
- Ignoring feedback-selection bias.
- Deploying without approval or rollback.
- Forgetting to monitor after retraining.

## 11. Completion checklist
- [ ] Compare retraining strategies.
- [ ] Define trigger and label maturity.
- [ ] Design feedback loop.
- [ ] Define promotion and rollback gates.
- [ ] Complete Retraining Workflow Design.

**Next lesson:** Day 75 — ML security, privacy and governance.
