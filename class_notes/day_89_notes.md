# Day 89: Capstone — Monitoring and Documentation

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved mini-project / deliverable:** Production-readiness report

## 1. Learning objectives
- Define operational and model-quality monitoring for the capstone.
- Specify data-quality, drift and delayed-label metrics.
- Design alerts with windows, thresholds and owners.
- Document model limitations, API behavior and operational procedures.
- Produce a production-readiness report with evidence and open risks.

## 2. What does production readiness mean?
A working demo is not automatically production-ready. A production-readiness review checks whether the system can be operated, secured, monitored and recovered by its owners.

Review:
- Service health and latency.
- Input quality and schema changes.
- Prediction distribution and data drift.
- Predictive performance when outcomes become available.
- Model/data/code version lineage.
- Security, privacy and access.
- Incident handling and rollback.
- Documentation and ownership.

## 3. Define monitoring indicators
Example monitoring plan:

| Area | Indicator | Example alert condition |
|---|---|---|
| API | Error rate | Above agreed threshold over a defined window |
| API | P95 latency | Exceeds service target |
| Input data | Missing feature rate | Unexpected increase |
| Input data | Unknown category rate | Above approved tolerance |
| Model | Prediction distribution | Material shift from reference |
| Model | Recall/PR-AUC | Below approved threshold after labels mature |
| Operations | Model version | Unexpected or unapproved version |

Thresholds must be based on system requirements and historical behavior. The values in a monitoring plan should not be invented simply to fill a table.

## 4. Record prediction events safely
A prediction event may contain:
- Pseudonymous prediction identifier.
- Timestamp.
- Model version.
- Feature schema version.
- Prediction and score.
- Request outcome and latency.
- Reference to a later outcome where permitted.

Avoid logging unnecessary raw personal data. Apply access and retention controls.

## 5. Design a delayed-label workflow
Prediction at time t → outcome matures → join outcome to prediction → validate label → compute performance metrics → report by period and segment.

Do not calculate supervised model metrics for predictions without valid ground-truth labels.

## 6. Create an alert definition
```python
alert_definition = {
    "name": "churn_recall_below_target",
    "metric": "recall",
    "window": "approved_labelled_evaluation_window",
    "minimum_sample_size": "defined_by_model_owner",
    "threshold": "approved_business_threshold",
    "severity": "high",
    "owner": "model_owner",
    "action": "investigate_data_and_model_performance"
}
print(alert_definition)
```

The strings are placeholders for the team's approved monitoring policy. Replace them with actual values and named roles before production use.

## 7. Create an operational runbook
A runbook should include:
1. Service overview and architecture.
2. Service owners and escalation route.
3. Health/readiness endpoints.
4. Dashboard and alert links.
5. Common failure symptoms and diagnostics.
6. Model version and artifact location.
7. Rollback procedure.
8. Data/model incident response.
9. Backup and recovery assumptions.
10. Last review date and document owner.

## 8. Document model limitations
Include:
- Intended use and prohibited use.
- Training population and data period.
- Target definition and prediction horizon.
- Known data-quality limitations.
- Segment performance limitations.
- Calibration and threshold caveats.
- Human review expectations.
- Conditions that require reassessment.

## 9. Approved deliverable — Production-readiness report

**Objective:** Assess whether the integrated churn prediction system has the evidence and controls required for a controlled release.

### Tasks
1. Document architecture, services and model version.
2. Define operational monitoring indicators.
3. Define data-quality and drift monitoring.
4. Define delayed-label performance monitoring.
5. Document thresholds, windows, sample sizes and owners.
6. Create an incident and rollback runbook.
7. Document security, privacy and intended-use limitations.
8. List implemented controls, missing controls and open risks.
9. Make a clear recommendation: ready for controlled pilot, not ready, or requires further evidence.

**Suggested structure**
```text
customer_churn_capstone/
├── docs/production_readiness.md
├── docs/monitoring_plan.md
├── docs/runbook.md
├── docs/model_limitations.md
└── reports/
```

**Validation:** Distinguish controls that are implemented and tested from controls that are only planned. Every alert needs a meaningful trigger and owner. Avoid claiming production readiness without evidence.

## 10. AI-assisted implementation guide
| Level | Prompt |
|---|---|
| Hint | "Give me one hint for separating operational and model-quality monitoring." |
| Plan | "Break my production-readiness report into evidence sections." |
| Example | "Show a concise runbook entry for an unavailable model artifact." |
| Guided implementation | "Help me define the API latency monitoring section first." |
| Explanation | "Explain how delayed labels affect churn performance monitoring." |
| Debugging | "Review this alert rule and identify why it may generate false alarms." |
| Review | "Audit my readiness report for unsupported claims, missing owners and missing rollback evidence." |

## 11. Daily quiz

**Q1. Which is an operational monitoring metric?**  
A. API P95 latency  
B. Churn target definition  
C. Feature importance only  
D. Model training seed  
**Correct: A.** Latency describes service behavior.

**Q2. When can supervised performance metrics be calculated?**  
A. Before the outcome exists  
B. When valid ground-truth labels are available  
C. From model probability alone  
D. From Docker logs  
**Correct: B.** Predictive metrics require actual outcomes.

**Q3. What should an alert definition include?**  
A. Only a metric name  
B. Metric, window, threshold, sample size, owner and action  
C. UI color  
D. Training notebook title  
**Correct: B.** Alerts need context and an actionable response.

**Q4. What should a runbook document?**  
A. Only model accuracy  
B. Diagnostics, owners, escalation and recovery steps  
C. Customer passwords  
D. Every raw training record  
**Correct: B.** It helps operators respond consistently.

**Q5. What is the difference between planned and implemented controls?**  
A. There is no difference  
B. Implemented controls have evidence; planned controls still need delivery and verification  
C. Planned controls are always safer  
D. Implemented controls never need tests  
**Correct: B.** Readiness must be evidence-based.

## 12. Interview preparation

### Q1. What would you monitor after deploying a churn model?
**Answer:** Monitor API availability, latency and errors; input schema, missingness and drift; prediction distributions; model version; and predictive metrics after ground truth matures. Also monitor segment performance and alert response.

**Follow-up:** How do you monitor performance when churn labels take 30 days to arrive?

### Q2. What is a production-readiness review?
**Answer:** It checks whether the system has validated functionality, security, monitoring, ownership, recovery and documentation. It records evidence, open risks and release conditions rather than relying on the fact that a demo runs.

**Follow-up:** Who should approve a controlled pilot?

## 13. Completion checklist
- [ ] Define operational monitoring.
- [ ] Define data and model monitoring.
- [ ] Document delayed-label evaluation.
- [ ] Create alert definitions and owners.
- [ ] Write runbook and rollback steps.
- [ ] Document limitations and open risks.
- [ ] Complete Production-readiness report.

**Next lesson:** Day 90 — Capstone: Final presentation and portfolio.
