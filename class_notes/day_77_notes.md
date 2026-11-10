# Day 77: Capstone — Problem Definition and Business Metrics

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Beginner to Intermediate  
**Estimated time:** 2–3 hours  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved mini-project / deliverable:** ML project requirements

## 1. Learning objectives
- Convert a business concern into a measurable ML problem.
- Define churn and the prediction unit clearly.
- Identify stakeholders, users and decision actions.
- Select suitable model and business evaluation metrics.
- Write a testable project requirements document.

## 2. Capstone business scenario
A subscription business wants to identify customers who may cancel their service. Customer-success teams need an early warning list so they can investigate and offer appropriate retention support.

The model is a decision-support tool. It does not decide automatically which customers should receive an offer.

## 3. Define the problem before selecting an algorithm
Answer these questions:
- **Who:** Which customer population is in scope?
- **What:** What event counts as churn?
- **When:** At what prediction date is the risk calculated?
- **Horizon:** How far into the future are we predicting?
- **Why:** What action can the business take?
- **How measured:** What defines a useful prediction?

Example problem statement:

"Using information available at a defined monthly snapshot, predict whether an eligible active customer will churn within the next 30 days, so the retention team can prioritize review."

This is an example specification. Confirm the actual churn definition and prediction horizon with the business owner before using real data.

## 4. Define the target carefully
A target definition must avoid ambiguity.

| Item | Example definition |
|---|---|
| Prediction unit | One customer at one snapshot date |
| Target | Churn within the following 30 days |
| Positive class | 1 = churn |
| Negative class | 0 = did not churn |
| Prediction time | Snapshot date |
| Feature availability | Only information known on or before snapshot date |

Do not use cancellation date, post-cancellation support notes or other future information as input features. That would create target leakage.

## 5. Translate business goals into metrics
Different metrics answer different questions.

- **Recall:** Of customers who churned, how many did we identify?
- **Precision:** Of customers flagged, how many actually churned?
- **F1:** Harmonic balance between precision and recall.
- **PR-AUC:** Useful for examining precision-recall behavior, especially when churn is relatively rare.
- **ROC-AUC:** Measures ranking discrimination across thresholds.
- **Confusion matrix:** Shows TP, FP, TN and FN at a chosen threshold.

Business measures may include:
- Retention team capacity: number of customers that can be reviewed.
- Cost of outreach.
- Expected value of a successful retention action.
- Cost of missing a customer likely to churn.

Do not optimize accuracy alone when classes are imbalanced or error costs differ.

## 6. Build a simple decision-cost example
Suppose the business can review 100 flagged customers each week. The model should therefore be assessed at a threshold that produces a manageable review volume.

```python
review_capacity = 100
flagged_customers = 135

if flagged_customers > review_capacity:
    print("Reduce the review list or prioritize by risk score.")
else:
    print("Review capacity is sufficient.")
```

Expected output:
```text
Reduce the review list or prioritize by risk score.
```

The threshold is a business decision informed by validation data, not a value chosen just to maximize a metric.

## 7. Define scope and acceptance criteria
Example requirements:
- Input: approved historical customer snapshot dataset.
- Target: agreed churn definition and prediction horizon.
- Output: churn risk score and class prediction.
- Model: compare a baseline with candidate classifiers.
- Evaluation: precision, recall, F1, ROC-AUC and PR-AUC where appropriate.
- Deployment: API and simple interactive interface.
- Governance: document limitations, privacy and human review.

Acceptance criteria should be measurable. Example: "The model must outperform the agreed baseline on the validation metric and must not violate the agreed review-capacity or segment-performance constraints."

## 8. Industry workflow
Business question → target definition → data availability review → metric selection → project requirements → dataset profiling → model development.

## 9. Approved deliverable — ML project requirements

**Objective:** Produce a concise requirements document for the Customer Churn Prediction and ML Deployment System.

### Tasks
1. Write the business problem in 2–4 sentences.
2. Define customer population, prediction unit and churn event.
3. Choose a prediction horizon and snapshot time.
4. Identify intended users and the action they may take.
5. Define model metrics and at least two business measures.
6. List data fields that may be available at prediction time.
7. List out-of-scope work and assumptions.
8. Write at least five measurable acceptance criteria.
9. Identify privacy, fairness and operational risks.

**Suggested structure**
```text
customer_churn_capstone/
└── docs/
    └── project_requirements.md
```

**Evidence:** Problem statement, target definition, metric rationale, scope, assumptions, acceptance criteria and stakeholder/action description.

**Validation questions**
- Can two people independently label the target using your definition?
- Are all features available at prediction time?
- Does each metric connect to a real decision?
- Is the proposed output actionable?

## 10. AI-assisted implementation guide
| Level | Prompt |
|---|---|
| Hint | "Give me one hint to clarify the churn target and prediction date; do not write my document." |
| Plan | "Break my project requirements deliverable into ordered tasks." |
| Example | "Show a short example of a churn problem statement for a subscription business." |
| Guided implementation | "Review my target definition first, then guide me through the next section." |
| Explanation | "Explain why prediction horizon and snapshot time matter." |
| Debugging | "Check my metric definitions and point out ambiguous or conflicting requirements." |
| Review | "Review my requirements against measurable acceptance criteria and leakage risks." |

## 11. Daily quiz

**Q1. What must be defined before training a churn model?**  
A. Docker image name  
B. Churn event and prediction horizon  
C. UI color  
D. API port  
**Correct: B.** The target and horizon determine what the model is learning.

**Q2. Which metric answers: "Of actual churners, how many did we identify?"**  
A. Precision  
B. Recall  
C. Accuracy  
D. R²  
**Correct: B.** Recall measures the proportion of actual positive cases detected.

**Q3. A retention team can review only 100 customers. What should influence threshold selection?**  
A. Random threshold  
B. Training accuracy only  
C. Review capacity and validation performance  
D. Number of columns  
**Correct: C.** The operating threshold should reflect capacity and error trade-offs.

**Q4. Which feature is a likely leakage risk?**  
A. Customer tenure at snapshot  
B. Monthly charge at snapshot  
C. Cancellation reason recorded after churn  
D. Number of support calls before snapshot  
**Correct: C.** It is only known after the outcome.

**Q5. Why is accuracy alone often insufficient?**  
A. It cannot be calculated  
B. It ignores the distribution of error types and may mislead with imbalance  
C. It is only for regression  
D. It requires Docker  
**Correct: B.** Precision, recall and business costs may matter more.

## 12. Interview preparation

### Q1. How do you convert a business problem into an ML problem?
**Answer:** Define the decision, prediction unit, target event, prediction time and horizon. Identify features available at that time, select evaluation metrics and document constraints. For churn, define exactly what cancellation means and how early the team needs the warning.

**Follow-up:** What happens if the business changes the definition of churn?

### Q2. Why should business metrics accompany model metrics?
**Answer:** A model metric does not directly describe operational value. Review capacity, outreach cost and the value of a retained customer help select a useful operating threshold and determine whether the model is practical.

**Follow-up:** How would you choose between high recall and high precision?

## 13. Completion checklist
- [ ] Define the churn event and prediction horizon.
- [ ] Identify prediction time and eligible features.
- [ ] Select technical and business metrics.
- [ ] Write measurable acceptance criteria.
- [ ] Complete ML project requirements.
- [ ] Explain assumptions and leakage risks.

**Next lesson:** Day 78 — Capstone: Data understanding and quality.
