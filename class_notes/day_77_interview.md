# Day 77: Interview Questions — Capstone Problem Definition and Business Metrics

**Phase:** 8 — End-to-End ML Capstone

## 1. Core questions

### Q1. How do you define a customer churn prediction problem?
**Answer:**
- Identify the customer population and prediction unit.
- Define the churn event precisely.
- Set the prediction snapshot and future horizon.
- Use only features available at prediction time.
- Define the action the business will take from the score.

**Example:** Predict whether an active subscriber will cancel within the next 30 days using information available at the current monthly snapshot.

**Follow-up:** Why is the snapshot date important?

### Q2. What is the difference between a business problem and an ML problem?
**Answer:**
- The business problem describes the desired outcome, such as reducing avoidable churn.
- The ML problem defines a prediction or estimation task that may support that outcome.
- ML is not always necessary; a rule-based process may be sufficient.
- The model's output must connect to an operational decision.

**Follow-up:** When might a rule-based system be preferable?

### Q3. How do you choose evaluation metrics for churn?
**Answer:**
- Inspect class balance and error costs.
- Use precision and recall to understand positive-class behavior.
- Use F1 or PR-AUC where appropriate.
- Use ROC-AUC to assess ranking discrimination.
- Add business constraints such as review capacity and outreach cost.

**Follow-up:** Why might PR-AUC be more informative for a rare churn class?

### Q4. What is target leakage?
**Answer:**
- It occurs when a model uses information unavailable at prediction time or information that reveals the target.
- It can create unrealistically high validation scores.
- Examples include cancellation reason or post-cancellation account status.
- Prevent it by enforcing a time-aware feature availability rule.

**Follow-up:** How would you detect leakage in a dataset?

### Q5. Why is a prediction threshold a business decision?
**Answer:**
- The threshold controls how many customers are flagged.
- Lower thresholds often increase recall but can also increase false positives.
- The team has limited review capacity and outreach cost.
- Select the threshold using validation results and business constraints.

**Follow-up:** What would you do if the team can review only 50 customers per day?

## 2. Scenario questions

### Q6. A manager asks for 99% accuracy. What do you ask?
**Answer:**
- What is the churn rate?
- What are the costs of false positives and false negatives?
- What review capacity exists?
- Which customer action follows a prediction?
- Which metric represents the intended business outcome?

**Follow-up:** What if a model predicts "no churn" for everyone?

### Q7. The dataset has a column named cancellation_reason. Can you use it?
**Answer:**
- First establish when the field is populated.
- If it is recorded after cancellation, it is unavailable at prediction time.
- Exclude it from predictive features.
- It may be useful for post-outcome analysis, not early prediction.

**Follow-up:** What if it is a customer-selected cancellation intention captured before the prediction date?

### Q8. The business wants a list of the "top 100" at-risk customers. Is accuracy the main concern?
**Answer:**
- Ranking quality and precision among the top 100 become important.
- Evaluate recall captured within the review capacity too.
- Check whether the selected customers are actionable.
- Validate the ranking and capacity on unseen data.

**Follow-up:** Which ranking metrics could help?

## 3. Project and communication questions

### Q9. What should an ML project requirements document contain?
**Answer:**
- Problem statement and intended users.
- Target, prediction time and horizon.
- Data scope and feature availability.
- Metrics and acceptance criteria.
- Constraints, assumptions, risks and out-of-scope items.

**Follow-up:** Who should approve the target definition?

### Q10. How would you explain a churn model to a business stakeholder?
**Answer:**
- Explain what is predicted and for which time horizon.
- Describe how the risk score supports prioritization.
- Explain limitations and uncertainty.
- Clarify that a human team decides the appropriate action.
- Report business outcomes, not only model scores.

**Follow-up:** How would you measure whether retention actions actually work?

## 4. Rapid revision
- Target: event the model learns to predict.
- Horizon: future period being predicted.
- Snapshot: time at which prediction is made.
- Leakage: future or target-revealing information in features.
- Recall: fraction of actual churners identified.
- Precision: fraction of flagged customers who churn.
- Threshold: converts a score into a decision.
