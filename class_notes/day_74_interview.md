# Day 74: Interview Questions — Retraining Strategies and Feedback Loops

**Phase:** 7 — Deployment and MLOps

## 1. Core questions

### Q1. What is model retraining?
**Answer:**
- Training a model again with updated or newly approved data.
- It may respond to population or relationship changes.
- Retraining needs data validation and evaluation.
- A new model should not be promoted automatically.

### Q2. Compare scheduled and event-triggered retraining.
**Answer:**
- Scheduled retraining runs at a fixed cadence.
- Event-triggered retraining starts after a defined signal or event.
- Scheduled is predictable but may train unnecessarily.
- Event-triggered is responsive but needs reliable triggers and safeguards.

### Q3. What is a feedback loop?
**Answer:**
- It connects model predictions to later outcomes and decisions.
- Outcomes are collected and validated.
- Monitoring informs whether retraining is needed.
- The new candidate is evaluated before deployment.

### Q4. Why is label maturity important?
**Answer:**
- Some outcomes take time to become known.
- Premature labels can be incomplete or incorrect.
- Evaluation should use a defined outcome window.
- Only mature, validated labels should support retraining decisions.

### Q5. What is selection bias in feedback data?
**Answer:**
- Outcomes are observed for a non-representative subset.
- Example: only customers contacted by a retention campaign have measured intervention outcomes.
- The training data may reflect past model decisions.
- Record selection mechanisms and evaluate the bias.

### Q6. What is champion-challenger evaluation?
**Answer:**
- Champion is the current approved model.
- Challenger is a candidate model.
- They are compared using a controlled evaluation or experiment.
- Promotion requires predefined evidence and approval.

## 2. Scenario questions

### Q7. Drift alert appears, but performance remains stable. Should you retrain?
**Answer:**
- Not automatically.
- Investigate the feature shift and data source.
- Check segment and future performance.
- Retrain only if evidence and policy support it.

### Q8. Recall drops below the approved threshold. What steps follow?
**Answer:**
- Verify labels and metric reliability.
- Check data quality, feature pipeline and model version.
- Investigate segments and business changes.
- Train a candidate if the issue is confirmed.
- Compare candidate with production and follow approval gates.

### Q9. New model improves F1 but harms one important segment. What do you do?
**Answer:**
- Do not rely on aggregate F1 alone.
- Investigate segment sample size and errors.
- Review fairness and business impact.
- Apply governance criteria and seek review.
- Reject or revise the candidate if requirements are not met.

### Q10. How can feedback loops reinforce bias?
**Answer:**
- Model decisions influence which outcomes are observed.
- Future training data then reflects those earlier decisions.
- Monitor selection and exposure.
- Use appropriate experimental or causal evaluation methods.

## 3. Project questions

### Q11. Describe your Retraining Workflow Design.
**Answer:**
- I defined scheduled and event-based triggers.
- I specified label maturity and data-quality checks.
- I designed candidate training and baseline comparison.
- I documented approval, rollout and rollback.
- I included post-release monitoring.

### Q12. Why should candidate and production be compared on the same data?
**Answer:**
- Different evaluation samples can explain score differences.
- The same validation population makes comparison more meaningful.
- Keep test data isolated for final evaluation.
- Use paired or repeated evaluation where appropriate.

### Q13. What should happen after deployment of a retrained model?
**Answer:**
- Record model version and deployment time.
- Monitor service health and prediction behavior.
- Evaluate delayed ground-truth metrics.
- Compare segment performance.
- Roll back if approved safety or quality conditions fail.

## 4. Rapid revision
- Retraining trigger is not automatic promotion.
- Label maturity matters.
- Feedback can be biased.
- Champion: current approved model.
- Challenger: candidate.
- Compare on consistent validation data.
- Preserve rollback and monitor after release.
