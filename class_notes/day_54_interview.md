# Day 54: Interview Questions — Gradient Boosting Fundamentals

**Phase:** 6 — Advanced ML and Practical Applications

## 1. Core questions

### Q1. What is ensemble learning?
**Answer:**
- It combines predictions from multiple models.
- It can improve accuracy, stability or robustness.
- Bagging and boosting are common ensemble approaches.
- Random Forest is a bagging example; Gradient Boosting is a boosting example.

### Q2. What is boosting?
**Answer:**
- Boosting trains simple learners sequentially.
- Each new learner focuses on the loss remaining from the current ensemble.
- Their contributions are combined to produce the final prediction.
- Small decision trees are common base learners.

### Q3. Explain Gradient Boosting step by step.
**Answer:**
1. Start with an initial prediction.
2. Calculate the loss gradient.
3. Fit a small tree to the direction of improvement.
4. Add a scaled contribution from that tree.
5. Repeat for multiple stages.

For squared-error regression, residuals correspond to the negative gradients.

### Q4. Compare bagging and boosting.
**Answer:**
- Bagging trains learners independently and combines them.
- Boosting builds learners sequentially.
- Bagging often reduces variance.
- Boosting focuses on reducing remaining loss.
- Both can overfit if poorly configured.

### Q5. What is learning rate?
**Answer:**
- It controls the contribution of each new learner.
- A smaller value makes updates more gradual.
- Smaller values may need more estimators.
- Tune it using validation data.

### Q6. What is n_estimators?
**Answer:**
- It specifies the number of boosting stages.
- More stages increase model capacity and training time.
- Too many stages may overfit.
- Select it with validation or cross-validation.

## 2. Practical scenarios

### Q7. Training error is low but test error is high. What do you investigate?
**Answer:**
- Check leakage and split design.
- Reduce tree depth or learner complexity.
- Tune learning rate and number of estimators.
- Use cross-validation.
- Compare with a simpler baseline.

### Q8. Random Forest performs better than Gradient Boosting. Which should you deploy?
**Answer:**
- Verify both used the same data split and evaluation process.
- Compare cross-validation stability.
- Consider latency, training cost and maintainability.
- Select the model that meets the business requirements reliably.

### Q9. Why use shallow trees in boosting?
**Answer:**
- They act as simple learners.
- Each stage makes a controlled correction.
- They limit individual tree complexity.
- The ensemble gains capacity by adding stages.

### Q10. Does feature importance prove causation?
**Answer:**
- No. It describes how the fitted model uses features under a chosen importance method.
- Correlated variables can complicate interpretation.
- Causal claims need appropriate causal analysis and assumptions.

## 3. Project questions

### Q11. How would you compare three regression models?
**Answer:**
1. Define the target and primary metric.
2. Inspect the dataset and leakage risks.
3. Create one reproducible split.
4. Train all models on the same training data.
5. Compare MAE, RMSE and R² on held-out data.
6. Consider runtime and complexity.
7. Validate the candidate using cross-validation.

### Q12. What happens if learning rate is too high?
**Answer:**
- Updates can be too aggressive.
- The model may generalize poorly or fail to converge to a useful solution.
- Reduce the learning rate and tune the number of stages.
- Confirm using validation results.

### Q13. What does subsample do?
**Answer:**
- It controls the fraction of training rows used for each boosting stage.
- Values below 1 introduce stochasticity.
- This can improve generalization in some cases.
- It is not guaranteed to help every dataset.

## 4. Rapid revision
- Boosting: sequential learners.
- Gradient: loss-reduction direction.
- Learning rate: contribution size.
- Estimators: number of stages.
- Shallow trees: complexity control.
- Validation: guides tuning and selection.
