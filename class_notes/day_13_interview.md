# Day 13: Interview Questions — Train, Validation and Test Split

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. Why split data into training and test sets?
**Answer:** Training data fits the model. Test data is held back to estimate performance on unseen observations.
**Follow-up:** What if you evaluate only on training data?  
**Answer:** The result may reflect memorization rather than generalization.

### Q2. What is the purpose of validation data?
**Answer:** It supports model comparison, hyperparameter tuning and threshold selection during development.
**Follow-up:** Why keep test data separate?  
**Answer:** Repeatedly using test results for decisions makes the test set part of development.

### Q3. What does random_state do?
**Answer:** It makes a random split repeatable for the same data and implementation.
**Follow-up:** Does it improve model quality?  
**Answer:** No. It supports reproducibility only.

### Q4. What is stratified splitting?
**Answer:** It attempts to preserve target-class proportions across subsets.
**Follow-up:** Can it always be used?  
**Answer:** No. Each class needs enough examples for the requested partitions.

## Scenario questions

### Q5. A churn dataset has 95% non-churn and 5% churn. What should you consider?
**Answer:** Inspect class counts and use stratification if enough examples exist. Evaluate with metrics that reflect minority-class performance.
**Follow-up:** What if there are only two churn examples?  
**Answer:** The desired split may not be feasible; collect more data or redesign evaluation.

### Q6. Each patient has ten records, and the goal is unseen-patient prediction. Which split?
**Answer:** Group-based splitting by patient ID, so one patient's records do not appear in both train and test.
**Follow-up:** What is the risk of random row splitting?  
**Answer:** Patient-specific patterns can leak across subsets and inflate evaluation.

### Q7. You are forecasting next month's sales. Can you randomly shuffle all months?
**Answer:** Not for a realistic future forecast. Use a chronological split.
**Follow-up:** What is rolling validation?  
**Answer:** Repeatedly train on past observations and validate on a subsequent time window.

### Q8. The test set has no positive examples. What do you do?
**Answer:** Inspect class counts and split design. Stratify if feasible, or obtain more positive examples.
**Follow-up:** Should you keep changing seeds until the score improves?  
**Answer:** No. That creates selection bias.

### Q9. The same customer appears in train and test. Is it always wrong?
**Answer:** It depends on the deployment goal. For unseen-customer evaluation, group separation is needed. For future predictions on known customers, a time-aware design may be more suitable.
**Follow-up:** What determines the split?  
**Answer:** The real prediction scenario.

### Q10. The team chooses the best of 50 models based on test score. Is the score independent?
**Answer:** No. Test results influenced model selection. Use validation or cross-validation for decisions and reserve test data for final evaluation.

## Code interpretation

### Q11. What does test_size=0.2 mean?
**Answer:** It reserves approximately 20% of observations for testing, subject to integer rounding.
**Follow-up:** Does it always create exactly 20%?  
**Answer:** Not necessarily for small datasets because row counts are integers.

### Q12. Why pass X and y together to train_test_split?
**Answer:** It keeps feature rows paired with their target values.
**Follow-up:** What if they are split independently?  
**Answer:** Features can become paired with incorrect labels.

### Q13. What does GroupShuffleSplit use?
**Answer:** It partitions groups rather than individual rows, keeping each group in one subset.
**Follow-up:** What should groups contain?  
**Answer:** The entity identifier that defines related observations, such as customer or patient ID.

**Senior developer principle:** A good split is not simply 80/20. It simulates the real prediction situation.
