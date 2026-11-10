# Day 51: Interview Questions — Imbalanced Datasets and Resampling

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering

## 1. Core interview questions

### Q1. What is an imbalanced dataset?
**Answer:**
- A dataset where one target class has many more observations than another.
- Examples include fraud detection, rare disease screening and defect detection.
- The minority class may be the most important class.
- A model can achieve high accuracy while failing to detect minority cases.

### Q2. Why is accuracy misleading on imbalanced data?
**Answer:**
- The majority class contributes most observations.
- A model may predict every row as the majority class.
- It can achieve high accuracy but zero recall for the minority class.
- Inspect confusion matrix, class-specific precision and recall, F1 and PR analysis.

### Q3. What is class weighting?
**Answer:**
- It assigns different importance to classes during model training.
- A balanced class-weight setting can increase the influence of minority examples.
- It does not create synthetic observations.
- It is supported by some estimators, not all.

### Q4. What is random oversampling?
**Answer:**
- It increases minority-class representation by duplicating existing minority observations.
- It keeps majority observations.
- It is simple but can increase overfitting because repeated records add no new information.

### Q5. What is random undersampling?
**Answer:**
- It reduces the majority class by removing some majority observations.
- It can reduce training time.
- It may discard useful majority-class information.

### Q6. What is SMOTE?
**Answer:**
- Synthetic Minority Over-sampling Technique.
- It creates synthetic minority samples using neighbouring minority observations.
- It can help some models learn a broader minority region.
- It may generate unrealistic or overlapping observations, so validate it carefully.

## 2. Resampling and validation questions

### Q7. Should you apply SMOTE before or after train-test split?
**Answer:**
- Split the data first.
- Apply SMOTE only to the training data.
- Keep validation and test data at the original class distribution.
- This avoids leakage and preserves realistic evaluation.

### Q8. Why should SMOTE be inside a cross-validation Pipeline?
**Answer:**
- Each training fold should generate its own synthetic samples.
- Validation-fold observations must not influence synthetic sample generation.
- An imbalanced-learn Pipeline applies resampling only during fitting of each training fold.

### Q9. Does balanced training data represent production data?
**Answer:**
- Not necessarily.
- Resampling changes the training class distribution.
- Production may still contain very rare positives.
- Evaluate on representative data and tune the decision threshold for the real operating environment.

### Q10. Which is better: SMOTE or class weights?
**Answer:**
- There is no universal winner.
- Class weights change the training penalty without creating samples.
- SMOTE creates synthetic minority examples.
- Compare both against a baseline using validation data and business-relevant metrics.

### Q11. Why can probability calibration change after resampling?
**Answer:**
- The model is trained on a different class distribution.
- Its score or probability outputs may not reflect the original population prevalence.
- Evaluate calibration and threshold behaviour on representative validation data.

## 3. Scenario-based interview questions

### Scenario 1: Fraud dataset has 99% normal and 1% fraud
**Expected answer:**
- Inspect class distribution and establish a baseline.
- Evaluate fraud precision, recall and Average Precision.
- Compare class weighting and resampling.
- Keep the test set at realistic prevalence.
- Select a threshold based on fraud loss and review capacity.

### Scenario 2: SMOTE gives excellent training results but weak test results
**Expected answer:**
- Check overfitting and synthetic sample quality.
- Confirm resampling was performed only inside training folds.
- Review whether minority examples contain noise or outliers.
- Compare with class weighting and the baseline.

### Scenario 3: Random undersampling improves recall but reduces precision
**Expected answer:**
- The model catches more positives but creates more false alerts.
- Quantify both changes using the confusion matrix.
- Check review capacity and false-negative cost.
- Decide whether the trade-off is acceptable for the business.

### Scenario 4: A team balances the entire dataset before splitting
**Expected answer:**
- This can leak information and create an unrealistic test distribution.
- Split first.
- Apply sampling only within training data or a training-fold Pipeline.
- Evaluate on untouched original test data.

### Scenario 5: SMOTE is applied to categorical labels that were converted to arbitrary integers
**Expected answer:**
- Ordinary SMOTE interpolates numerical feature values and may create invalid intermediate category codes.
- Use an approach suitable for mixed numerical and categorical data, such as an appropriate categorical-aware sampler, or encode and model the data with a suitable workflow.
- Validate that generated samples are meaningful.

## 4. Strong points to remember in interviews

- Imbalance means unequal class representation, not automatically bad data.
- Accuracy can hide failure on the minority class.
- Class weights change training importance without generating samples.
- Oversampling duplicates minority observations.
- Undersampling removes majority observations.
- SMOTE creates synthetic minority observations.
- Resample training data only.
- Put resampling inside the cross-validation Pipeline.
- Keep validation and test distributions realistic.
- Compare strategies using minority-class metrics and business costs.

**Senior developer principle:** Resampling is a training strategy, not an evaluation shortcut. Always test the model on data that represents the real population.
