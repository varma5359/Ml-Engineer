# Day 60: Interview Questions — Model Pipelines and Preprocessing Integration

**Phase:** 6 — Advanced ML and Practical Applications

## 1. Core questions

### Q1. What is an ML pipeline?
**Answer:**
- A pipeline chains preprocessing and model steps into one object.
- It applies steps in the correct order.
- It keeps training and inference transformations consistent.
- It simplifies validation and deployment.

### Q2. What is the difference between Pipeline and ColumnTransformer?
**Answer:**
- Pipeline chains sequential transformations and an estimator.
- ColumnTransformer applies different transformations to selected column groups.
- They are often combined for mixed tabular data.

### Q3. Why should preprocessing be fitted only on training data?
**Answer:**
- Learned statistics such as means, medians and scaling parameters must not use test data.
- Fitting on all data leaks information.
- A pipeline inside cross-validation fits preprocessing separately within each training fold.

### Q4. What is OneHotEncoder?
**Answer:**
- It converts categorical values into binary indicator columns.
- It avoids implying a false numeric order for nominal categories.
- `handle_unknown="ignore"` can help handle categories not seen during fit.
- High-cardinality columns can create many output features.

### Q5. Why use SimpleImputer?
**Answer:**
- It fills missing values using a defined strategy.
- Numeric columns may use median or mean.
- Categorical columns may use most frequent value or a constant label.
- The imputation statistic is learned from training data.

### Q6. Why serialize the full pipeline?
**Answer:**
- The model expects the same transformations at inference.
- Saving only the estimator may omit imputation, scaling or encoding.
- A full pipeline reduces training-serving mismatch.
- Artifact versioning and schema validation are still required.

## 2. Practical scenarios

### Q7. Test performance looks suspiciously high after scaling. What do you check?
**Answer:**
- Was the scaler fitted before splitting?
- Were target-derived columns included?
- Was the test set used during feature selection?
- Is the split appropriate for time or groups?
- Rebuild preprocessing inside a pipeline and re-evaluate.

### Q8. A new customer has a contract type unseen during training. What should happen?
**Answer:**
- Define expected behavior for unseen categories.
- OneHotEncoder with `handle_unknown="ignore"` can avoid an error.
- Monitor unknown-category frequency.
- Decide whether new categories require retraining or data review.

### Q9. How do you evaluate preprocessing correctly with cross-validation?
**Answer:**
- Pass the complete pipeline to cross-validation.
- Each fold fits transformations on its training fold.
- The validation fold is transformed using those fitted steps.
- Use group-aware or time-aware folds when the data requires them.

### Q10. Production predictions differ from notebook predictions. What do you investigate?
**Answer:**
- Feature names, order and data types.
- Missing-value behavior.
- Category normalization and unseen categories.
- Model and preprocessing artifact versions.
- Library versions and input schema.
- Compare the same input through both environments.

## 3. Project questions

### Q11. Describe your Reusable ML Pipeline project.
**Answer:**
- I separated numeric and categorical columns.
- I used imputers and scaling for numeric features.
- I used categorical imputation and one-hot encoding for categories.
- I integrated the preprocessing with a classifier.
- I evaluated the complete pipeline with held-out data and cross-validation.
- I tested inference using raw feature records.

### Q12. Does a pipeline automatically prevent all data leakage?
**Answer:**
- No. It protects fitted transformations when used correctly.
- Leaked features, incorrect time splits or target-derived columns can still cause leakage.
- The data and feature-generation process must be audited.

### Q13. How would you test a pipeline?
**Answer:**
- Test expected columns and data types.
- Test missing values and unseen categories.
- Test output shape and valid predictions.
- Test serialization and reload.
- Test that the same input gives consistent predictions under the same artifact and environment.

## 4. Rapid revision
- Pipeline: ordered workflow.
- ColumnTransformer: column-specific transformations.
- Fit on training data only.
- Use full pipeline in cross-validation.
- Save preprocessing and estimator together.
- Validate input schema at inference.
