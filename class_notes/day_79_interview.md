# Day 79: Interview Questions — Capstone Preprocessing and Feature Engineering

**Phase:** 8 — End-to-End ML Capstone

## 1. Core questions

### Q1. Why use a preprocessing pipeline?
**Answer:**
- It combines transformations and an estimator.
- It applies the same fitted transformations during training and inference.
- It reduces the risk of training-serving mismatch.
- It can be evaluated correctly within cross-validation.

**Follow-up:** What happens if the API applies a different category encoding?

### Q2. What is ColumnTransformer?
**Answer:**
- A scikit-learn utility for applying different transformations to selected columns.
- Numeric features can use imputation and scaling.
- Categorical features can use imputation and one-hot encoding.
- It combines the transformed feature groups.

**Follow-up:** How do you handle columns added after training?

### Q3. Why fit an imputer only on training data?
**Answer:**
- The imputer learns statistics from data.
- Including test data leaks information into the training process.
- Fit on the training set and apply the learned transformation to validation/test data.
- A pipeline helps enforce this behavior.

**Follow-up:** How is this handled during cross-validation?

### Q4. What is OneHotEncoder?
**Answer:**
- It converts categorical values into binary indicator columns.
- It avoids implying an arbitrary numeric order.
- It can handle unseen categories with an appropriate configuration.
- It may increase dimensionality for high-cardinality features.

**Follow-up:** When might target encoding be considered, and what leakage controls are needed?

### Q5. Why save the complete pipeline?
**Answer:**
- It preserves imputers, encoders, scalers and the fitted estimator.
- It reduces mismatch between training and serving.
- It makes inference simpler.
- The artifact must be trusted and versioned.

**Follow-up:** What metadata should accompany the artifact?

## 2. Scenario questions

### Q6. The API receives a new contract category that was not present during training. What should happen?
**Answer:**
- The encoder should have a deliberate unknown-category policy.
- With handle_unknown="ignore", transformation can continue.
- Monitor unknown-category frequency.
- Investigate whether the feature schema or business process changed.

**Follow-up:** Is ignoring an unknown category always safe?

### Q7. Validation score is unusually high after scaling the full dataset before splitting. What is wrong?
**Answer:**
- The scaler learned information from the held-out data.
- This is preprocessing leakage.
- Rebuild the split and fit transformations only inside training folds.
- Re-evaluate with a pipeline.

**Follow-up:** Can leakage happen even without target values being included in X?

### Q8. A customer appears in both train and test snapshots. What is the concern?
**Answer:**
- The model may see highly related records in both sets.
- Evaluation can overstate generalization to unseen customers.
- Use a group-aware split if the deployment goal requires unseen-customer generalization.
- Use a time-aware split if the goal is future snapshots for known and new customers.

**Follow-up:** Which split would you choose for a monthly production forecast?

## 3. Project questions

### Q9. Describe your reusable preprocessing pipeline.
**Answer:**
- I separated target and identifiers.
- I selected an appropriate data split.
- I built numeric and categorical transformations.
- I combined them using ColumnTransformer and Pipeline.
- I fitted only on training data, tested inference and saved the complete pipeline.

**Follow-up:** How do you verify the serving input schema?

### Q10. What is the difference between fit, transform and fit_transform?
**Answer:**
- fit learns transformation parameters from data.
- transform applies the learned transformation.
- fit_transform typically fits and transforms the supplied training data.
- For held-out data, use transform with the already-fitted transformer.

**Follow-up:** Why should fit not be called on production inference data?

## 4. Rapid revision
- Pipeline: ordered transformations plus estimator.
- ColumnTransformer: separate column-specific transformations.
- Imputer: handles missing values.
- OneHotEncoder: categorical indicator features.
- Fit: learn parameters.
- Transform: apply learned parameters.
- Leakage prevention: fit preprocessing on training folds only.
