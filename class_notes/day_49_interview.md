# Day 49: Interview Questions — Feature Selection and Feature Importance

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering

## 1. Core interview questions

### Q1. What is feature selection?
**Answer:**
- Selecting a useful subset of the original input features.
- It can reduce noise, model complexity and data collection cost.
- It may improve generalization, but this must be validated.
- It can make a model easier to explain and maintain.

**Follow-up:** Does selecting fewer features always improve a model?  
**Answer:** No. Removing useful information can reduce performance. Compare the selected-feature model with a baseline using the same validation strategy.

### Q2. What is the difference between feature selection and feature extraction?
**Answer:**
- Feature selection keeps some original columns.
- Feature extraction creates new representations from existing columns.
- Selecting tenure and monthly bill is feature selection.
- PCA components are feature extraction.

### Q3. Explain filter, wrapper and embedded feature selection.
**Answer:**
- **Filter:** Score features using statistical or data-based criteria before fitting a specific model. Examples include variance checks and mutual information.
- **Wrapper:** Repeatedly evaluate subsets using a model. Example: RFE.
- **Embedded:** Selection or importance is part of model fitting. Examples include Lasso and tree-based models.

**Follow-up:** Which is computationally expensive?  
**Answer:** Wrapper methods can be expensive because they repeatedly fit and evaluate models.

### Q4. What is VarianceThreshold?
**Answer:**
- A filter method that removes features with variance below a chosen threshold.
- It can remove constant or near-constant columns.
- A low-variance feature is not automatically useless; rare events can be important.

### Q5. What is SelectKBest?
**Answer:**
- A filter method that scores features using a selected scoring function.
- It keeps the requested number of top-scoring features.
- The score function should match the problem type.
- Fit it using training data only.

### Q6. What is Recursive Feature Elimination?
**Answer:**
- RFE repeatedly fits an estimator and removes the least important features.
- It continues until the requested number of features remains.
- It is model-dependent and may require substantial computation.

## 2. Feature importance questions

### Q7. What is feature importance in a tree-based model?
**Answer:**
- It summarizes how features contributed to decisions in the fitted tree or ensemble.
- It helps identify features the model used.
- It is not a causal explanation.
- Correlation and feature type can affect the importance values.

### Q8. What is permutation importance?
**Answer:**
- Shuffle one feature in evaluation data.
- Measure how much the fitted model's score changes.
- A larger score decrease suggests the model relied more on that feature.
- It depends on the chosen scoring metric and evaluation dataset.

**Follow-up:** What if two features are highly correlated?  
**Answer:** The model may use one as a substitute for the other. Shuffling one feature may have little impact even when the information shared by both is useful.

### Q9. Does high feature importance mean the feature causes the prediction?
**Answer:**
- No.
- Importance describes model reliance or predictive association under a particular method.
- It does not establish causality.
- Causal questions require a different design and assumptions.

### Q10. Why can built-in tree importance be misleading?
**Answer:**
- Correlated features can share importance.
- Impurity-based importance can favour features with many possible split points.
- Values can change with data and model settings.
- Compare with permutation importance and domain knowledge.

## 3. Leakage and model validation questions

### Q11. Why should feature selection happen after the train-test split?
**Answer:**
- Supervised feature selection uses target labels.
- Using the full dataset allows test labels to influence which features are selected.
- Fit selection only on training data.
- Put selection inside a Pipeline during cross-validation.

### Q12. A feature is highly predictive but becomes available only after the customer churns. Can you use it?
**Answer:**
- No, not for a prediction made before churn.
- It is unavailable at the decision time and may leak the outcome.
- Remove it and use only information available when the prediction is required.

### Q13. Feature selection improves training score but reduces validation score. What might be happening?
**Answer:**
- The selection may be overfitting.
- The method may be unstable or the data may be small.
- Recheck leakage and validation design.
- Compare fold-level performance and a baseline.

## 4. Scenario-based interview questions

### Scenario 1: A dataset has 500 features
**Expected answer:**
- Inspect data quality, feature types, redundancy and prediction-time availability.
- Establish a baseline.
- Apply filter methods for initial screening where appropriate.
- Compare model-based selection and permutation importance.
- Validate selected subsets through cross-validation.

### Scenario 2: A feature has high importance but is unavailable at prediction time
**Expected answer:**
- It cannot be used in the production feature set.
- Check whether it is a target leak or post-outcome field.
- Remove it and retrain using features available at the real decision time.

### Scenario 3: Two correlated features have low individual permutation importance
**Expected answer:**
- The model may use one feature as a substitute for the other.
- Evaluate the features together or inspect correlation and grouped importance.
- Do not conclude that both are useless based only on individual permutation scores.

### Scenario 4: A business stakeholder says an important feature causes churn
**Expected answer:**
- Clarify that predictive importance is not causal evidence.
- Explain the method used to calculate importance.
- Recommend a suitable causal analysis or experiment if the business wants to measure impact.

### Scenario 5: A team removes every feature with low variance
**Expected answer:**
- Check whether rare but meaningful signals are being removed.
- Review feature definitions and business relevance.
- Compare model results with and without the features.
- Do not use one automatic threshold without validation.

## 5. Strong points to remember in interviews

- Feature selection keeps original columns; feature extraction creates new representations.
- Filter methods are fast screening tools.
- Wrapper methods evaluate feature subsets with a model.
- Embedded methods perform selection or importance estimation during fitting.
- RFE can be computationally expensive.
- Permutation importance measures score change after shuffling a feature.
- Importance is not causality.
- Correlated features can distort individual importance.
- Fit supervised feature selection inside a validation Pipeline.
- Features must be available at prediction time.

**Senior developer principle:** A feature is useful only when it improves validated predictions and is reliable, available and appropriate for the real decision.
