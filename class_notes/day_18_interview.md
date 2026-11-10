# Day 18: Interview Questions — ML Workflow, Baselines and Reproducibility

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is an ML workflow?
**Answer:** A repeatable sequence from problem definition and data understanding through splitting, preprocessing, baseline training, evaluation, deployment and monitoring.
**Follow-up:** Why define the problem before selecting an algorithm?  
**Answer:** The target, prediction time and business cost determine which data, evaluation strategy and model are appropriate.

### Q2. What is a baseline model?
**Answer:** A simple reference prediction or existing business rule used to judge whether a more complex model adds value.
**Follow-up:** Why is it important?  
**Answer:** Without a baseline, a complex model may be celebrated even though it performs no better than a simple strategy.

### Q3. What does DummyClassifier(strategy="most_frequent") do?
**Answer:** It predicts the most frequent class learned from training labels, without using feature relationships.
**Follow-up:** Why can its accuracy be high on imbalanced data?  
**Answer:** The majority class dominates the dataset, so predicting it can be correct often while missing the minority class.

### Q4. What does DummyRegressor(strategy="mean") do?
**Answer:** It predicts the mean target value learned from the training data for every input row.
**Follow-up:** Why compare a regression model with it?  
**Answer:** To check whether the model improves over a simple constant prediction.

### Q5. What is reproducibility?
**Answer:** Recording enough information about data, code, split strategy, preprocessing, random seeds, parameters, software versions and metrics so the experiment can be understood and repeated.
**Follow-up:** Does a fixed seed guarantee identical results everywhere?  
**Answer:** No. Hardware, package versions and nondeterministic operations may affect results.

## Scenario questions

### Q6. A complex classifier performs worse than the majority baseline. What do you do?
**Answer:** Check target definition, data quality, class distribution, split strategy, preprocessing and metrics. Determine whether the complex model adds value before increasing complexity.
**Follow-up:** Should you report only accuracy?  
**Answer:** No. Inspect class-level metrics and business costs, especially for minority classes.

### Q7. A model has 98% accuracy but misses almost every fraud case. Is it successful?
**Answer:** Not necessarily. Accuracy hides minority-class failures. Evaluate recall, precision, PR curves and business costs.
**Follow-up:** Why might false negatives be costly?  
**Answer:** They represent fraud that the system fails to flag, potentially causing financial loss.

### Q8. A colleague cannot reproduce your experiment. What do you provide?
**Answer:** Dataset version, code revision, split strategy, random seed, preprocessing, model parameters, package versions and evaluation details.
**Follow-up:** Is random_state enough?  
**Answer:** No. It is only one part of experiment configuration.

### Q9. A model artifact loads but production predictions are wrong. What do you investigate?
**Answer:** Verify artifact version, feature order, preprocessing pipeline, input schema, package versions and whether production data matches training assumptions.
**Follow-up:** Why save the whole pipeline?  
**Answer:** The estimator expects the same feature transformations used during training.

### Q10. The team tried 50 models and chose the highest test score. What is the issue?
**Answer:** The test set influenced model selection and is no longer an independent final evaluation. Use validation or cross-validation for decisions and reserve test data for final assessment.

## Code interpretation

### Q11. What does DummyClassifier most_frequent learn?
**Answer:** The most frequent class in the training labels.
**Follow-up:** Does it learn feature importance?  
**Answer:** No.

### Q12. Why save experiment configuration as JSON?
**Answer:** It records settings in a structured, human-readable format that can be loaded and reviewed.
**Follow-up:** What should never be stored in a public config?  
**Answer:** API keys, passwords, private keys or customer secrets.

### Q13. Why should you load only trusted joblib artifacts?
**Answer:** joblib uses pickle-based serialization mechanisms, and loading an untrusted artifact can execute malicious code.
**Follow-up:** What should be recorded with a model artifact?  
**Answer:** Its source, version, dependencies, training data reference and intended input schema.

**Senior developer principle:** A reproducible experiment can still be a poor experiment. Reproducibility records what happened; validation determines whether the result is useful.
