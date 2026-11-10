# Day 52: Interview Questions — Model Bias, Variance and Overfitting

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering

## 1. Core interview questions

### Q1. What is underfitting?
**Answer:**
- The model has not learned the important patterns in the data.
- It may be too simple or insufficiently trained.
- Training and validation performance are both poor.
- Consider better features, a more suitable model or reduced regularization.

### Q2. What is overfitting?
**Answer:**
- The model learns training-specific details, including noise.
- Training performance is strong but validation or test performance is much weaker.
- The model does not generalize reliably.
- Reduce complexity, use regularization and validate with appropriate splits.

### Q3. What is bias in machine learning?
**Answer:**
- Bias is error associated with overly simple assumptions.
- A high-bias model misses important patterns.
- It is often associated with underfitting.
- A more suitable model or better features may help.

### Q4. What is variance?
**Answer:**
- Variance describes how sensitive a learned model is to changes in the training data.
- High-variance models can change substantially across samples.
- It is often associated with overfitting.
- Regularization, simpler models and more representative data may help.

### Q5. Explain the bias-variance trade-off.
**Answer:**
- Very simple models may miss useful patterns.
- Very flexible models may learn noise.
- The goal is a model that generalizes well to unseen data.
- Compare validation performance and stability rather than maximizing training performance.

### Q6. How do you detect overfitting?
**Answer:**
- Compare training and validation performance.
- Inspect cross-validation fold variation.
- Use learning curves.
- Check for leakage and unrealistic validation splits.
- Evaluate on an untouched test set.

## 2. Practical improvement questions

### Q7. How do you reduce overfitting?
**Answer:**
- Reduce model complexity.
- Apply regularization.
- Use cross-validation for model selection.
- Improve data quality and representativeness.
- Remove irrelevant features when validation supports it.
- Use early stopping for iterative models when appropriate.

### Q8. Does adding more data always solve overfitting?
**Answer:**
- No.
- More representative data can help reduce sensitivity to individual observations.
- Poor labels, leakage and distribution mismatch remain problems.
- Check data quality and whether new data represents future use.

### Q9. What is early stopping?
**Answer:**
- A training control that stops iterative learning when validation performance stops improving.
- It can reduce unnecessary fitting to training-specific noise.
- Use a validation set for the stopping decision.
- Do not repeatedly use the final test set for early stopping.

### Q10. What are learning curves?
**Answer:**
- Plots showing training and validation performance as training data size changes.
- They help identify underfitting, overfitting and whether more data may help.
- Interpret the metric direction correctly.

## 3. Scenario-based interview questions

### Scenario 1: Training score is excellent but test score is poor
**Expected answer:**
- Suspect overfitting or a train-test distribution mismatch.
- Inspect the split, leakage, model complexity and feature quality.
- Compare cross-validation results.
- Apply regularization or simplify the model and re-evaluate.

### Scenario 2: Both training and validation scores are poor
**Expected answer:**
- Suspect underfitting, weak features or data quality issues.
- Check whether the model is too simple.
- Review feature relevance and target quality.
- Compare with a more suitable model.

### Scenario 3: Training and validation performance vary greatly across folds
**Expected answer:**
- Check dataset size and subgroup distribution.
- Inspect whether the split strategy reflects time or group structure.
- Review outliers and rare cases.
- Report variation rather than only the mean score.

### Scenario 4: A deep tree performs perfectly on training data
**Expected answer:**
- Perfect training fit is not evidence of production quality.
- Compare validation performance.
- Inspect tree complexity and leaf sizes.
- Tune with cross-validation and evaluate on untouched test data.

### Scenario 5: A team says a complex model must be better because its training score is higher
**Expected answer:**
- Training score alone does not measure generalization.
- Compare validation and cross-validation results.
- Consider complexity, inference cost and maintenance.
- Choose the model that meets business needs on unseen data.

## 4. Strong points to remember in interviews

- Underfitting: poor training and validation performance.
- Overfitting: strong training performance but weaker validation performance.
- Bias: overly simple assumptions.
- Variance: sensitivity to training data changes.
- Generalization is the real objective.
- Compare training, validation and test results.
- Use regularization and complexity controls thoughtfully.
- Cross-validation reveals performance variation.
- More data helps only when it is relevant and representative.
- Never select a model based only on training performance.

**Senior developer principle:** A model is valuable when it performs reliably on data it has not seen, not when it memorizes the training dataset.
