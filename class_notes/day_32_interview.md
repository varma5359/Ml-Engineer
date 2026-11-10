# Day 32: Interview Questions — Supervised Model Comparison and Model Selection

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is model selection?
**Answer:** It is choosing a suitable model by comparing predictive performance, stability, error costs, interpretability and operational constraints.
**Follow-up:** Is the highest accuracy model always best?  
**Answer:** No. It may fail on an important class or be too slow or difficult to maintain.

### Q2. Why establish a baseline?
**Answer:** It provides a simple reference so the team can determine whether a complex model adds meaningful value.
**Follow-up:** What is a classification baseline?  
**Answer:** A simple strategy such as predicting the most frequent training class.

### Q3. Why use cross-validation?
**Answer:** It evaluates models across multiple partitions and reduces reliance on one split. It supports more stable comparisons.
**Follow-up:** Why use the same folds for all candidates?  
**Answer:** It makes the comparison fair because each model sees the same validation partitions.

### Q4. What is the difference between underfitting and overfitting?
**Answer:** Underfitting occurs when a model is too simple to capture useful patterns. Overfitting occurs when it fits training details but fails to generalize.
**Follow-up:** How can you identify overfitting?  
**Answer:** Compare training and validation performance and inspect the gap.

## Scenario questions

### Q5. Model A has slightly higher mean F1 but much larger fold variation than Model B. What do you consider?
**Answer:** Compare stability, sample size, business impact and the practical size of the difference. A more stable model may be preferable.
**Follow-up:** Is standard deviation the only uncertainty measure?  
**Answer:** No. Consider confidence intervals, sample size and the evaluation design where appropriate.

### Q6. A complex model improves F1 slightly but doubles inference latency. What do you do?
**Answer:** Determine whether the performance gain justifies the latency and operational cost under the service requirements.
**Follow-up:** Who defines acceptable latency?  
**Answer:** Product, engineering and business stakeholders should agree on service requirements.

### Q7. One model has high accuracy but low fraud recall. Should it be selected?
**Answer:** Not without reviewing the cost of missed fraud and appropriate class-level metrics such as recall and precision.
**Follow-up:** Why can accuracy be misleading?  
**Answer:** The majority class can dominate the score.

### Q8. The team chooses the best model using test accuracy after trying many candidates. What is wrong?
**Answer:** The test set has influenced model selection and is no longer an independent final evaluation.
**Follow-up:** What should they use for selection?  
**Answer:** Validation data or cross-validation, then reserve test data for final evaluation.

### Q9. Two models perform similarly. How do you choose?
**Answer:** Compare stability, interpretability, latency, memory, reliability, data requirements and maintenance effort.
**Follow-up:** Is a simpler model often worth considering?  
**Answer:** Yes, if performance is comparable and it is easier to operate or explain.

## Code interpretation

### Q10. What does StratifiedKFold do?
**Answer:** It creates folds while attempting to preserve class proportions in each fold.
**Follow-up:** Can it solve an extremely small minority class?  
**Answer:** No. The data must contain enough examples to support the chosen number of folds.

### Q11. Why use cross_validate with a Pipeline?
**Answer:** Each fold fits preprocessing on its own training partition, reducing leakage during model comparison.

### Q12. Why should the final test set be evaluated only after model selection?
**Answer:** To keep it independent from development decisions and provide a more honest final performance estimate.

**Senior developer principle:** Select a model for the real operating environment, not merely the best number in a notebook.
