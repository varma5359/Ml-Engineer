# Day 21: Interview Questions — Gradient Descent and Learning Rate

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is gradient descent?
**Answer:** It is an iterative optimization method that adjusts model parameters to reduce a selected training objective. The model makes predictions, measures error and updates its parameters repeatedly.
**Follow-up:** Is gradient descent itself a machine-learning model?  
**Answer:** No. It is an optimization method used to train or optimize some models.

### Q2. What is learning rate?
**Answer:** It controls the size of parameter updates during optimization.
**Follow-up:** What happens if it is too small?  
**Answer:** Training may make progress very slowly.

### Q3. What happens if learning rate is too large?
**Answer:** Updates may overshoot a useful solution, cause oscillation or prevent convergence.
**Follow-up:** Is the largest learning rate always fastest?  
**Answer:** No. An unstable learning rate may fail to settle.

### Q4. What is stochastic gradient descent?
**Answer:** It updates model parameters using one training observation at a time.
**Follow-up:** What is mini-batch gradient descent?  
**Answer:** It uses a small group of observations for each update.

### Q5. Why scale features for gradient-based optimization?
**Answer:** Features with very different numerical ranges can make optimization inefficient or unstable. Scaling can make feature magnitudes more comparable.
**Follow-up:** Does every model require scaling?  
**Answer:** No. The need depends on the estimator.

## Practical scenarios

### Q6. Training is very slow with a tiny learning rate. What do you inspect?
**Answer:** Feature scaling, learning-rate schedule, iteration count, stopping tolerance and data quality.
**Follow-up:** Should you immediately increase the rate by a large amount?  
**Answer:** No. Change it carefully and compare convergence and validation results.

### Q7. Training loss oscillates with a high learning rate. What might help?
**Answer:** Try a smaller learning rate or a suitable adaptive schedule, then inspect convergence.
**Follow-up:** How do you know the change helped?  
**Answer:** Compare training behaviour and held-out performance using the same evaluation setup.

### Q8. A model converges on training data but performs poorly on test data. Is this only an optimizer problem?
**Answer:** No. It may be overfitting, distribution mismatch, leakage or an unsuitable split.
**Follow-up:** What should you verify first?  
**Answer:** Data splitting and leakage controls, then training/test error patterns.

### Q9. A convergence warning appears. What do you do?
**Answer:** Inspect feature scales, learning rate, maximum iterations, tolerance, input quality and estimator configuration.
**Follow-up:** Is increasing max_iter always enough?  
**Answer:** No. The underlying cause may be scaling or optimization settings.

### Q10. Does a fixed random seed guarantee identical results on every machine?
**Answer:** No. It improves repeatability but hardware, software versions and nondeterministic operations may still affect results.

## Code interpretation

### Q11. What does eta0 represent in SGDRegressor?
**Answer:** The initial learning-rate value for supported learning-rate schedules.
**Follow-up:** Does eta0 alone define the whole learning-rate behaviour?  
**Answer:** No. The learning_rate setting controls how the rate is used or changed.

### Q12. Why use a Pipeline with StandardScaler and SGDRegressor?
**Answer:** It ensures scaling is fitted on training data and applied consistently before the estimator receives inputs.

**Senior developer principle:** A model that finishes training is not necessarily converged or useful. Inspect optimization behaviour and validation performance.
