# Day 30: Interview Questions — Random Forest and Ensemble Learning

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is ensemble learning?
**Answer:** It combines predictions from multiple models to produce a final prediction that can be more stable or accurate than one individual model.
**Follow-up:** Name common ensemble approaches.  
**Answer:** Bagging, boosting, voting and stacking.

### Q2. What is Random Forest?
**Answer:** It is an ensemble of decision trees trained using bootstrap samples and randomized feature selection. Classification combines votes; regression combines numerical predictions.
**Follow-up:** Why use multiple trees?  
**Answer:** Combining diverse trees can reduce the instability of an individual tree.

### Q3. What is bootstrap sampling?
**Answer:** It creates a sample by selecting observations randomly with replacement. Some rows can appear more than once and others may be left out for a particular tree.
**Follow-up:** What are out-of-bag observations?  
**Answer:** Training rows not selected in a particular tree's bootstrap sample.

### Q4. Why does Random Forest select random feature subsets?
**Answer:** It encourages tree diversity so the ensemble does not simply repeat the same splits in every tree.
**Follow-up:** What can happen if trees are too similar?  
**Answer:** Their errors may be highly correlated, reducing the benefit of combining them.

### Q5. What does n_estimators control?
**Answer:** The number of trees in the forest.
**Follow-up:** Does adding trees always improve the model enough to justify the cost?  
**Answer:** No. Compare performance gains with training, memory and inference cost.

## Scenario questions

### Q6. A single decision tree has unstable performance across splits. What could you try?
**Answer:** Random Forest may improve stability by combining trees trained on different bootstrap samples and feature subsets.
**Follow-up:** Is it guaranteed to outperform the tree?  
**Answer:** No. Compare using the same validation strategy.

### Q7. Increasing n_estimators makes inference slower. What do you consider?
**Answer:** Measure whether the performance gain justifies additional latency and memory. Select a practical forest size.
**Follow-up:** What if the system has strict latency limits?  
**Answer:** Benchmark candidate settings and consider a smaller forest or another estimator.

### Q8. Random Forest ranks customer ID highly. What do you investigate?
**Answer:** Check whether the ID encodes ordering, source-system details or leakage. Verify whether it generalizes and whether it is a legitimate predictive feature.
**Follow-up:** Is impurity importance proof of business value?  
**Answer:** No. It is a model-specific diagnostic.

### Q9. A forest has excellent training accuracy but poor test accuracy. What do you inspect?
**Answer:** Tree depth, leaf constraints, data leakage, split strategy and distribution differences.
**Follow-up:** Can max_depth help?  
**Answer:** It may reduce complexity, but validate the change.

### Q10. Can RandomForestRegressor extrapolate far beyond the training target range like a linear model?
**Answer:** Generally no. Tree predictions are based on target values within leaves and usually remain within the range represented by training targets.
**Follow-up:** Why does that matter?  
**Answer:** Forecasting or pricing tasks may require predictions beyond observed ranges, so model behaviour must match the task.

## Practical questions

### Q11. What is out-of-bag evaluation?
**Answer:** It estimates performance using training observations left out of individual trees' bootstrap samples.
**Follow-up:** Does it replace a final test set?  
**Answer:** No. Keep an independent evaluation set when one is available.

### Q12. What is the limitation of Random Forest feature importance?
**Answer:** Impurity-based importance is not causal, can favour high-cardinality features and can be distorted by correlated predictors. Permutation importance offers another diagnostic.

**Senior developer principle:** A forest is not automatically better because it contains many trees. Evaluate performance, stability and operational cost.
