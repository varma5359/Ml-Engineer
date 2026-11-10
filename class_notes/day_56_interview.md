# Day 56: Interview Questions — LightGBM and CatBoost

**Phase:** 6 — Advanced ML and Practical Applications

## 1. Core questions

### Q1. What is LightGBM?
**Answer:**
- A gradient-boosting framework focused on efficient training.
- It uses histogram-based split finding.
- Its common growth strategy is leaf-wise.
- It is frequently used for tabular classification and regression.

**Follow-up:** What is a risk of leaf-wise growth?  
**Answer:** It may create deep, complex trees and overfit, especially on small datasets, unless constraints are used.

### Q2. What is CatBoost?
**Answer:**
- A gradient-boosting library designed to handle categorical features.
- It provides categorical feature processing.
- It uses ordered boosting techniques to address certain training-time prediction-shift issues.
- It is useful for categorical-heavy tabular datasets.

### Q3. What is histogram-based split finding?
**Answer:**
- Continuous values are grouped into bins.
- Candidate splits are evaluated using those bins.
- It can reduce computation and memory requirements.
- Binning can trade some split precision for efficiency.

### Q4. Compare leaf-wise and level-wise growth.
**Answer:**
- Leaf-wise expands the leaf with the greatest estimated loss reduction.
- Level-wise expands nodes by depth level.
- Leaf-wise may reduce loss quickly but create uneven branches.
- Complexity controls are important.

### Q5. Why is CatBoost useful for categorical data?
**Answer:**
- It has built-in categorical feature handling.
- It can reduce manual one-hot encoding in many workflows.
- It uses ordered target-statistic techniques and ordered boosting concepts.
- Categorical columns must still be declared and handled consistently.

### Q6. Compare XGBoost, LightGBM and CatBoost.
**Answer:**
- XGBoost is a broadly used optimized boosting framework.
- LightGBM emphasizes efficient histogram learning and leaf-wise growth.
- CatBoost emphasizes categorical feature handling and ordered boosting.
- Choose based on data, accuracy, runtime, memory and deployment constraints.

## 2. Scenario questions

### Q7. Your dataset contains many categorical columns. Which library would you test?
**Answer:**
- CatBoost is a strong candidate.
- I would still establish a baseline and compare alternatives.
- I would test missing and unseen category behavior.
- I would compare validation metrics and operational requirements.

### Q8. LightGBM training performance is high but validation performance is poor. What do you inspect?
**Answer:**
- num_leaves and max_depth may be too high.
- min_child_samples may be too low.
- Data may be small or noisy.
- Leakage or split design may be wrong.
- Constrain complexity and validate with cross-validation.

### Q9. One model has slightly better F1 but takes much longer to train. What do you recommend?
**Answer:**
- Check whether the difference is stable across folds.
- Estimate business value of the improvement.
- Consider retraining frequency, latency, memory and cost.
- Choose the simplest model that meets requirements.

### Q10. How do you ensure a fair comparison?
**Answer:**
- Same target and evaluation records.
- Same split strategy.
- Leakage-safe preprocessing.
- Consistent tuning budget and validation method.
- Appropriate metrics and runtime.
- Record library versions and parameters.

## 3. Project questions

### Q11. How would you compare the libraries on mixed-type data?
**Answer:**
1. Profile target and features.
2. Separate numerical and categorical columns.
3. Create a leakage-safe split.
4. Use each library's supported categorical workflow.
5. Tune with validation data.
6. Compare metrics and runtime.
7. Test inference with unseen categories.

### Q12. Can these libraries prevent data leakage?
**Answer:**
- No. They cannot fix an invalid target or future-derived features.
- Remove information unavailable at prediction time.
- Split data according to the real deployment scenario.
- Fit preprocessing only on training data.

### Q13. What should be recorded for reproducibility?
**Answer:**
- Dataset version and target definition.
- Split method and random seed.
- Library and Python versions.
- Hyperparameters and metrics.
- Preprocessing and runtime.
- Model artifact and code revision.

## 4. Rapid revision
- LightGBM: histogram-based, commonly leaf-wise.
- CatBoost: categorical handling and ordered boosting.
- XGBoost: general-purpose optimized boosting.
- Compare with consistent data and metrics.
- Defaults are not a fair tuned comparison.
