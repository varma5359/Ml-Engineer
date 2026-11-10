# Day 22: Interview Questions — Multiple Linear Regression

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is multiple linear regression?
**Answer:** It is a supervised regression model that predicts a numerical target using more than one input feature.
**Follow-up:** How is it different from simple linear regression?  
**Answer:** Simple linear regression uses one feature; multiple linear regression uses several.

### Q2. How do you prepare X and y?
**Answer:** X contains the selected input feature columns as a two-dimensional table. y contains the numerical target values, aligned row by row.
**Follow-up:** Why keep the target out of X?  
**Answer:** Including it would allow the model to access the answer during training.

### Q3. How do you interpret a coefficient in multiple regression?
**Answer:** It describes the model's predicted change associated with a one-unit change in that feature while holding the other included features constant.
**Follow-up:** Does this establish causation?  
**Answer:** No. It is a model-based association.

### Q4. Why do coefficient units matter?
**Answer:** Features may use different units and ranges. A coefficient's numerical size cannot be compared meaningfully without considering those units and the feature scaling.

### Q5. What is multicollinearity?
**Answer:** It occurs when predictors contain strong linear relationships with each other. It can make individual coefficients unstable or difficult to interpret.
**Follow-up:** Does it always make prediction impossible?  
**Answer:** No. Its impact depends on the model and objective.

## Scenario questions

### Q6. Adding bedrooms reduces training error but increases test error. What might be happening?
**Answer:** The added feature may cause overfitting, add noise or fail to generalize. Compare feature sets using validation and inspect data quality.
**Follow-up:** Should you always keep more features?  
**Answer:** No. Keep features that improve validated performance or meet a justified business requirement.

### Q7. The distance coefficient is negative. What does it mean?
**Answer:** In this fitted model and dataset, greater distance is associated with a lower predicted target when other included features are held constant. It does not prove causation.
**Follow-up:** Could the sign change in another dataset?  
**Answer:** Yes. Population, feature relationships, units and model specification can change the fitted coefficient.

### Q8. The model performs well overall but poorly for older properties. What do you do?
**Answer:** Compare residuals by property-age group, sample sizes and feature coverage. Check whether the model misses a distinct relationship or lacks relevant inputs.
**Follow-up:** Why inspect segment sizes?  
**Answer:** Small groups can produce unstable metrics and conclusions.

### Q9. The model uses area and total rooms. What relationship should you inspect?
**Answer:** Whether total rooms overlaps with bedrooms or other features and creates redundant information. Inspect feature definitions and correlations.
**Follow-up:** Should one feature be removed automatically?  
**Answer:** No. Validate the decision based on model stability, performance and business meaning.

### Q10. An API request supplies only three of the four expected features. What should happen?
**Answer:** Reject it with a clear schema-validation error. Do not invent a missing feature silently.
**Follow-up:** Why?  
**Answer:** The model was trained with a specific input schema; missing or misordered values can create invalid predictions.

## Code interpretation

### Q11. What does model.coef_ contain?
**Answer:** One learned coefficient for each input feature, in the order used during fitting.
**Follow-up:** Why keep the feature names?  
**Answer:** So each coefficient is interpreted against the correct feature.

### Q12. Why compare simple and multiple regression on the same split?
**Answer:** It makes the comparison fair by evaluating both models on the same held-out observations.
**Follow-up:** Is one split enough for a strong conclusion?  
**Answer:** Not always. Cross-validation or a larger representative dataset can provide a more reliable comparison.

**Senior developer principle:** More input features do not automatically create a better model. Evaluate the value and reliability of each feature.
