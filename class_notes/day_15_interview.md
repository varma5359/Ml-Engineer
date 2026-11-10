# Day 15: Interview Questions — Correlation and Multicollinearity

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is correlation?
**Answer:** Correlation describes the direction and strength of association between two variables. It does not prove causation.
**Follow-up:** What are positive and negative correlation?  
**Answer:** Positive variables tend to move in the same direction; negative variables tend to move in opposite directions.

### Q2. Compare Pearson and Spearman correlation.
**Answer:** Pearson measures linear association using original values. Spearman measures rank-based monotonic association.
**Follow-up:** Which may reveal a monotonic but non-linear pattern?  
**Answer:** Spearman.

### Q3. What is multicollinearity?
**Answer:** It occurs when two or more predictor features are strongly linearly related. It can make coefficients in linear models unstable and difficult to interpret.
**Follow-up:** Does it always make prediction impossible?  
**Answer:** No. Its effect depends on the model and objective.

### Q4. What is VIF?
**Answer:** Variance Inflation Factor is a diagnostic that assesses how strongly a predictor can be explained by other predictors in a linear relationship.
**Follow-up:** Is VIF an automatic feature-removal rule?  
**Answer:** No. It is a diagnostic to support investigation.

## Scenario questions

### Q5. Advertising spend and sales have high positive correlation. Can you say advertising caused sales to increase?
**Answer:** No. Seasonality, product availability, promotions or market demand may affect both.
**Follow-up:** Can the feature still be useful for prediction?  
**Answer:** Yes, if available at prediction time and validated on new data. Predictive association is not causation.

### Q6. Two house-area columns use square feet and square metres. What is the concern?
**Answer:** They contain nearly duplicate information and may cause multicollinearity in linear models. Consider keeping the clearer unit.
**Follow-up:** Should you remove one without checking?  
**Answer:** No. Confirm units, business needs and model behaviour.

### Q7. Pearson correlation is near zero but a scatter plot is U-shaped. Explain.
**Answer:** Pearson summarizes linear association. A U-shaped non-linear relationship may still exist.
**Follow-up:** What should you do?  
**Answer:** Inspect non-linear patterns and use modelling or analysis suited to the relationship.

### Q8. A feature has very high correlation with churn. What do you check?
**Answer:** Check its meaning and timestamp. It may be a post-outcome field or direct target proxy.
**Follow-up:** Why can target correlation be misleading?  
**Answer:** Strong association may come from leakage or an invalid feature.

### Q9. VIF is high for two predictors. Should you remove one immediately?
**Answer:** No. Inspect definitions, sample size, model type and coefficient stability, then compare justified alternatives.
**Follow-up:** What if the goal is prediction rather than coefficient interpretation?  
**Answer:** Evaluate predictive performance and stability; high correlation alone does not settle the decision.

## Code interpretation

### Q10. What does df.corr(method="spearman") calculate?
**Answer:** A pairwise Spearman rank-correlation matrix for eligible numeric columns.
**Follow-up:** Does it include categorical text automatically?  
**Answer:** Not as ordinary numeric correlation; categories need appropriate handling.

### Q11. Why visualize before interpreting correlation?
**Answer:** A scatter plot can reveal outliers, clusters and non-linear relationships that a single coefficient hides.

### Q12. What is a correlation heatmap useful for?
**Answer:** It gives a compact view of pairwise correlations among selected features. It should support, not replace, detailed inspection.

**Senior developer principle:** A correlation coefficient is a clue about association, not an explanation of why a pattern exists.
