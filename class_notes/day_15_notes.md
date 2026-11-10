# Day 15: Correlation and Multicollinearity

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## 1. Learning objectives
By the end, you can define the concept, explain why it is needed, use its basic syntax, interpret results and discuss its risks in a real ML workflow.

## 2. Definition from zero

**Correlation and Multicollinearity:** Correlation measures association between variables. Multicollinearity occurs when predictors contain strong linear relationships with each other.

In simple terms, this is a step used to make data or a model more reliable. First understand what the input represents, what output is expected and what assumptions are being made.

## 3. Why do we need it?

It helps an ML engineer prepare trustworthy data, choose valid methods and avoid errors that otherwise affect model quality. Skipping it can lead to unreliable metrics or poor production decisions.

## 4. Syntax and key operations

Pearson measures linear association; Spearman measures rank-based monotonic association. Use DataFrame.corr() and Variance Inflation Factor as diagnostic tools.

### How to read a syntax example
1. Identify the imported library or class.
2. Identify the input data and its expected type/shape.
3. Identify the method and its parameters.
4. Inspect the output and validate it against the business meaning.

## 5. Small worked example

Use a tiny sample first. Calculate or predict the expected result manually, run the code, and compare. Then change one input and observe what changes. Finally test an empty, missing or unusual input where applicable.

## 6. Real-time industry scenario

**Scenario:** A property dataset includes area in square feet and area in square metres. Both encode nearly the same information, making linear-model coefficients unstable.

### Senior developer approach
- Clarify the business decision and prediction time.
- Inspect representative records and source-system definitions.
- Check assumptions, units, missingness and edge cases.
- Choose the simplest method that solves the requirement.
- Validate on data that was not used to fit the transformation/model.
- Explain uncertainty and monitor after deployment.

### What goes wrong if ignored?

Do not treat correlation as causation or drop a useful feature only because its pairwise correlation is low. Inspect non-linear relationships and domain meaning.

## 7. Common mistakes

| Mistake | Why it is a problem | Better practice |
|---|---|---|
| Applying a method without understanding the data | Output may be technically valid but semantically wrong | Check definitions, types, units and examples |
| Learning preprocessing from all records | Evaluation becomes optimistic | Fit on training folds only |
| Treating every unusual value as invalid | Valid rare cases may be lost | Investigate against domain rules |
| Reporting one metric only | Important failure segments remain hidden | Evaluate segments and error costs |

## 8. Interview-ready summary

Explain the definition, why the step is required, its syntax, assumptions, a business example and the failure mode if it is skipped. Avoid claiming a cause before examining evidence.

## 9. Self-check
1. Define the topic in your own words.
2. When should you use it and when should you avoid it?
3. What assumptions must be checked?
4. What can go wrong if it is skipped?
5. How would you validate the result?

## 10. Related interview practice
See `day_15_interview.md` for scenario questions and model answers.


---

## Detailed Beginner Teaching Expansion

### Beginner explanation
Correlation describes how two variables move together. A positive correlation means larger values of one tend to accompany larger values of the other; negative means one tends to be smaller when the other is larger. A value near zero means little measured association under that correlation method, not necessarily no relationship.

Multicollinearity is a modelling issue where input features are strongly related to one another. It can make coefficient estimates in linear models unstable and hard to interpret.

### Pearson and Spearman
- Pearson correlation: measures linear association and is sensitive to outliers.
- Spearman correlation: measures monotonic association using ranks; useful when relationships are monotonic but not linear or values are ordinal.

### Syntax
```python
corr = df[["area_sqft", "bedrooms", "price"]].corr(method="pearson")
print(corr)
```
A correlation matrix contains values from -1 to +1. The diagonal is 1 because each feature is perfectly correlated with itself.

### Example interpretation
If area and price have correlation +0.82, larger houses tend to have higher prices in this sample. It does not prove increasing area alone causes higher prices; location, condition and market demand may also matter.

### Multicollinearity checks
- Inspect pairwise correlations among predictors.
- Use domain knowledge to identify duplicate or derived fields.
- Variance Inflation Factor (VIF) can help diagnose linear dependence among predictors.
- Compare coefficient stability across samples or model specifications.

High correlation is not an automatic reason to delete a feature. Prediction performance, interpretability and domain meaning all matter.
