# Day 11: Outlier Detection and Treatment

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## 1. Learning objectives
By the end, you can define the concept, explain why it is needed, use its basic syntax, interpret results and discuss its risks in a real ML workflow.

## 2. Definition from zero

**Outlier Detection and Treatment:** An outlier is a value unusually far from the pattern of a dataset. It may be a data-entry error, a rare but valid event, or evidence of a new business condition.

In simple terms, this is a step used to make data or a model more reliable. First understand what the input represents, what output is expected and what assumptions are being made.

## 3. Why do we need it?

It helps an ML engineer prepare trustworthy data, choose valid methods and avoid errors that otherwise affect model quality. Skipping it can lead to unreliable metrics or poor production decisions.

## 4. Syntax and key operations

Use domain checks, box plots, IQR fences and z-scores as investigation tools. IQR = Q3 - Q1; lower fence = Q1 - 1.5 × IQR; upper fence = Q3 + 1.5 × IQR.

### How to read a syntax example
1. Identify the imported library or class.
2. Identify the input data and its expected type/shape.
3. Identify the method and its parameters.
4. Inspect the output and validate it against the business meaning.

## 5. Small worked example

Use a tiny sample first. Calculate or predict the expected result manually, run the code, and compare. Then change one input and observe what changes. Finally test an empty, missing or unusual input where applicable.

## 6. Real-time industry scenario

**Scenario:** A ₹9 lakh payment among mostly ₹500–₹10,000 transactions might be fraud, a genuine bulk payment or an extra zero. Verify transaction source and customer context before deleting or capping it.

### Senior developer approach
- Clarify the business decision and prediction time.
- Inspect representative records and source-system definitions.
- Check assumptions, units, missingness and edge cases.
- Choose the simplest method that solves the requirement.
- Validate on data that was not used to fit the transformation/model.
- Explain uncertainty and monitor after deployment.

### What goes wrong if ignored?

Blind deletion can remove the exact fraud cases or premium customers the model must detect. Compare model performance with and without treatment, using thresholds learned from training data only.

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
See `day_11_interview.md` for scenario questions and model answers.


---

## Detailed Beginner Teaching Expansion

### Beginner explanation: what makes a value an outlier?
An outlier is a value that is unusually far from the pattern of the other observations. “Unusual” depends on the feature, population and business context. A ₹5 lakh transaction may be unusual for a small grocery purchase but normal for a wholesale supplier.

**Three important possibilities**
- Data error: an extra zero was typed, a unit was converted incorrectly, or a sensor malfunctioned.
- Rare but valid event: a premium purchase, a major claim or a genuine fraud attempt.
- Process change: a new business channel or pricing policy changed the normal range.

### Detection methods and when to use them
| Method | Meaning | Useful when | Limitation |
|---|---|---|---|
| Domain rule | Check a value against business constraints | Age cannot be negative; quantity must be positive | Rules must be verified with domain owners |
| IQR | Uses the middle 50% of values | Skewed data; quick univariate screening | Can flag valid tail observations |
| Z-score | Measures distance from mean in standard deviations | Roughly bell-shaped numeric data | Sensitive to outliers; distribution assumption matters |
| Visual inspection | Box plot, histogram, scatter plot | Understanding context and multivariate patterns | Requires careful interpretation |

### Syntax with a complete worked example
```python
import pandas as pd

amount = pd.Series([100, 120, 130, 125, 140, 150, 900])

q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outlier_mask = (amount < lower) | (amount > upper)
print(amount[outlier_mask])
```
- `quantile(0.25)` calculates the first quartile.
- `quantile(0.75)` calculates the third quartile.
- IQR is the distance between these quartiles.
- The mask returns True for values outside the fences.

**Expected output:** the output identifies the unusually high value (900) for this sample. Exact quartile calculation can vary with interpolation settings.

### Treatment choices
1. Correct the value if the source system confirms a recording error.
2. Retain it if it is a valid business event.
3. Cap/winsorize only when the business meaning supports limiting extreme influence.
4. Apply a transformation such as log to a positive, heavily skewed feature when appropriate.
5. Use robust methods that are less sensitive to extremes.
6. Keep an indicator flag when the fact that a value was extreme may itself be informative.

**Do not automatically delete outliers.** Compare row counts, distributions and model performance before and after treatment. If a threshold is part of a model pipeline, learn it from training data only.
