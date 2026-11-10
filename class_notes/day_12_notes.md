# Day 12: Encoding Categorical Variables and Scaling Numerical Data

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## 1. Learning objectives
By the end, you can define the concept, explain why it is needed, use its basic syntax, interpret results and discuss its risks in a real ML workflow.

## 2. Definition from zero

**Encoding Categorical Variables and Scaling Numerical Data:** Encoding converts categories into numeric representations. Scaling changes numeric feature magnitudes so that they are comparable.

In simple terms, this is a step used to make data or a model more reliable. First understand what the input represents, what output is expected and what assumptions are being made.

## 3. Why do we need it?

It helps an ML engineer prepare trustworthy data, choose valid methods and avoid errors that otherwise affect model quality. Skipping it can lead to unreliable metrics or poor production decisions.

## 4. Syntax and key operations

One-hot encode unordered categories; use ordinal encoding only when a real order exists. StandardScaler uses (x - mean) / standard deviation; MinMaxScaler maps values to a chosen range. Fit transformations on training data only.

### How to read a syntax example
1. Identify the imported library or class.
2. Identify the input data and its expected type/shape.
3. Identify the method and its parameters.
4. Inspect the output and validate it against the business meaning.

## 5. Small worked example

Use a tiny sample first. Calculate or predict the expected result manually, run the code, and compare. Then change one input and observe what changes. Finally test an empty, missing or unusual input where applicable.

## 6. Real-time industry scenario

**Scenario:** A loan dataset contains city, education, age and income. City has no natural ranking; education may have a defined order. KNN may be dominated by income unless numeric features are scaled.

### Senior developer approach
- Clarify the business decision and prediction time.
- Inspect representative records and source-system definitions.
- Check assumptions, units, missingness and edge cases.
- Choose the simplest method that solves the requirement.
- Validate on data that was not used to fit the transformation/model.
- Explain uncertainty and monitor after deployment.

### What goes wrong if ignored?

Arbitrary category numbers invent order, while fitting a scaler on the full dataset leaks test information. Use a ColumnTransformer and Pipeline for consistent training/inference.

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
See `day_12_interview.md` for scenario questions and model answers.


---

## Detailed Beginner Teaching Expansion

### Beginner explanation
Most machine-learning estimators work with numeric feature matrices. Real business data contains text categories such as city, payment type, product group and education. Encoding turns categories into numbers without inventing meaning. Scaling adjusts numeric magnitudes so a large-unit feature does not dominate methods that depend on distance or optimization.

### Encoding choices
| Technique | Example | Use it when | Risk |
|---|---|---|---|
| One-hot encoding | City → city_Hyd, city_Vizag | Categories have no natural order | More columns for high-cardinality fields |
| Ordinal encoding | Low=0, Medium=1, High=2 | There is a real, agreed order | Invents order if used on nominal labels |
| Frequency/target encoding | Replace category with frequency or target statistic | High-cardinality data with careful validation | Target leakage if not fitted correctly |

### Scaling choices
- Standardization: subtract mean and divide by standard deviation. Often useful for linear models with regularization, SVM and KNN.
- Min-max scaling: maps values to a chosen interval, commonly 0–1. Sensitive to extreme values.
- Robust scaling: uses median and interquartile range; useful when extremes distort mean and standard deviation.

### Syntax: one-hot encoding
```python
import pandas as pd

customers = pd.DataFrame({
    "city": ["Hyderabad", "Vizag", "Hyderabad"],
    "income": [30000, 45000, 50000]
})

encoded = pd.get_dummies(customers, columns=["city"], dtype=int)
print(encoded)
```
**Expected output:** columns include `income`, `city_Hyderabad` and `city_Vizag`; category values become 0/1 indicators.

### Syntax: scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- `fit_transform` learns the mean and standard deviation from training data and transforms it.
- `transform` applies those same learned values to test data.
- Never call `fit_transform` separately on the test data.

### How to choose
Tree-based models are usually less sensitive to monotonic feature scaling than distance-based methods. KNN and SVM are usually sensitive. Always check the estimator's requirements and compare using a correct validation design.
