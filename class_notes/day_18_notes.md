# Day 18: ML Workflow, Baselines and Reproducibility

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## 1. Learning objectives
By the end, you can define the concept, explain why it is needed, use its basic syntax, interpret results and discuss its risks in a real ML workflow.

## 2. Definition from zero

**ML Workflow, Baselines and Reproducibility:** An ML workflow is a repeatable sequence from problem definition through data, training, evaluation and monitoring. A baseline is a simple reference strategy. Reproducibility means being able to recreate results.

In simple terms, this is a step used to make data or a model more reliable. First understand what the input represents, what output is expected and what assumptions are being made.

## 3. Why do we need it?

It helps an ML engineer prepare trustworthy data, choose valid methods and avoid errors that otherwise affect model quality. Skipping it can lead to unreliable metrics or poor production decisions.

## 4. Syntax and key operations

Start with DummyClassifier or DummyRegressor where appropriate. Record dataset version, split strategy, random seed, features, package versions, parameters and metrics.

### How to read a syntax example
1. Identify the imported library or class.
2. Identify the input data and its expected type/shape.
3. Identify the method and its parameters.
4. Inspect the output and validate it against the business meaning.

## 5. Small worked example

Use a tiny sample first. Calculate or predict the expected result manually, run the code, and compare. Then change one input and observe what changes. Finally test an empty, missing or unusual input where applicable.

## 6. Real-time industry scenario

**Scenario:** A delivery company predicts late deliveries. Compare a complex model with a simple majority-class or historical-average baseline before claiming the new model adds value.

### Senior developer approach
- Clarify the business decision and prediction time.
- Inspect representative records and source-system definitions.
- Check assumptions, units, missingness and edge cases.
- Choose the simplest method that solves the requirement.
- Validate on data that was not used to fit the transformation/model.
- Explain uncertainty and monitor after deployment.

### What goes wrong if ignored?

Without a baseline, a model can look impressive while adding no value. Without recorded data and settings, results cannot be reliably reproduced.

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
See `day_18_interview.md` for scenario questions and model answers.


---

## Detailed Beginner Teaching Expansion

### From zero
An ML workflow is the repeatable path from a business problem to a monitored model. A baseline is a simple reference approach used to judge whether a more complex model adds value. Reproducibility means being able to recreate a result using recorded data, code and settings.

### Typical workflow
Problem definition → data understanding → split → preprocessing → baseline → candidate models → evaluation → deployment → monitoring.

### Baseline syntax
```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
pred = baseline.predict(X_test)
print(accuracy_score(y_test, pred))
```
DummyClassifier predicts using a simple strategy, not learned feature relationships. For regression, DummyRegressor provides simple reference predictions.

### What to record
Dataset snapshot/version, target definition, feature list, split strategy, random seed, library versions, preprocessing, model parameters, metrics and code revision.
