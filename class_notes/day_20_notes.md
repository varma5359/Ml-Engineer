# Day 20: Cost Functions and Residual Analysis

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## 1. Learning objectives
By the end, you can define the concept, explain why it is needed, use its basic syntax, interpret results and discuss its risks in a real ML workflow.

## 2. Definition from zero

**Cost Functions and Residual Analysis:** A residual is actual value minus predicted value. A loss or cost function quantifies model error.

In simple terms, this is a step used to make data or a model more reliable. First understand what the input represents, what output is expected and what assumptions are being made.

## 3. Why do we need it?

It helps an ML engineer prepare trustworthy data, choose valid methods and avoid errors that otherwise affect model quality. Skipping it can lead to unreliable metrics or poor production decisions.

## 4. Syntax and key operations

MAE is average absolute error; MSE is average squared error; RMSE is the square root of MSE. Use mean_absolute_error and mean_squared_error from sklearn.metrics.

### How to read a syntax example
1. Identify the imported library or class.
2. Identify the input data and its expected type/shape.
3. Identify the method and its parameters.
4. Inspect the output and validate it against the business meaning.

## 5. Small worked example

Use a tiny sample first. Calculate or predict the expected result manually, run the code, and compare. Then change one input and observe what changes. Finally test an empty, missing or unusual input where applicable.

## 6. Real-time industry scenario

**Scenario:** A house-price model performs well for mid-range homes but underpredicts luxury homes. Segment residual analysis exposes a failure hidden by the overall average.

### Senior developer approach
- Clarify the business decision and prediction time.
- Inspect representative records and source-system definitions.
- Check assumptions, units, missingness and edge cases.
- Choose the simplest method that solves the requirement.
- Validate on data that was not used to fit the transformation/model.
- Explain uncertainty and monitor after deployment.

### What goes wrong if ignored?

Lower training loss does not guarantee better unseen predictions. MSE heavily penalizes large errors; choose metrics according to business costs.

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
See `day_20_interview.md` for scenario questions and model answers.
