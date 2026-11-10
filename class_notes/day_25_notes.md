# Day 25: Logistic Regression and Sigmoid Function

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## 1. Learning objectives
Define the topic, explain its purpose, write its core syntax, interpret outputs, identify assumptions and discuss its limitations in a business setting.

## 2. Definition from zero

**Logistic Regression and Sigmoid Function:** Logistic regression models class probability using a sigmoid transformation of a linear score. Despite its name, it is used for classification.

In simple language, this method helps a computer learn a relationship from examples. Understand what is supplied, what is learned and what result is expected before opening the library documentation.

## 3. Why it is needed

It gives an engineer a systematic way to learn from historical examples rather than hard-coding every possible case. It is useful only when the data, target and evaluation setup match the real decision.

**If ignored or misunderstood:** the model may optimize the wrong thing, produce unstable results or fail on new records.

## 4. Syntax and structure

p = 1/(1+exp(-z)); LogisticRegression(max_iter=1000).fit(X_train,y_train); use predict_proba() for probabilities.

Read every call by identifying the input data, model/transformer, important parameters, output and validation step. Check the official library documentation for version-specific options and defaults.

## 5. Small worked example

Start with a tiny dataset where you can calculate or reason about the expected output. Fit or run the operation, inspect the result, then change one input or parameter. Compare the outputs and explain why they changed. Finally test an edge case.

## 6. Real-world scenario

**Scenario:** A telecom company estimates churn probability. The probability can prioritize retention outreach, but a decision threshold must reflect campaign capacity and error costs.

### Senior engineer's investigation
1. Confirm the business objective and prediction time.
2. Check data definitions, missing values, target quality and class balance.
3. Establish a simple baseline before selecting a complex method.
4. Split data correctly and fit preprocessing only on training data.
5. Evaluate with metrics that reflect business error costs.
6. Review errors by important segment and document limitations.
7. Monitor live inputs and delayed outcomes after deployment.

### What goes wrong if ignored?

A 0.5 threshold is not mandatory. Check calibration, class balance and false-positive/negative costs.

## 7. Common beginner mistakes

| Mistake | Consequence | Better approach |
|---|---|---|
| Choosing an algorithm because it is popular | It may not suit the target or data | Start from problem type and baseline |
| Ignoring feature scale or quality | Training/prediction can be misleading | Inspect and preprocess correctly |
| Evaluating only training data | Overfitting remains hidden | Use validation and final test data |
| Reporting only accuracy | Error costs and class imbalance are hidden | Select relevant metrics and inspect confusion matrix |

## 8. Interview-ready explanation
Explain the definition, why the method is useful, its core syntax, a practical business example, assumptions and limitations. Be prepared to explain what evidence would make you change your approach.

## 9. Self-check
1. What does the method learn or calculate?
2. What are its inputs and outputs?
3. Which parameters affect its behaviour?
4. What are its assumptions and failure modes?
5. How will you validate it on unseen data?
