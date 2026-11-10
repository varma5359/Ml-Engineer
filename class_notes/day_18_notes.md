# Day 18: ML Workflow, Baselines and Reproducibility

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data Preparation and EDA  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Describe a basic end-to-end machine-learning workflow.
- Explain why problem definition comes before algorithm selection.
- Understand the purpose of a baseline model.
- Create a classification baseline using DummyClassifier.
- Create a regression baseline using DummyRegressor.
- Compare a candidate model against a baseline fairly.
- Record data, code, split, preprocessing and model settings.
- Use a fixed random seed where appropriate.
- Save and reload a trained model artifact.
- Explain why reproducibility does not mean every environment will always produce identical results.

## 2. What is an ML workflow?

An **ML workflow** is a repeatable sequence of activities used to define a problem, prepare data, train and evaluate a model, deploy it and monitor its behaviour.

A simple workflow:

```text
Business problem
      |
      v
Target and success metric
      |
      v
Data understanding and validation
      |
      v
Train / validation / test strategy
      |
      v
Preprocessing
      |
      v
Baseline model
      |
      v
Candidate model training
      |
      v
Evaluation and model selection
      |
      v
Deployment and monitoring
```

The sequence matters. Training an algorithm before defining the target and evaluation strategy can waste time or produce misleading results.

## 3. Define the business problem first

Before writing model code, answer:
- What decision should the prediction support?
- Who will use the prediction?
- What exactly should be predicted?
- When should the prediction be made?
- What information is available at that time?
- What errors are costly?
- How will success be measured?
- What is the existing business process or rule to beat?

### Example: Customer churn

A subscription business wants to identify customers who may cancel within the next 30 days.

A useful problem definition includes:
- Observation unit: one customer at a monthly snapshot.
- Prediction time: the snapshot date.
- Target: whether the customer cancels during the following 30 days.
- Features: customer information available on the snapshot date.
- Business action: a retention team may contact selected customers.
- Evaluation: choose metrics that reflect the costs of missed churn and unnecessary outreach.

A model that predicts a different definition of churn is not solving the same business problem.

## 4. Understand the baseline model

A **baseline** is a simple reference used to determine whether a more complex model adds value.

Baselines can include:
- A simple business rule.
- Predicting the majority class.
- Predicting the average or median target.
- A previously deployed model.
- A simple, interpretable statistical model.

A complex model should be compared with a meaningful baseline using the same data split and evaluation method.

## 5. Classification baseline with DummyClassifier

A classification baseline can produce simple predictions without learning complex relationships from the features.

### 5.1 Import the required libraries

### 5.1. Import Pandas and scikit-learn tools

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Classification tools ready")
```

**Expected output**

```text
Classification tools ready
```

**Explanation**

DummyClassifier creates a simple baseline. The metrics summarize predictions. Later lessons explain classification metrics in more detail.



### 5.2 Create a small classification dataset

### 5.2. Prepare customer churn features

```python
X = pd.DataFrame({
    "tenure_months": [2, 4, 6, 8, 12, 18, 24, 30, 36, 48, 3, 9, 15, 27, 42, 50],
    "monthly_spend": [1200, 1500, 1800, 2200, 3000, 3500, 4200, 5000, 5500, 6200, 1300, 2400, 3300, 4700, 5800, 6500]
})

y = pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0])

print(X.head())
print(y.value_counts())
```

**Expected output**

```text
Feature preview and class counts
```

**Explanation**

This is a small synthetic example. Real churn labels must be generated from a documented outcome window.



### 5.3 Split the dataset

### 5.3. Create a stratified train-test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(X_train.shape, X_test.shape)
```

**Expected output**

```text
Training and test shapes
```

**Explanation**

The test set is held back. Stratification preserves approximate class proportions where the sample supports it.



### 5.4 Create a majority-class baseline

### 5.4. Initialize DummyClassifier

```python
baseline = DummyClassifier(strategy="most_frequent")

print(baseline)
```

**Expected output**

```text
DummyClassifier(strategy='most_frequent')
```

**Explanation**

The most_frequent strategy predicts the class that appears most often in the training labels.



### 5.5 Fit the baseline

### 5.5. Train the baseline

```python
baseline.fit(X_train, y_train)

print("Baseline fitted")
```

**Expected output**

```text
Baseline fitted
```

**Explanation**

The dummy model learns the most frequent class from training labels. It does not learn feature relationships.



### 5.6 Generate baseline predictions

### 5.6. Predict on test data

```python
baseline_predictions = baseline.predict(X_test)

print(baseline_predictions)
```

**Expected output**

```text
A sequence containing the most frequent training class
```

**Explanation**

The predictions provide a reference point for evaluating a more complex classifier.



### 5.7 Calculate baseline accuracy

### 5.7. Calculate baseline accuracy

```python
baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print(baseline_accuracy)
```

**Expected output**

```text
An accuracy value between zero and one
```

**Explanation**

This is the proportion of test observations predicted correctly. On imbalanced data, majority-class accuracy may be high while the model misses the minority class.



### 5.8 Inspect the classification report

### 5.8. Review baseline class metrics

```python
print(classification_report(
    y_test,
    baseline_predictions,
    zero_division=0
))
```

**Expected output**

```text
A report showing precision, recall, F1-score and support for each class
```

**Explanation**

This makes it clear whether the majority baseline fails to identify the minority class. Do not rely on accuracy alone for imbalanced problems.



## 6. Regression baseline with DummyRegressor

A regression baseline can predict a simple reference value, such as the mean or median of the training target.

### 6.1 Create a regression dataset

### 6.1. Prepare house-price features and target

```python
X_reg = pd.DataFrame({
    "area": [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
    "bedrooms": [1, 2, 2, 2, 3, 3, 3, 3, 4, 4]
})

y_reg = pd.Series([25, 28, 32, 35, 38, 42, 45, 48, 52, 56])

print(X_reg.head())
print(y_reg.head())
```

**Expected output**

```text
Feature rows and house-price target values
```

**Explanation**

The target values are illustrative price units. Confirm the currency and unit in a real dataset.



### 6.2 Split regression data

### 6.2. Create a regression train-test split

```python
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

print(X_train_r.shape, X_test_r.shape)
```

**Expected output**

```text
Training and test shapes
```

**Explanation**

The test observations are held back for final evaluation. The split is random because this example represents independent observations.



### 6.3 Create a mean baseline

### 6.3. Initialize DummyRegressor

```python
from sklearn.dummy import DummyRegressor

regression_baseline = DummyRegressor(strategy="mean")

print(regression_baseline)
```

**Expected output**

```text
DummyRegressor(strategy='mean')
```

**Explanation**

The model will predict the average target value learned from the training set for every test observation.



### 6.4 Fit the baseline

### 6.4. Train the regression baseline

```python
regression_baseline.fit(X_train_r, y_train_r)

print("Regression baseline fitted")
```

**Expected output**

```text
Regression baseline fitted
```

**Explanation**

The baseline learns the mean target from training data only.



### 6.5 Generate predictions

### 6.5. Predict house prices

```python
baseline_reg_predictions = regression_baseline.predict(X_test_r)

print(baseline_reg_predictions)
```

**Expected output**

```text
A repeated mean prediction for each test observation
```

**Explanation**

A mean baseline does not use the feature values to distinguish houses. It is a reference for candidate models.



### 6.6 Calculate a regression error metric

### 6.6. Calculate mean absolute error

```python
from sklearn.metrics import mean_absolute_error

baseline_mae = mean_absolute_error(
    y_test_r,
    baseline_reg_predictions
)

print(baseline_mae)
```

**Expected output**

```text
A non-negative error value in the target's units
```

**Explanation**

MAE measures the average absolute difference between predictions and actual values. Later lessons cover regression metrics in detail.



## 7. Compare a baseline with a candidate model

A baseline is useful only when the comparison is fair:
- Use the same training/test split.
- Use the same target definition.
- Apply the same preprocessing policy.
- Compare suitable metrics.
- Consider segment-level performance and business costs.

### 7.1 Train a simple candidate classifier

### 7.1. Fit logistic regression

```python
from sklearn.linear_model import LogisticRegression

candidate = LogisticRegression(max_iter=1000)

candidate.fit(X_train, y_train)

print("Candidate fitted")
```

**Expected output**

```text
Candidate fitted
```

**Explanation**

This candidate learns a relationship between features and the target. It is a teaching example, not proof that logistic regression is suitable for every dataset.



### 7.2 Predict with the candidate

### 7.2. Generate candidate predictions

```python
candidate_predictions = candidate.predict(X_test)

print(candidate_predictions)
```

**Expected output**

```text
Predicted labels from logistic regression
```

**Explanation**

The candidate uses feature information rather than always predicting one class.



### 7.3 Calculate candidate accuracy

### 7.3. Calculate candidate accuracy

```python
candidate_accuracy = accuracy_score(
    y_test,
    candidate_predictions
)

print(candidate_accuracy)
```

**Expected output**

```text
An accuracy value between zero and one
```

**Explanation**

Use the same test observations as the baseline. Later, select a metric aligned with the actual business costs.



### 7.4 Compare baseline and candidate

### 7.4. Create a comparison table

```python
comparison = pd.DataFrame({
    "model": ["Majority baseline", "Logistic regression"],
    "accuracy": [baseline_accuracy, candidate_accuracy]
})

print(comparison)
```

**Expected output**

```text
A two-row comparison table
```

**Explanation**

This comparison is only one view. Inspect precision, recall, class counts and business impact before selecting a model.



### 7.5 Understand when a candidate does not beat the baseline

If a complex model performs worse than the baseline:
- Check the target and feature definitions.
- Verify data quality and labels.
- Inspect class imbalance.
- Confirm preprocessing and split strategy.
- Check whether the model is underfitting or overfitting.
- Reconsider whether the chosen metric matches the business goal.
- Avoid increasing complexity without evidence.

## 8. Reproducibility

**Reproducibility** means recording enough information so another person can understand and repeat the experiment under the same or documented conditions.

Record:
- Dataset source and version.
- Data extraction date or snapshot.
- Code revision.
- Train/validation/test strategy.
- Random seeds.
- Preprocessing steps.
- Model class and parameters.
- Library versions.
- Evaluation metrics.
- Known limitations.

A random seed alone does not guarantee complete reproducibility. Software versions, hardware, numerical libraries and nondeterministic operations may also affect results.

## 9. Set random seeds where appropriate

### 9.1 Create a NumPy random generator

### 9.1. Use default_rng with a fixed seed

```python
import numpy as np

rng = np.random.default_rng(42)

sample_values = rng.integers(1, 100, size=5)

print(sample_values)
```

**Expected output**

```text
A repeatable sequence of five integers
```

**Explanation**

Using a fixed seed makes this generator repeatable for the same implementation and calls.



### 9.2 Recreate the generator with the same seed

### 9.2. Repeat the same random generation

```python
rng_again = np.random.default_rng(42)

sample_again = rng_again.integers(1, 100, size=5)

print(sample_again)
```

**Expected output**

```text
The same five integers as the previous cell
```

**Explanation**

A new generator initialized with the same seed and same calls reproduces the sequence.



### 9.3 Understand why random_state appears in splitting

### 9.3. Use a fixed split seed

```python
first_split = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

second_split = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(first_split[1].index.equals(second_split[1].index))
```

**Expected output**

```text
True
```

**Explanation**

The same random state gives a repeatable partition for the same input data and library behaviour.



## 10. Record experiment configuration

### 10.1 Create a configuration dictionary

### 10.1. Record model and split settings

```python
experiment_config = {
    "experiment_name": "customer_churn_baseline",
    "model": "LogisticRegression",
    "test_size": 0.25,
    "random_state": 42,
    "max_iter": 1000,
    "target": "churn"
}

print(experiment_config)
```

**Expected output**

```text
A dictionary containing the experiment settings
```

**Explanation**

A configuration makes important choices explicit instead of hiding them in notebook cells.



### 10.2 Save configuration to JSON

### 10.2. Write a JSON configuration file

```python
import json

with open("experiment_config.json", "w") as file:
    json.dump(experiment_config, file, indent=4)

print("Configuration saved")
```

**Expected output**

```text
Configuration saved
```

**Explanation**

JSON is a human-readable format for structured configuration. Store only non-secret settings; never put API keys or passwords in this file.



### 10.3 Read the configuration again

### 10.3. Load the saved configuration

```python
with open("experiment_config.json", "r") as file:
    loaded_config = json.load(file)

print(loaded_config)
```

**Expected output**

```text
The configuration dictionary read from the JSON file
```

**Explanation**

Loading the file confirms that the saved settings can be recovered.



## 11. Record package versions

### 11.1 Inspect Python and package versions

### 11.1. Record environment versions

```python
import sys
import pandas as pd
import numpy as np
import sklearn

environment_info = {
    "python": sys.version,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "scikit_learn": sklearn.__version__
}

print(environment_info)
```

**Expected output**

```text
A dictionary containing Python and package versions
```

**Explanation**

Different versions can change defaults or numerical behaviour. Record them with important experiments.



### 11.2 Save environment information

### 11.2. Write environment information

```python
with open("environment_info.json", "w") as file:
    json.dump(environment_info, file, indent=4)

print("Environment information saved")
```

**Expected output**

```text
Environment information saved
```

**Explanation**

This file supports debugging and reproducibility. A full project should also maintain an explicit dependency file.



## 12. Save and reload a model artifact

A model artifact is a saved representation of a trained model or pipeline. It allows an application to load the trained object without training from scratch.

Only load trusted artifacts. Pickle-based formats can execute code during loading and should not be accepted from untrusted sources.

### 12.1 Fit a final teaching model

### 12.1. Fit the candidate on training data

```python
candidate.fit(X_train, y_train)

print("Model ready to save")
```

**Expected output**

```text
Model ready to save
```

**Explanation**

The model is fitted using training data. Do not fit on test data.



### 12.2 Save the model with joblib

### 12.2. Persist the trained model

```python
import joblib

joblib.dump(candidate, "churn_model.joblib")

print("Model artifact saved")
```

**Expected output**

```text
Model artifact saved
```

**Explanation**

joblib is commonly used to persist scikit-learn models. Store artifacts in controlled locations and record their source and versions.



### 12.3 Load the saved model

### 12.3. Load the saved model artifact

```python
loaded_model = joblib.load("churn_model.joblib")

print(type(loaded_model))
```

**Expected output**

```text
The LogisticRegression class type
```

**Explanation**

The saved estimator has been loaded. Use only artifacts from trusted sources.



### 12.4 Compare predictions before and after loading

### 12.4. Validate the reloaded model

```python
pred_before = candidate.predict(X_test)
pred_after = loaded_model.predict(X_test)

print(np.array_equal(pred_before, pred_after))
```

**Expected output**

```text
True
```

**Explanation**

For this simple case, predictions from the original and reloaded estimator should match. This is a useful artifact validation check.



## 13. Reproducibility versus repeatability

| Term | Practical meaning |
|---|---|
| Repeatable run | The same setup and inputs produce the same result again |
| Reproducible experiment | Another person can reconstruct the experiment from recorded data, code and settings |
| Reliable model | The model continues to perform acceptably on relevant new data |
| Generalization | The model performs well on observations not used during fitting |

A reproducible experiment can still produce a poor model. Reproducibility helps verify what happened; it does not guarantee model quality.

## 14. Common beginner mistakes

| Mistake | Why it is risky | Better approach |
|---|---|---|
| Train a complex model without a baseline | No simple reference exists | Establish a baseline first |
| Compare models on different splits | Comparison is unfair | Use the same evaluation design |
| Use accuracy alone on imbalanced data | Minority-class failures can be hidden | Inspect class-level metrics and business costs |
| Record only random_state | Other settings remain unknown | Record data, code, versions and preprocessing |
| Refit preprocessing on test data | Evaluation leakage | Use a fitted pipeline |
| Save only the estimator but not preprocessing | Inference may receive a different representation | Save the full pipeline |
| Load an untrusted joblib file | Pickle-based loading can execute code | Load only trusted artifacts |
| Assume reproducible means good | Repeated experiments can repeat a bad design | Validate the data, split and business objective |

## 15. Interview-ready explanations

**What is an ML workflow?**  
It is a repeatable process from problem definition and data understanding through splitting, preprocessing, baseline training, model evaluation, deployment and monitoring.

**What is a baseline model?**  
A simple reference prediction or existing business rule used to judge whether a more complex model adds useful value.

**Why is a baseline important?**  
Without a baseline, a team may celebrate a complex model that performs no better than a simple strategy.

**What is reproducibility?**  
It means recording data, code, split strategy, preprocessing, random seeds, parameters, software versions and metrics so an experiment can be understood and repeated.

**Does a fixed random seed guarantee identical results everywhere?**  
No. Hardware, package versions, numerical libraries and nondeterministic operations may affect results.

## 16. Scenario questions

1. A complex classifier has lower accuracy than a majority-class baseline. What do you do?  
   **Answer:** Inspect class-level metrics, data quality, target definition, split strategy and whether the complex model adds business value. Do not assume complexity means improvement.

2. A colleague cannot reproduce your result. What information do you share?  
   **Answer:** Dataset version, code revision, split strategy, random seeds, preprocessing, model parameters, library versions and evaluation method.

3. A model artifact loads successfully but production predictions are wrong. What do you investigate?  
   **Answer:** Confirm the artifact version, feature order, preprocessing pipeline, input schema, library versions and prediction-time data.

4. A model has 98% accuracy but misses nearly all fraud cases. Is it successful?  
   **Answer:** Not necessarily. Accuracy hides the minority-class failure. Evaluate recall, precision, PR curves and business costs.

5. Two developers get different scores from the same notebook. What do you compare?  
   **Answer:** Data version, train-test split, random seeds, library versions, preprocessing, model parameters and execution order.

## 17. Lesson completion checklist

- [ ] I can describe the basic ML workflow.
- [ ] I understand why problem definition comes first.
- [ ] I can create a classification baseline.
- [ ] I can create a regression baseline.
- [ ] I can compare a candidate model fairly.
- [ ] I can explain reproducibility.
- [ ] I can record experiment configuration and package versions.
- [ ] I can save and reload a trusted model artifact.
- [ ] I know why the complete preprocessing pipeline should be saved.
- [ ] I can explain why a baseline and reproducibility do not guarantee model quality.

**Next lesson:** Day 19 — Linear Regression, Assumptions and Equation.
