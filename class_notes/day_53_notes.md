# Day 53: Model Interpretability and Responsible ML

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives
- Explain model interpretability and explainability.
- Distinguish global and local explanations.
- Understand transparent models and post-hoc explanation methods.
- Inspect model behaviour with feature importance and permutation importance.
- Understand partial dependence and individual prediction explanations.
- Identify fairness, privacy, security and governance risks.
- Document model limitations and intended use.
- Communicate explanations without claiming causation.

## 2. Why model interpretability matters

A model may produce a prediction, but a business user may ask:
- Why was this customer marked as high risk?
- Which factors influenced this prediction?
- Does the model behave differently for different customer groups?
- Can we explain the model's limitations?
- Is the prediction suitable for an important decision?

Interpretability helps developers, users and reviewers understand model behaviour.

It can support:
- Debugging.
- Model validation.
- Stakeholder trust.
- Risk management.
- Compliance and governance.
- Identifying unexpected model behaviour.

Interpretability does not automatically make a model fair, correct or safe.

## 3. Interpretability versus explainability

The terms are often used together.

- **Interpretability:** How directly a person can understand how a model works or how inputs relate to its outputs.
- **Explainability:** Methods or information used to explain a model's behaviour or an individual prediction.

Examples:
- A small decision tree may be directly interpretable.
- A large ensemble may require post-hoc explanation tools.
- A feature importance chart provides one view of model reliance, not a complete explanation.

## 4. Global versus local explanations

### Global explanation
Describes overall model behaviour across a dataset.

Examples:
- Which features are important overall?
- How does a model generally respond to monthly bill?
- Which groups have higher average predicted risk?

### Local explanation
Describes one prediction.

Examples:
- Why did this particular loan application receive a high-risk score?
- Which input values influenced this one prediction?
- What factors distinguish this case from a reference case?

A global explanation does not automatically explain every individual prediction.

## 5. Transparent models versus post-hoc explanations

| Approach | Meaning | Example |
|---|---|---|
| Inherently interpretable model | Model structure can be examined directly | Small decision tree, linear model with suitable feature design |
| Post-hoc explanation | Explanation is generated after fitting a more complex model | Permutation importance, partial dependence, SHAP |
| Example-based explanation | Shows similar or contrasting examples | Nearest similar records or counterfactual examples |

No explanation method is perfect. Choose a method that matches the model, audience and decision.

## 6. Create a small classification model

### 6.1 Import libraries

~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Libraries ready")
~~~

### 6.2 Create a sample dataset

~~~python
from sklearn.datasets import make_classification

X_values, y = make_classification(
    n_samples=500,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    random_state=42
)

feature_names = [
    "monthly_bill",
    "tenure",
    "support_calls",
    "late_payments",
    "usage_change",
    "plan_type_code"
]

X = pd.DataFrame(X_values, columns=feature_names)

print(X.head())
~~~

**Explanation**
This is synthetic teaching data. The feature names are illustrative and do not establish real-world relationships.

### 6.3 Split training and test data

~~~python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(X_train.shape, X_test.shape)
~~~

## 7. Train a model for interpretation

### 7.1 Fit a Random Forest

~~~python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained")
~~~

### 7.2 Evaluate the model first

~~~python
from sklearn.metrics import classification_report

predictions = model.predict(X_test)

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)
~~~

**Explanation**
An explanation chart does not replace performance evaluation. First understand whether the model is useful and how it performs on held-out observations.

## 8. Global explanation — built-in feature importance

### 8.1 Read feature importances

~~~python
importance_table = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print(importance_table)
~~~

**Explanation**
The table summarizes how the fitted Random Forest used features under its built-in importance calculation.

Cautions:
- Importance is not causation.
- Correlated features can share importance.
- Some impurity-based methods can favour features with many possible split points.
- Values depend on the model and dataset.

### 8.2 Visualize global importance

~~~python
importance_table.sort_values("importance").plot(
    x="feature",
    y="importance",
    kind="barh",
    legend=False
)

plt.xlabel("Importance")
plt.title("Global Feature Importance")
plt.show()
~~~

**Expected output**
A horizontal bar chart ranking the features by built-in importance.

**Explanation**
This is a global view. It does not tell us why a specific customer's prediction was positive.

## 9. Permutation importance

Permutation importance evaluates how much the model's score changes when one feature is shuffled.

### 9.1 Calculate permutation importance

~~~python
from sklearn.inspection import permutation_importance

permutation_result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="f1"
)

print(permutation_result.importances_mean)
~~~

### 9.2 Create a report

~~~python
permutation_table = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": permutation_result.importances_mean,
    "importance_std": permutation_result.importances_std
}).sort_values("importance_mean", ascending=False)

print(permutation_table)
~~~

**Explanation**
- A larger score drop after shuffling suggests the model relies on that feature for the selected score.
- A near-zero or negative value may mean little measured contribution in this evaluation setup.
- Correlated features can substitute for each other.
- This is a model-reliance measure, not causal importance.

## 10. Partial dependence — global behaviour view

A partial dependence plot shows how the model's average prediction changes as one selected feature is varied, while averaging over other feature values.

Use it to explore broad model behaviour.

Limitations:
- Correlated features can make the plot misleading.
- It averages across observations and can hide individual differences.
- It may show unrealistic combinations when features are dependent.
- It does not prove that changing the feature causes the outcome.

### 10.1 Generate a partial dependence plot

~~~python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model,
    X_test,
    features=["monthly_bill", "tenure"]
)

plt.show()
~~~

**Expected output**
Partial dependence plots for the selected features.

**Explanation**
The plot shows the model's average response over the selected feature range. Interpret it as model behaviour, not a causal relationship.

## 11. Local explanation — inspect one prediction

A local explanation focuses on one record.

A simple first step is to inspect:
- The input values.
- The model's predicted class.
- The model's score for each class.
- Relevant global feature information.
- Similar cases or a model-specific explanation method.

### 11.1 Select one test observation

~~~python
one_customer = X_test.iloc[[0]]

print(one_customer)
~~~

### 11.2 Generate the prediction

~~~python
one_prediction = model.predict(one_customer)

print("Predicted class:", one_prediction[0])
~~~

### 11.3 Inspect class scores

~~~python
one_scores = model.predict_proba(one_customer)

print("Class scores:", one_scores)
~~~

**Explanation**
These scores are model outputs. They should not automatically be treated as perfectly calibrated probabilities.

### 11.4 Compare with a few other records

~~~python
comparison_records = X_test.head(5).copy()
comparison_records["predicted_class"] = model.predict(
    comparison_records
)

print(comparison_records)
~~~

**Explanation**
This gives a simple way to inspect inputs and predictions together. It is not a complete explanation of why the selected record received its prediction.

## 12. SHAP and other explanation tools

SHAP is a popular explanation framework that assigns feature contribution values for model predictions under a defined explanation setup.

It can support:
- Global importance summaries.
- Local explanations for individual predictions.
- Visual inspection of feature contributions.

Important limitations:
- Explanations depend on the background data and explainer configuration.
- Correlated features complicate contribution attribution.
- Contribution values are not causal effects.
- Complex plots still need careful interpretation.
- Explanation fidelity and computational cost vary by model.

Other tools include LIME, partial dependence, individual conditional expectation and counterfactual explanations. Select methods based on model compatibility and the question being asked.

## 13. Responsible ML: fairness and bias

A model can produce different error rates or outcomes across groups.

Potential sources:
- Historical data reflects unequal treatment.
- Some groups are underrepresented.
- Labels are inconsistent or biased.
- Features act as proxies for sensitive characteristics.
- Measurement quality differs between groups.
- A model is deployed in a new population.

Responsible checks:
- Define the decision and affected population.
- Review data coverage and label quality.
- Evaluate relevant subgroups.
- Compare false positive and false negative rates where appropriate.
- Consult domain, legal and governance experts.
- Document limitations and mitigation decisions.

Fairness is context-dependent. A single fairness metric does not resolve every ethical or legal question.

## 14. Privacy, security and governance

Before using personal or sensitive data:
- Collect only what is necessary for the defined purpose.
- Follow applicable privacy and organizational policies.
- Restrict access to data and model outputs.
- Avoid exposing sensitive information in logs or reports.
- Protect model artifacts and prediction APIs.
- Document retention and deletion practices.
- Review whether a feature is appropriate and permitted for the decision.

## 15. Human oversight and safe use

For high-impact decisions:
- Do not assume a model prediction should automatically determine the outcome.
- Define who reviews uncertain or high-risk cases.
- Provide a way to challenge or correct data.
- Record decisions and review outcomes.
- Monitor performance and subgroup impacts after deployment.
- Define escalation and rollback procedures.

A model explanation should support a responsible decision process, not replace it.

## 16. Model documentation

A useful model card or model summary should describe:
- Intended purpose.
- Intended users.
- Out-of-scope uses.
- Training and evaluation data.
- Key features and data limitations.
- Evaluation metrics and subgroup results.
- Known failure modes.
- Explanation methods used.
- Privacy and fairness considerations.
- Monitoring and review plan.

## 17. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Say feature importance proves causation | Describe model reliance, not cause |
| Use global importance to explain one record | Use an appropriate local explanation |
| Assume a transparent model is automatically fair | Evaluate data and outcomes |
| Trust a post-hoc explanation without checking | Assess explanation limitations and model fidelity |
| Ignore correlated features | Explain attribution uncertainty |
| Publish sensitive records in explanation reports | Apply privacy controls and minimization |
| Assume one fairness metric solves fairness | Review context, groups and impacts |
| Use a model beyond its intended purpose | Document intended and out-of-scope uses |

## 18. Interview-ready explanations

**What is model interpretability?**
- How directly people can understand model structure or behaviour.
- It supports debugging, validation, trust and governance.

**What is the difference between global and local explanation?**
- Global describes overall model behaviour.
- Local describes one prediction or observation.

**What is permutation importance?**
- Shuffle a feature and measure the change in model score.
- It indicates model reliance under the selected evaluation setup.
- It does not prove causation.

**What is SHAP?**
- A framework for explaining model predictions using feature contribution values.
- It can provide global and local views.
- Results depend on assumptions and background data and are not causal effects.

**What is responsible ML?**
- Developing and using models with attention to fairness, privacy, security, transparency, accountability and appropriate human oversight.

## 19. Scenario-based interview practice

### Scenario 1: A bank asks why one loan applicant was rejected
**Expected response:**
- Provide an appropriate local explanation of the factors influencing the model output.
- Check whether inputs were accurate and available at decision time.
- Explain the limitations of the model and explanation method.
- Follow the organization's review and appeal process.
- Do not present feature contribution as proof of causation.

### Scenario 2: A model has high overall performance but poor results for one customer group
**Expected response:**
- Report subgroup metrics and sample sizes.
- Inspect data representation and label quality.
- Review error types and possible proxy features.
- Involve responsible domain and governance reviewers.
- Document mitigation and re-evaluate.

### Scenario 3: A feature is highly important but represents sensitive personal information
**Expected response:**
- Check legal, policy and business purpose requirements.
- Determine whether the feature is necessary and permitted.
- Assess privacy and fairness risks, including proxy effects.
- Consider safer alternatives and document the decision.

### Scenario 4: SHAP says a feature contributed strongly to one prediction
**Expected response:**
- Explain that the contribution describes the model output under the SHAP setup.
- Check background data and feature dependencies.
- Do not claim that the feature caused the outcome.
- Combine explanation with domain review and data validation.

## 20. Lesson completion checklist

- [ ] I can explain interpretability and explainability.
- [ ] I can distinguish global and local explanations.
- [ ] I can inspect built-in and permutation importance.
- [ ] I understand partial dependence limitations.
- [ ] I can explain SHAP at a high level.
- [ ] I understand fairness, privacy and governance risks.
- [ ] I can describe human oversight for high-impact decisions.
- [ ] I can outline a model card.
- [ ] I can answer responsible ML interview scenarios.

**Next lesson:** Day 54 — Gradient Boosting Fundamentals.
