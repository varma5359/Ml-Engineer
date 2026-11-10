# Day 84: Capstone — Explainability and Business Interpretation

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved mini-project / deliverable:** Model interpretation report

## 1. Learning objectives
- Explain why model predictions need interpretation.
- Distinguish global and local explanations.
- Inspect model feature importance.
- Use permutation importance and SHAP where appropriate.
- Communicate limitations and avoid causal claims.

## 2. Why explain model predictions?
Business users need to understand what a model has learned, where it may be unreliable and how to use its output responsibly.

Explainability can help:
- Identify influential features.
- Debug unexpected model behavior.
- Review individual predictions.
- Communicate limitations.
- Support governance and model review.

An explanation describes model behavior; it does not automatically establish causality or prove a prediction is correct.

## 3. Global and local explanations
- **Global explanation:** summarizes behavior across a dataset, such as overall feature importance.
- **Local explanation:** describes factors influencing one prediction.
- **Intrinsic interpretability:** model structure is comparatively understandable, such as a simple linear model.
- **Post-hoc explanation:** an additional method explains a fitted model.

## 4. Inspect permutation importance
Permutation importance measures how much a chosen evaluation score changes when one feature is shuffled.

```python
from sklearn.inspection import permutation_importance
import pandas as pd

result = permutation_importance(
    best_model,
    X_validation,
    y_validation,
    scoring="average_precision",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance = pd.DataFrame({
    "feature": X_validation.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values("importance_mean", ascending=False)

print(importance.head(15))
```

This example assumes `best_model` accepts the original raw feature columns through its full pipeline. Correlated features can share or mask importance. Results depend on the scoring metric and validation population.

## 5. Visualize global importance
```python
import matplotlib.pyplot as plt

top = importance.head(12).sort_values("importance_mean")
plt.figure(figsize=(8, 5))
plt.barh(top["feature"], top["importance_mean"])
plt.xlabel("Mean score decrease after permutation")
plt.title("Permutation Feature Importance")
plt.tight_layout()
plt.show()
```

Importance is not a causal effect and can be misleading with correlated predictors.

## 6. SHAP overview
SHAP assigns feature contributions to a prediction relative to a reference expectation. It can provide global and local explanations, but the appropriate explainer depends on the model and preprocessing representation.

Install if needed:
```bash
python -m pip install shap
```

For a tree model, a simplified example is:
```python
import shap

tree_model = best_model.named_steps["classifier"]
X_transformed = best_model.named_steps["preprocessor"].transform(
    X_validation.iloc[:100]
)

explainer = shap.TreeExplainer(tree_model)
shap_values = explainer.shap_values(X_transformed)
```

SHAP output shapes and class conventions vary by library/model version. Check the returned structure before plotting. Map transformed feature names back to understandable source fields where possible.

## 7. Explain one prediction
A useful local explanation should state:
- The predicted risk and model version.
- The features that contributed most under the selected explainer.
- The reference population or baseline.
- The explanation's limitations.
- The action available to the business user.

Do not tell a customer that a feature "caused" churn based solely on feature contribution.

## 8. Translate technical results into business language
Example:
- Technical: "Contract type and tenure contribute strongly to the model's ranking."
- Business interpretation: "The model often assigns higher risk to some shorter-tenure contract patterns in this dataset."
- Limitation: "This is an association learned from historical records, not proof that changing contract type will prevent churn."

## 9. Approved deliverable — Model interpretation report

**Objective:** Explain global model behavior and selected individual predictions for the churn capstone.

### Tasks
1. Select a validation dataset and chosen model.
2. Calculate permutation importance using a justified metric.
3. Visualize the most influential features.
4. Use SHAP or another suitable local explanation method where feasible.
5. Explain at least two individual predictions.
6. Translate results into plain business language.
7. Document correlated-feature, sampling and explanation limitations.
8. State why feature contribution is not causation.

**Suggested structure**
```text
customer_churn_capstone/
├── reports/model_interpretation.md
├── reports/figures/
└── notebooks/06_explainability.ipynb
```

**Validation:** Confirm explanations use the selected model and appropriate data. Verify feature names align after preprocessing. Do not present causal claims or sensitive individual information unnecessarily.

## 10. AI-assisted implementation guide
| Level | Prompt |
|---|---|
| Hint | "Give me one hint for interpreting permutation importance." |
| Plan | "Plan a global and local explainability report." |
| Example | "Show a small example of a feature contribution explanation." |
| Guided implementation | "Help me calculate permutation importance for my fitted pipeline." |
| Explanation | "Explain why correlated features can distort importance." |
| Debugging | "Help me interpret the SHAP output shape returned by my model." |
| Review | "Review my interpretation report for causal overclaims and privacy risks." |

## 11. Daily quiz

**Q1. What does global explainability summarize?**  
A. One request only  
B. Model behavior across a dataset  
C. API uptime  
D. Docker layers  
**Correct: B.** Global methods summarize patterns across observations.

**Q2. What does permutation importance do?**  
A. Shuffles a feature and measures score change  
B. Deletes the target  
C. Retrains a new dataset version  
D. Changes API port  
**Correct: A.** It estimates reliance on a feature under a chosen score.

**Q3. Does feature importance prove causation?**  
A. Yes, always  
B. No, it describes model behavior or association  
C. Only for trees  
D. Only for SHAP  
**Correct: B.** Explanations do not establish causal effects.

**Q4. What is a local explanation?**  
A. Explanation of an individual prediction  
B. A whole-dataset schema  
C. A model container  
D. A training schedule  
**Correct: A.** It describes contributions for a particular observation.

**Q5. Why should transformed feature names be checked?**  
A. To make the API faster  
B. To ensure explanations map to the correct original concepts  
C. To increase target prevalence  
D. To avoid saving models  
**Correct: B.** One-hot encoding and transformations change the feature representation.

## 12. Interview preparation

### Q1. Explain global versus local explainability.
**Answer:** Global explainability summarizes how a model behaves across a dataset, while local explainability examines factors contributing to one prediction. Both are useful: global methods support model review, and local methods help investigate individual outputs.

**Follow-up:** Why can local explanations differ for two similar customers?

### Q2. Does SHAP explain why a customer churned?
**Answer:** SHAP explains how features contribute to a model prediction relative to a reference expectation. It does not prove why the real customer churned or that changing a feature will change the outcome. Causal claims need a suitable causal design and evidence.

**Follow-up:** How would you explain that limitation to a business stakeholder?

## 13. Completion checklist
- [ ] Calculate global permutation importance.
- [ ] Visualize important features.
- [ ] Explain selected individual predictions.
- [ ] Map transformed features correctly.
- [ ] Document limitations and avoid causal claims.
- [ ] Complete Model interpretation report.

**Next lesson:** Day 85 — Capstone: Prediction API.
