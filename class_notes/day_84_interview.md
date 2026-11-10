# Day 84: Interview Questions — Capstone Explainability and Business Interpretation

**Phase:** 8 — End-to-End ML Capstone

## 1. Core questions

### Q1. What is model explainability?
**Answer:**
- Methods that help understand model behavior and predictions.
- They can be global or local.
- They support debugging, communication and governance.
- They do not automatically prove a model is correct or causal.

**Follow-up:** When might an interpretable model be preferred over a complex model?

### Q2. Compare global and local explanations.
**Answer:**
- Global methods summarize patterns across many observations.
- Local methods explain one prediction.
- Global analysis supports overall model review.
- Local analysis helps investigate a specific output.

**Follow-up:** Can a feature be important globally but have little contribution for one customer?

### Q3. What is permutation importance?
**Answer:**
- It measures score change after shuffling a feature.
- A larger score decrease suggests the model relies on that feature for the chosen metric.
- Results depend on data, scoring and feature correlation.
- It is not a causal effect.

**Follow-up:** Why can correlated features share importance?

### Q4. What is SHAP?
**Answer:**
- A framework based on Shapley-value ideas for feature contributions.
- It can provide local and global views.
- It explains model output relative to a reference expectation.
- Explainer choice and output interpretation depend on model and data representation.

**Follow-up:** Why must transformed feature names be mapped carefully?

### Q5. Why is feature importance not causation?
**Answer:**
- A predictive model learns statistical patterns.
- Correlation or contribution does not show an intervention would change the outcome.
- Confounding and historical policy effects may influence relationships.
- Causal claims need an appropriate study design.

**Follow-up:** What type of experiment could test a retention intervention?

## 2. Scenario questions

### Q6. A manager says "tenure causes churn" because tenure ranks highly. How do you respond?
**Answer:**
- Explain that the model found predictive association.
- It does not establish causal direction or intervention effect.
- Show the model's intended use and limitations.
- Recommend causal analysis or a controlled experiment for intervention claims.

**Follow-up:** What would a randomized retention experiment measure?

### Q7. SHAP returns more feature values than the original input columns. Why?
**Answer:**
- Preprocessing may expand categorical variables into multiple one-hot columns.
- The explainer may operate on transformed features.
- Inspect the transformed feature names and SHAP output dimensions.
- Map each contribution back to the correct encoded feature or grouped original field.

**Follow-up:** How would you group one-hot contributions for a business report?

### Q8. Two correlated features have low individual permutation importance. What might explain it?
**Answer:**
- The model may use one feature as a substitute for the other.
- Shuffling one feature may not reduce performance much because correlated information remains.
- Examine feature relationships and grouped/permutation approaches.
- Interpret importance with caution.

**Follow-up:** What other analysis could help understand feature redundancy?

## 3. Project questions

### Q9. What should a model interpretation report contain?
**Answer:**
- Model and dataset version.
- Global importance method and metric.
- Important feature plots.
- Local prediction explanations.
- Plain-language interpretation.
- Limitations, correlation and causality caveats.
- Privacy considerations.

**Follow-up:** How do you ensure the report corresponds to the deployed model?

### Q10. How do you explain a risk score to a non-technical user?
**Answer:**
- Explain the target and prediction horizon.
- State that it is an estimated risk, not certainty.
- Describe key model factors without causal overclaim.
- Explain what action is appropriate and who makes the decision.
- Communicate known limitations.

**Follow-up:** What if a user treats a high score as a guaranteed churn event?

## 4. Rapid revision
- Global: dataset-level model behavior.
- Local: one prediction.
- Permutation importance: score change after shuffling.
- SHAP: feature contribution relative to a reference.
- Feature importance is not causation.
- Map encoded features to understandable fields.
