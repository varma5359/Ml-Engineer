# Day 53: Interview Questions — Model Interpretability and Responsible ML

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering

## 1. Core interview questions

### Q1. What is model interpretability?
**Answer:**
- How directly a person can understand a model's structure or behaviour.
- It helps developers inspect decisions and identify unexpected patterns.
- It supports debugging, validation, stakeholder communication and governance.
- Interpretability does not automatically prove correctness or fairness.

### Q2. What is the difference between global and local explanations?
**Answer:**
- Global explanations describe overall model behaviour across a dataset.
- Local explanations describe one particular prediction.
- Feature importance is commonly a global view.
- A local explanation is needed to discuss one customer's prediction.

### Q3. What is the difference between an interpretable model and a post-hoc explanation?
**Answer:**
- An interpretable model has a structure people can inspect directly, such as a small decision tree.
- A post-hoc method explains a model after it has been trained.
- A complex ensemble may use feature importance, SHAP or partial dependence for additional insight.
- Post-hoc explanations have limitations and should not be treated as a perfect description of the model.

### Q4. What is permutation importance?
**Answer:**
- Shuffle one feature in evaluation data.
- Measure how much the model's score changes.
- A larger score drop suggests the model relied more on that feature for the selected score.
- It is not causal importance.

**Follow-up:** What if features are correlated?  
**Answer:** The model may use one feature as a substitute for another. Individual permutation importance can understate their shared information.

### Q5. What is SHAP?
**Answer:**
- A framework that explains predictions using feature contribution values.
- It can provide global and local explanations.
- Results depend on the explainer setup and background data.
- Contributions describe model behaviour; they do not prove that a feature caused the outcome.

### Q6. What is a partial dependence plot?
**Answer:**
- It shows how the model's average prediction changes as a selected feature varies.
- It is a global model-behaviour view.
- Correlated features can make it misleading.
- It does not establish a causal effect.

## 2. Responsible ML questions

### Q7. What is responsible machine learning?
**Answer:**
- Developing and using models with attention to fairness, privacy, security, transparency and accountability.
- Considering who is affected by the model.
- Evaluating performance across relevant groups.
- Providing human oversight where decisions have significant impact.

### Q8. How can bias enter a machine learning model?
**Answer:**
- Historical data may reflect unequal treatment.
- Some groups may be underrepresented.
- Labels may be inconsistent or biased.
- Features may act as proxies for sensitive attributes.
- Data quality and measurement can differ across groups.

### Q9. How would you evaluate fairness?
**Answer:**
- Define the decision and affected groups.
- Check data coverage and label quality.
- Compare relevant performance metrics across groups.
- Inspect false positive and false negative rates where appropriate.
- Involve domain and governance experts.
- Document limitations and mitigation actions.

**Follow-up:** Is one fairness metric enough?  
**Answer:** No. Fairness depends on context, impact, legal requirements and the decision being made. A single metric cannot answer every concern.

### Q10. Why is privacy important in ML?
**Answer:**
- Models often use personal or sensitive information.
- Collect only data needed for the defined purpose.
- Restrict access and protect logs, datasets and artifacts.
- Follow applicable privacy and organizational policies.
- Document retention and use limitations.

### Q11. What is a model card?
**Answer:**
- A structured document describing a model's intended use, data, evaluation, limitations and risks.
- It helps users understand where the model is appropriate and where it should not be used.
- It supports transparency and responsible operation.

## 3. Scenario-based interview questions

### Scenario 1: A bank asks why one loan applicant was rejected
**Expected answer:**
- Provide an appropriate local explanation of the factors influencing the model output.
- Check whether inputs were accurate and available at decision time.
- Explain limitations of the model and explanation method.
- Follow the organization's review and appeal process.
- Do not present feature contribution as proof of causation.

### Scenario 2: A model has high overall performance but poor results for one customer group
**Expected answer:**
- Report subgroup metrics and sample sizes.
- Inspect data representation and label quality.
- Review error types and possible proxy features.
- Involve responsible domain and governance reviewers.
- Document mitigation and re-evaluate.

### Scenario 3: A feature is highly important but represents sensitive personal information
**Expected answer:**
- Check legal, policy and business purpose requirements.
- Determine whether the feature is necessary and permitted.
- Assess privacy and fairness risks, including proxy effects.
- Consider safer alternatives and document the decision.

### Scenario 4: SHAP says a feature contributed strongly to one prediction
**Expected answer:**
- Explain that the contribution describes the model output under the SHAP setup.
- Check background data and feature dependencies.
- Do not claim that the feature caused the outcome.
- Combine explanation with domain review and data validation.

### Scenario 5: A stakeholder wants to use a model outside its documented purpose
**Expected answer:**
- Explain the intended use and known limitations.
- Assess whether the new use changes the affected population or risk.
- Require appropriate validation and governance review before use.
- Update documentation and monitoring requirements if approved.

## 4. Strong points to remember in interviews

- Interpretability is about understanding model structure or behaviour.
- Global explanations describe overall behaviour; local explanations describe one prediction.
- Permutation importance measures score change after shuffling a feature.
- SHAP explains model contributions but does not prove causality.
- Partial dependence can be misleading with correlated features.
- High overall accuracy can hide subgroup failures.
- Fairness is context-dependent and needs more than one metric.
- Protect personal data and model artifacts.
- Document intended use, limitations and monitoring.
- Use human review for high-impact decisions where appropriate.

**Senior developer principle:** An explanation helps people inspect a model; responsible ML ensures the model is appropriate, fair enough for its context, secure and governed throughout its lifecycle.
