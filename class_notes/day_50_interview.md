# Day 50: Interview Questions — Feature Engineering and Transformation

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering

## 1. Core interview questions

### Q1. What is feature engineering?
**Answer:**
- Creating, combining or transforming input variables to represent useful patterns.
- Examples include customer tenure, days since last login, purchase frequency and price multiplied by quantity.
- It can improve model usefulness when the features reflect real behaviour.
- Validate engineered features rather than assuming they help.

### Q2. Why is feature engineering important?
**Answer:**
- Raw data may not directly express the pattern needed by a model.
- Derived features can represent recency, frequency, ratios, seasonality or interactions.
- Better inputs can help a model learn useful relationships.
- Poorly designed features can introduce leakage or noise.

### Q3. How do you create features from date columns?
**Answer:**
- Convert strings to datetime safely.
- Extract relevant components such as month, weekday or hour.
- Calculate recency or tenure relative to a defined prediction timestamp.
- Ensure that all dates used were available at prediction time.

### Q4. What is one-hot encoding?
**Answer:**
- It converts a nominal categorical feature into separate indicator columns.
- A row receives a 1 for its category and 0 for the other categories.
- It avoids assigning arbitrary numeric order to categories.
- Consider category cardinality and unseen values.

### Q5. When would you use ordinal encoding?
**Answer:**
- When categories have a genuine order, such as low, medium and high.
- The assigned order must reflect the domain meaning.
- Do not use arbitrary ordinal codes for categories without a natural ranking.

### Q6. Why use log transformation?
**Answer:**
- It compresses large values in a right-skewed positive feature.
- It can make patterns easier for some models to learn.
- It does not guarantee better performance and should be validated.
- Check the value range before applying a logarithmic transformation.

### Q7. What are interaction features?
**Answer:**
- Features created by combining two or more inputs.
- Example: price multiplied by quantity to represent line-item value.
- They can help represent combined effects.
- Too many interactions can increase complexity and overfitting.

## 2. Pipeline and leakage questions

### Q8. Why use ColumnTransformer?
**Answer:**
- It applies different transformations to different feature columns.
- Numerical columns can be imputed and scaled.
- Categorical columns can be imputed and encoded.
- It keeps preprocessing organized and repeatable.

### Q9. Why put feature transformations inside a Pipeline?
**Answer:**
- The same transformations are applied during training and prediction.
- Learned transformations are fitted only on training data within each fold.
- It reduces leakage and manual preprocessing mismatch.
- It makes the workflow easier to serialize and deploy.

### Q10. What is prediction-time leakage?
**Answer:**
- A feature contains information that would not be available when the prediction is required.
- Example: using a cancellation date to predict churn before cancellation.
- It creates unrealistically strong evaluation results.
- Define a prediction timestamp and feature observation window.

### Q11. A feature is calculated using the customer's complete lifetime history. What should you check?
**Answer:**
- Whether that history includes events after the prediction timestamp.
- Whether the feature can be recreated at prediction time.
- Restrict the calculation to the allowed observation window.
- Validate the feature generation process with time-based examples.

## 3. Scenario-based interview questions

### Scenario 1: Customer tenure is a strong feature
**Expected answer:**
- Define tenure relative to the prediction date.
- Check that signup date is valid.
- Calculate tenure consistently for training and future inputs.
- Review whether tenure adds value on held-out data.

### Scenario 2: A churn model uses cancellation date
**Expected answer:**
- Cancellation date may reveal the outcome.
- It is unavailable before the customer churns.
- Remove it from pre-churn prediction features.
- Rebuild features using only information available at prediction time.

### Scenario 3: A new subscription category appears after deployment
**Expected answer:**
- Check the encoder's unknown-category behaviour.
- Use a Pipeline with a documented category handling policy.
- Monitor unseen category frequency and decide whether retraining or schema updates are needed.

### Scenario 4: Log transformation improves one metric but reduces interpretability
**Expected answer:**
- Explain why the transformation was applied.
- Compare performance and residuals.
- Decide whether the improvement justifies the interpretation cost.
- Keep clear documentation and transform predictions back when necessary for business reporting.

### Scenario 5: The model performs well in training but fails on new data
**Expected answer:**
- Check whether engineered features are calculated identically in both environments.
- Look for future information, inconsistent date handling and unseen categories.
- Verify feature names, types, units and missing-value behaviour.
- Add tests for the feature generation pipeline.

## 4. Strong points to remember in interviews

- Feature engineering creates or transforms inputs.
- Define the prediction timestamp before deriving time-based features.
- Use one-hot encoding for nominal categories.
- Use ordinal encoding only when category order is meaningful.
- Log transformation compresses large positive values but does not guarantee better results.
- Interaction features should have a reason and be validated.
- Use ColumnTransformer for mixed feature types.
- Put learned preprocessing inside a Pipeline.
- Never use future information to create historical prediction features.

**Senior developer principle:** A feature is valid only if it can be generated correctly, consistently and at the moment the real prediction must be made.
