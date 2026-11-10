# Day 16: Interview Questions — Data Leakage and Preprocessing Pipelines

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is data leakage?
**Answer:** Data leakage occurs when training or evaluation uses information that would not legitimately be available at prediction time, or when held-out data influences model development.
**Follow-up:** Why is it dangerous?  
**Answer:** It can produce unrealistically high evaluation results that do not hold in production.

### Q2. What is target leakage?
**Answer:** A feature contains information created after or derived from the outcome, such as cancellation reason recorded after a customer cancels.
**Follow-up:** How do you detect it?  
**Answer:** Inspect feature definitions, source systems and timestamps relative to prediction time.

### Q3. Why fit preprocessing on training data only?
**Answer:** Imputers, scalers and encoders learn information from data. Fitting on the full dataset lets held-out information influence those transformations.
**Follow-up:** What should happen to test data?  
**Answer:** Transform it using the fitted training transformation, without refitting.

### Q4. What is a Pipeline?
**Answer:** A Pipeline chains preprocessing steps and an estimator so they are fitted and applied in a consistent order.
**Follow-up:** Why is it useful in cross-validation?  
**Answer:** Each fold fits preprocessing only on that fold's training partition.

## Scenario questions

### Q5. A churn model uses cancellation_reason. What do you investigate?
**Answer:** When the field is recorded. If it is entered after cancellation, it is target leakage and should not be used for pre-cancellation prediction.
**Follow-up:** What if the reason is captured before the prediction timestamp?  
**Answer:** Validate its actual availability, definition and reliability before deciding.

### Q6. A scaler is fitted on all rows before train-test split. What is wrong?
**Answer:** Test information influenced the learned scaling statistics. Split first and fit the scaler on training data only.
**Follow-up:** How can you prevent this?  
**Answer:** Use a Pipeline or ColumnTransformer inside the training workflow.

### Q7. A model scores 99% but fails in production. What do you investigate?
**Answer:** Target leakage, split design, training/production preprocessing differences, feature availability, data shift and evaluation representativeness.
**Follow-up:** Should you immediately retrain?  
**Answer:** No. Diagnose the root cause first; retraining on bad data may worsen the problem.

### Q8. The same patient appears in train and test, but the goal is unseen-patient prediction. What is wrong?
**Answer:** Entity leakage. Use group-based splitting by patient.
**Follow-up:** Would group splitting be required for every use case?  
**Answer:** No. Choose based on whether deployment predicts for new or known entities and whether time order matters.

### Q9. A new city appears in production. What should the pipeline do?
**Answer:** Apply the documented unknown-category handling and monitor the new value. If the encoder is configured to ignore unknown categories, it can transform without an error.
**Follow-up:** Does this mean the model learned the new city's effect?  
**Answer:** No. It only handles the representation.

## Code interpretation

### Q10. Why use ColumnTransformer?
**Answer:** It applies different transformations to selected column groups, such as scaling numerical features and encoding categorical features.

### Q11. What is the difference between fit_transform and transform?
**Answer:** fit_transform learns transformation parameters and applies them. transform applies parameters already learned during fit.

### Q12. Why save the full pipeline instead of only the estimator?
**Answer:** The estimator expects the same transformed features used during training. Saving the pipeline preserves preprocessing and model steps together.

**Senior developer principle:** The deployed prediction path must use the same fitted preprocessing logic as the training path.
