# Day 63: Interview Questions — Scikit-learn Pipelines and Serialization

**Phase:** 6 — Advanced ML and Practical Applications

## 1. Core questions

### Q1. What is a scikit-learn Pipeline?
**Answer:**
- It chains transformers and a final estimator.
- Steps execute in sequence.
- It keeps preprocessing consistent between training and inference.
- It can be used with cross-validation and hyperparameter search.

### Q2. Explain fit, transform and predict.
**Answer:**
- fit learns parameters from data.
- transform applies learned transformations.
- predict generates model outputs.
- A pipeline's predict method applies intermediate transformations before the final estimator predicts.

### Q3. Why serialize a model?
**Answer:**
- To save a trained artifact for later use.
- It avoids retraining every time the service starts.
- It supports model handoff and deployment.
- The artifact must be versioned and loaded in a compatible environment.

### Q4. What is joblib?
**Answer:**
- A Python library commonly used to persist scikit-learn objects.
- It can save and load fitted pipelines.
- It is convenient for Python-based inference.
- Only load trusted files because deserialization can execute code.

### Q5. Why validate input schema?
**Answer:**
- The model expects specific features.
- Missing or reordered columns can cause errors or incorrect predictions.
- Validate names, types and allowed values.
- Return clear errors rather than silently accepting invalid input.

### Q6. How do you test a saved model?
**Answer:**
- Load the artifact in a clean process.
- Use a known input record.
- Compare predictions before and after serialization.
- Validate output shape and schema.
- Test missing, extra and malformed fields.

## 2. Practical scenarios

### Q7. A loaded model gives different predictions from the original notebook. What do you check?
**Answer:**
- Artifact path and version.
- Preprocessing included in the artifact.
- Input feature names, order and types.
- Library versions.
- Model configuration and prediction code.
- Compare predictions using exactly the same input.

### Q8. The API receives a column that was not used during training. What should happen?
**Answer:**
- Validate against the expected schema.
- Decide whether extra columns should be rejected or explicitly ignored.
- Never silently change feature meaning.
- Log schema errors without exposing sensitive values.

### Q9. Can you load a joblib file uploaded by an unknown user?
**Answer:**
- No, not safely by default.
- joblib relies on pickle-based object deserialization.
- Malicious files can execute code during loading.
- Verify provenance and use controlled artifact storage.

### Q10. Why is saving only a StandardScaler insufficient?
**Answer:**
- The estimator is also required to generate predictions.
- Save the full pipeline containing preprocessing and estimator.
- Keep schema and metadata with the artifact.

## 3. Project questions

### Q11. Explain your Model Serialization Tool.
**Answer:**
- I trained a scikit-learn pipeline.
- I saved the complete pipeline with joblib.
- I recorded feature schema and dependency versions.
- I loaded the artifact in a separate workflow.
- I validated input columns and compared predictions before and after loading.

### Q12. What is model compatibility?
**Answer:**
- A saved model depends on software libraries and runtime behavior.
- Version changes may break loading or alter behavior.
- Record tested versions and validate after upgrades.
- Use controlled environments for production inference.

### Q13. What is the difference between model artifact and metadata?
**Answer:**
- Artifact contains the fitted model or pipeline.
- Metadata describes its identity, schema, versions and training context.
- Both are needed for safe and traceable use.

## 4. Rapid revision
- Pipeline: preprocessing plus estimator.
- fit: learn.
- transform: apply transformation.
- predict: generate output.
- joblib: persist Python objects.
- Schema validation: protects inference inputs.
- Only load trusted serialized artifacts.
