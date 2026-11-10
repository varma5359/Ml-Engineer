# Day 66: Interview Questions — Streamlit or Equivalent UI for Model Demonstrations

**Phase:** 7 — Deployment and MLOps

## 1. Core questions

### Q1. Why build a UI for an ML model?
**Answer:**
- It lets non-technical users submit inputs and view predictions.
- It makes model behavior easier to demonstrate.
- It can support internal decision-support workflows.
- The UI must communicate limitations and should not imply predictions are guaranteed.

### Q2. What is Streamlit?
**Answer:**
- A Python framework for interactive data applications.
- It provides widgets, charts, tables and metrics.
- It can call a Python model directly.
- It is useful for prototypes and internal applications.

### Q3. Explain the UI-to-model flow.
**Answer:**
1. User enters feature values.
2. UI validates the inputs.
3. Inputs are converted to the model's expected structure.
4. The fitted pipeline generates a prediction.
5. The UI displays the result and relevant context.

### Q4. Why use a serialized pipeline rather than only a model estimator?
**Answer:**
- The pipeline includes preprocessing and the estimator.
- It applies training transformations consistently.
- It reduces training-serving mismatch.
- The artifact and its source must be trusted.

### Q5. What is st.cache_resource?
**Answer:**
- A Streamlit caching decorator for resources such as models or database connections.
- It avoids recreating expensive resources on every rerun.
- Cache invalidation and model updates need deliberate handling.
- Cached resources should be safe to share under the app's concurrency model.

### Q6. Is a Streamlit demo automatically production-ready?
**Answer:**
- No.
- Production requires access controls, secure secrets, monitoring and deployment planning.
- Data privacy, reliability and scaling must be considered.
- A prototype should not be exposed publicly without review.

## 2. Scenario questions

### Q7. The app works but every prediction is wrong. What do you check?
**Answer:**
- Input feature names, order, types and units.
- Whether the correct model artifact was loaded.
- Preprocessing consistency.
- Target label mapping.
- Compare the same input in a trusted test script and UI.

### Q8. The model file is missing. What should the UI do?
**Answer:**
- Show a clear, non-technical error.
- Stop the prediction workflow.
- Avoid exposing filesystem paths or stack traces.
- Document how the artifact is created and deployed.

### Q9. A user interprets a 0.8 probability as an 80% guarantee. How do you respond?
**Answer:**
- Explain that it is a model-estimated probability.
- Check whether probabilities are calibrated.
- Explain the target definition and evaluation population.
- Show uncertainty and limitations where appropriate.

### Q10. The UI displays sensitive customer inputs. What controls are needed?
**Answer:**
- Collect only necessary fields.
- Restrict access and use secure transport.
- Avoid storing or logging raw sensitive values unnecessarily.
- Apply retention and deletion policies.

## 3. Project questions

### Q11. Describe your Interactive ML Application.
**Answer:**
- I loaded a trusted fitted pipeline.
- I created widgets matching the model's feature schema.
- I validated inputs before prediction.
- I displayed the output and model version.
- I tested normal, boundary and missing-artifact cases.

### Q12. Why cache the model?
**Answer:**
- Streamlit reruns the script when users interact.
- Caching avoids repeatedly loading the artifact.
- It improves responsiveness.
- Model updates need cache invalidation or controlled application restart.

### Q13. How would you test an ML UI?
**Answer:**
- Test input constraints and required fields.
- Test valid prediction and output format.
- Test missing or incompatible artifacts.
- Compare UI predictions with direct pipeline predictions.
- Test access and privacy controls before deployment.

## 4. Rapid revision
- Streamlit: Python UI framework.
- Widget: user input control.
- Pipeline: preprocessing plus estimator.
- Cache resource: reuse expensive resources.
- Probability is not certainty.
- A demo UI is not automatically a secure production service.
