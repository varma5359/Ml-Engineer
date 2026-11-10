# Day 86: Interview Questions — Capstone Interactive Application

**Phase:** 8 — End-to-End ML Capstone

## 1. Core questions

### Q1. Why create an interactive UI for an ML model?
**Answer:**
- It enables non-technical users to submit inputs.
- It displays model results in an understandable format.
- It can support a controlled business workflow.
- It should explain uncertainty and limitations.

**Follow-up:** What user feedback would you collect before finalizing the UI?

### Q2. Why should the UI call an API instead of loading its own model?
**Answer:**
- The API centralizes model loading and inference logic.
- Multiple clients can reuse the same endpoint.
- Model versioning and validation stay consistent.
- It reduces duplicated preprocessing logic.

**Follow-up:** What would change if the UI and API were deployed as separate services?

### Q3. Why set a request timeout?
**Answer:**
- Network calls can stall.
- A timeout bounds waiting time.
- It lets the UI show a useful error and remain responsive.
- Timeout values should reflect the service's latency objective.

**Follow-up:** How do you choose a timeout for a slow model?

### Q4. What is the role of a model version in the UI?
**Answer:**
- It shows which model generated the result.
- It supports troubleshooting and traceability.
- It helps compare behavior after releases.
- It should come from the API response or trusted configuration.

**Follow-up:** Why should the UI not guess the active model version?

### Q5. How do you handle API errors in Streamlit?
**Answer:**
- Set a timeout.
- Check HTTP status.
- Catch connection and request exceptions.
- Display a safe message.
- Avoid fabricating results or showing internal stack traces.

**Follow-up:** How would you distinguish a validation error from a server outage?

## 2. Scenario questions

### Q6. The UI works but every request returns 422. What do you inspect?
**Answer:**
- Compare payload field names with Pydantic schema.
- Check required fields, types, units and allowed values.
- Inspect the API validation response.
- Test the same payload directly against the API docs.

**Follow-up:** How can an API schema change break an older UI?

### Q7. The API is unavailable. Should the UI display the last prediction?
**Answer:**
- Not as if it were a new result.
- Clearly label any cached result with its timestamp and model version if caching is approved.
- Otherwise show service unavailability.
- Avoid misleading users with stale outputs.

**Follow-up:** What is the risk of caching customer-specific predictions?

### Q8. A user believes a 70% churn probability means certain cancellation. How do you address it?
**Answer:**
- Explain that it is a model-estimated probability.
- Clarify the target horizon and calibration.
- Explain that it is not a guarantee.
- Provide a human-review workflow and limitations.

**Follow-up:** What is probability calibration?

## 3. Project questions

### Q9. Describe your Prediction interface.
**Answer:**
- I created a form matching the API feature schema.
- The UI sends a request with a timeout.
- It displays class, probability and model version.
- It handles validation and connectivity errors.
- It communicates that the result supports human review.

**Follow-up:** How did you test the integration?

### Q10. How would you make the interface production-ready?
**Answer:**
- Add authentication and authorization.
- Use secure HTTPS and managed secrets.
- Add input and output logging with privacy controls.
- Monitor UI and API errors.
- Make accessibility, usability and support requirements explicit.

**Follow-up:** How would you restrict access to different business teams?

## 4. Rapid revision
- UI: collects inputs and presents outputs.
- API: owns inference contract.
- Timeout: bounds request waiting.
- HTTP status: communicates request outcome.
- Model version: traceability.
- Probability: estimate, not certainty.
