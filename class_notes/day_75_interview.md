# Day 75: Interview Questions — ML Security, Privacy and Governance

**Phase:** 7 — Deployment and MLOps

## 1. Core questions

### Q1. Why is security important in ML systems?
**Answer:**
- ML systems process data and valuable model artifacts.
- APIs and pipelines create attack surfaces.
- Unauthorized access can expose data or manipulate predictions.
- Security must cover training, storage, serving and monitoring.

### Q2. What is least privilege?
**Answer:**
- Give each identity only the permissions needed for its role.
- Separate training and inference permissions.
- Restrict artifact and data access.
- Review and revoke permissions when no longer needed.

### Q3. What is data poisoning?
**Answer:**
- Malicious or manipulated data influences training.
- It can change model behavior or create targeted failures.
- Validate provenance, schema and suspicious records.
- Restrict data write access and review training changes.

### Q4. What is model theft?
**Answer:**
- Unauthorized copying or extraction of a model.
- It may occur through artifact access or repeated API queries.
- Protect model storage and endpoints.
- Apply authentication, authorization, rate limits and monitoring.

### Q5. Why are pickle and joblib artifacts sensitive?
**Answer:**
- Their deserialization can execute code.
- Untrusted files can compromise the loading environment.
- Load only artifacts from trusted, verified sources.
- Control registry access and provenance.

### Q6. What is data minimization?
**Answer:**
- Collect and process only data necessary for the defined purpose.
- Avoid unnecessary personal attributes.
- Restrict retention and access.
- Review whether fields can be removed or transformed.

## 2. Governance questions

### Q7. What should model documentation contain?
**Answer:**
- Intended use and limitations.
- Data sources and known gaps.
- Evaluation methods and results.
- Model owner and approval.
- Version, monitoring and incident process.
- Human oversight requirements.

### Q8. What is the difference between privacy and security?
**Answer:**
- Privacy concerns appropriate collection, use, sharing and retention of personal information.
- Security protects systems and information from unauthorized access or damage.
- They overlap but are not identical.
- Both need defined controls and accountability.

### Q9. Does a risk register prove regulatory compliance?
**Answer:**
- No.
- It records identified risks and planned controls.
- Compliance depends on applicable requirements and actual implementation.
- Legal, privacy and risk owners should review the system.

## 3. Scenario questions

### Q10. An API key appears in a Git commit. What do you do?
**Answer:**
- Revoke or rotate it immediately.
- Assess whether it was accessed or used.
- Remove it from active code and secure secret storage.
- Follow repository history cleanup policy if required.
- Review logs and prevent recurrence.

### Q11. A developer wants to log every raw request for debugging. What do you recommend?
**Answer:**
- Identify the minimum diagnostic information required.
- Redact or tokenize sensitive fields.
- Restrict log access and retention.
- Use synthetic examples where possible.
- Follow privacy and security policy.

### Q12. A model is used for a purpose outside its documentation. What is the concern?
**Answer:**
- The model may not have been validated for that population or decision.
- Errors and harms may be different.
- Pause or restrict the use pending review.
- Reassess data, metrics, oversight and governance.

## 4. Project questions

### Q13. Describe your ML Risk Assessment.
**Answer:**
- I documented system purpose, data and users.
- I identified risks across data, training, artifacts and API serving.
- I rated likelihood and impact.
- I proposed controls, owners and review frequencies.
- I documented privacy, governance and incident response.

### Q14. How do you secure a model artifact?
**Answer:**
- Store it in a controlled registry or storage service.
- Restrict read/write access.
- Verify provenance and checksum.
- Use trusted serialization formats and controlled loading.
- Record version and dependencies.

### Q15. What should happen after a security incident?
**Answer:**
- Contain the issue.
- Preserve evidence securely.
- Notify designated owners.
- Assess impact.
- Restore a known-good service if needed.
- Document root cause and corrective action.

## 5. Rapid revision
- Least privilege: minimum permissions.
- Data poisoning: malicious training data.
- Model theft: unauthorized model extraction.
- Data minimization: collect only necessary data.
- Governance: ownership, intended use, review and accountability.
- Risk register: risk, impact, control and owner.
