# Day 75: ML Security, Privacy and Governance

**Pathway:** Machine Learning Engineer  
**Phase:** 7 — Deployment and MLOps  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Approved mini-project:** ML Risk Assessment

## 1. Learning objectives
- Identify security and privacy risks across the ML lifecycle.
- Explain least privilege and secrets management.
- Recognize data poisoning, model theft and unsafe artifact risks.
- Build a basic ML risk register.
- Define governance, review and incident-response controls.

## 2. Why ML security and governance matter
An ML system processes data, produces decisions and often connects to APIs, storage and deployment infrastructure. Risks can occur during data collection, training, artifact storage, inference and monitoring.

Security protects systems and information. Privacy governs responsible handling of personal data. Governance defines ownership, accountability, approval and acceptable use.

## 3. Common ML risks

| Risk | Example | Possible control |
|---|---|---|
| Data exposure | Sensitive records in logs | Minimize, redact and restrict logs |
| Data poisoning | Malicious or corrupted training records | Provenance, validation and review |
| Model theft | Unauthorized access to model endpoint/artifact | Authentication and access control |
| Adversarial input | Crafted inputs manipulate prediction | Input validation and robustness tests |
| Unsafe artifact | Untrusted serialized model loaded | Trusted registry and integrity checks |
| Supply-chain risk | Compromised dependency | Pin, scan and review dependencies |
| Model misuse | Prediction used outside intended purpose | Usage policy and oversight |

## 4. Security controls

### Least privilege
Give each service and user only the permissions required for their task. Separate training, registry and inference identities where practical.

### Secrets management
- Do not hardcode API keys or passwords.
- Use environment injection or a secret manager.
- Restrict who can read secrets.
- Rotate exposed credentials.

### Artifact security
- Store artifacts in controlled repositories.
- Verify provenance and checksums.
- Do not load untrusted pickle/joblib files.
- Record artifact and dependency versions.

### API security
- Authenticate and authorize users.
- Use HTTPS.
- Validate request schema and size.
- Apply rate limits.
- Avoid sensitive data in error messages and logs.

## 5. Privacy and data minimization
Before using data, ask:
- Is each field necessary for the stated purpose?
- Is there an approved basis and permission to process it?
- Can identifiers be removed or pseudonymized?
- Who can access the data?
- How long should it be retained?
- How can deletion and access requests be handled?

Pseudonymized data can still be personal data if it can be re-linked. Follow applicable law and organizational policy; this lesson is not legal advice.

## 6. Governance essentials
Document:
- Model owner and business owner.
- Intended use and prohibited use.
- Training data sources and limitations.
- Validation and acceptance criteria.
- Human oversight and escalation.
- Version and change history.
- Monitoring and incident response.
- Review and retirement conditions.

## 7. Create a simple risk register
```python
import pandas as pd

risks = pd.DataFrame([
    {
        "risk": "Sensitive data in application logs",
        "likelihood": "Medium",
        "impact": "High",
        "control": "Redaction and restricted log access",
        "owner": "Platform team"
    },
    {
        "risk": "Untrusted model artifact loaded",
        "likelihood": "Low",
        "impact": "Critical",
        "control": "Trusted registry and provenance checks",
        "owner": "ML engineering"
    },
    {
        "risk": "Feature drift",
        "likelihood": "Medium",
        "impact": "Medium",
        "control": "Drift and performance monitoring",
        "owner": "Model owner"
    }
])
print(risks)
```

Likelihood and impact values are illustrative. A real assessment must be based on the system context and risk method.

## 8. Risk scoring example
A simple qualitative score can prioritize discussion:
```python
severity = {
    ("Low", "Low"): "Low",
    ("Low", "High"): "Medium",
    ("Medium", "High"): "High",
    ("High", "High"): "Critical"
}

print(severity[("Medium", "High")])
```

This mapping is deliberately partial and illustrative. Organizations should use their approved risk matrix rather than this example.

## 9. Incident response
If a model or data incident occurs:
1. Identify and contain the affected service or credential.
2. Preserve logs and relevant evidence securely.
3. Assess affected data, models and users.
4. Notify the designated security/privacy owners.
5. Restore a known-good model or service where appropriate.
6. Document root cause and corrective actions.
7. Review whether controls need improvement.

## 10. Approved mini-project — ML Risk Assessment

**Scenario:** A company is preparing a customer prediction model for internal use and needs to assess security, privacy and governance risks.

**Requirements**
1. Describe the system, users, data and model outputs.
2. Identify at least eight realistic risks.
3. Rate likelihood and impact using a stated method.
4. Propose a preventive or detective control for each risk.
5. Assign an owner and review frequency.
6. Document data minimization, retention and access controls.
7. Define intended and prohibited uses.
8. Write a short incident-response outline.

**Suggested structure**
```text
ml_risk_assessment/
├── docs/system_overview.md
├── docs/risk_register.csv
├── docs/governance.md
└── README.md
```

**Validation:** Each risk has a clear scenario, impact, control and owner. Distinguish implemented controls from proposed controls. Do not claim compliance merely because a checklist exists.

## 11. Common mistakes
- Treating pseudonymized data as automatically anonymous.
- Logging raw sensitive inputs for debugging.
- Giving broad permissions to every service.
- Loading artifacts from unknown sources.
- Ignoring dependency and registry security.
- Treating governance as paperwork without owners and review.

## 12. Completion checklist
- [ ] Identify ML lifecycle risks.
- [ ] Explain least privilege and secrets management.
- [ ] Create a risk register.
- [ ] Define privacy and governance controls.
- [ ] Complete ML Risk Assessment.

**Next lesson:** Day 76 — Production ML architecture and reliability.
