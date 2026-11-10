# Day 76: Interview Questions — Production ML Architecture and Reliability

**Phase:** 7 — Deployment and MLOps

## 1. Core questions

### Q1. What is production ML architecture?
**Answer:**
- It describes the components and workflows needed to operate an ML system.
- It includes data, training, evaluation, registry, serving and monitoring.
- It also covers security, reliability and ownership.
- The model is only one component.

### Q2. Explain offline training and online inference.
**Answer:**
- Offline training uses historical data to prepare and evaluate model candidates.
- An approved artifact is registered and released.
- Online inference receives a request and generates a prediction.
- The serving workflow must use compatible features and model versions.

### Q3. What is the difference between availability and reliability?
**Answer:**
- Availability describes whether a service is accessible when needed.
- Reliability concerns consistent correct operation over time.
- A service can be available but return incorrect predictions.
- Both operational and model-quality monitoring matter.

### Q4. Explain latency and throughput.
**Answer:**
- Latency is the time taken to process a request.
- Throughput is the number of requests processed per unit time.
- Both should be measured under realistic load.
- Capacity planning balances performance and cost.

### Q5. What are liveness and readiness checks?
**Answer:**
- Liveness checks whether the process is alive.
- Readiness checks whether it can serve requests safely.
- Readiness can verify model and dependency availability.
- A failed readiness check should prevent traffic from reaching an unready instance.

### Q6. What is an SLO?
**Answer:**
- A Service Level Objective is a target for a service-level indicator.
- Examples include availability or latency targets.
- It guides reliability work and alerting.
- Targets should reflect user and business needs.

## 2. Scenario questions

### Q7. The API is responding, but predictions are incorrect. Is the service healthy?
**Answer:**
- Operationally it may be available, but model quality is not healthy.
- Verify feature schema and artifact version.
- Compare predictions with known test cases.
- Inspect data and preprocessing changes.
- Monitor predictive quality when labels arrive.

### Q8. Model storage is unavailable when a new instance starts. What should happen?
**Answer:**
- The instance should not report readiness if the model is required.
- Use bounded retries and clear failure logs.
- Keep existing healthy instances serving if the architecture allows.
- Restore storage access or deploy a known-good artifact through the approved path.

### Q9. A new model increases latency significantly. What do you do?
**Answer:**
- Compare latency under representative load.
- Inspect model size, preprocessing and resource use.
- Check whether the SLO is breached.
- Optimize or scale if justified.
- Roll back if user impact exceeds approved limits.

### Q10. How would you design rollback?
**Answer:**
- Keep a known-good model and image version.
- Record the deployed version and configuration.
- Validate artifact and schema compatibility.
- Switch traffic or configuration to the prior release.
- Monitor after rollback and record the incident.

### Q11. What happens if monitoring itself fails?
**Answer:**
- Treat observability as a production dependency.
- Alert on missing telemetry or pipeline heartbeat.
- Buffer or retry telemetry where safe.
- Ensure core service behavior does not depend unnecessarily on the monitoring pipeline.
- Define an operational fallback.

## 3. Project questions

### Q12. Describe your Production ML Architecture.
**Answer:**
- I separated offline training from online inference.
- I included data validation, feature preparation and model registry.
- I designed a versioned prediction service behind an API boundary.
- I added readiness, operational monitoring and model-quality monitoring.
- I documented failure modes, security and rollback.

### Q13. How do you make a production ML system reproducible?
**Answer:**
- Version code, data, configuration and model artifacts.
- Record dependencies and training parameters.
- Link model registry versions to experiment runs.
- Use container images and automated validation.
- Preserve deployment metadata.

### Q14. How do you choose batch or online inference?
**Answer:**
- Consider response-time needs and prediction freshness.
- Batch is suitable for scheduled large-scale scoring.
- Online is suitable for interactive requests.
- Compare infrastructure cost, volume and reliability requirements.

### Q15. What should be included in an operational runbook?
**Answer:**
- Service architecture and owners.
- Health and readiness endpoints.
- Dashboards and alert definitions.
- Common failure symptoms and diagnostics.
- Rollback and recovery steps.
- Escalation and incident communication.

## 4. Rapid revision
- Offline: train and evaluate.
- Online: serve predictions.
- Availability: accessible.
- Reliability: consistently correct operation.
- Latency: response time.
- Throughput: requests per unit time.
- Liveness: process alive.
- Readiness: able to serve.
- SLO: measurable service target.
- Rollback: restore a known-good release.
