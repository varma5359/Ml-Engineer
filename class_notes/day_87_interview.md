# Day 87: Interview Questions — Capstone Docker and Deployment Preparation

**Phase:** 8 — End-to-End ML Capstone

## 1. Core questions

### Q1. Why containerize the capstone?
**Answer:**
- Package application and runtime dependencies.
- Reduce differences between developer and deployment environments.
- Run API and UI as separate services.
- Support repeatable builds and releases.

**Follow-up:** Does a container guarantee complete reproducibility?

### Q2. Why use separate containers for API and UI?
**Answer:**
- They have different responsibilities and dependencies.
- They can be deployed and scaled independently.
- A single API can serve multiple client applications.
- Failures can be isolated more clearly.

**Follow-up:** When might a single container be acceptable for a prototype?

### Q3. How do services communicate in Docker Compose?
**Answer:**
- Compose creates a shared network.
- Services can resolve each other by service name.
- The client uses the target service's internal port.
- Host port mapping is for access from outside the network.

**Follow-up:** Why is localhost usually wrong for container-to-container communication?

### Q4. What is a Docker health check?
**Answer:**
- A configured command that reports container health.
- It can test whether a service endpoint responds.
- A useful model API check should reflect model readiness.
- It supports orchestration and startup coordination.

**Follow-up:** What is the difference between liveness and readiness?

### Q5. Why pin Python dependencies?
**Answer:**
- Dependency changes can alter runtime behavior.
- Serialized scikit-learn artifacts may be version-sensitive.
- Pinned versions make builds more controlled.
- Upgrade dependencies through tested changes.

**Follow-up:** What additional information helps reproduce an image build?

## 2. Scenario questions

### Q6. Streamlit reports connection refused when calling the API. What do you check?
**Answer:**
- API container status and logs.
- Service name and internal port.
- API binding to 0.0.0.0.
- Network membership and health.
- Environment variable value in the UI container.

**Follow-up:** Why might the host port not be needed for UI-to-API traffic?

### Q7. The API starts but fails to load the model. What do you investigate?
**Answer:**
- Artifact copy or mount path.
- File permissions.
- Dependency versions.
- Artifact integrity and trust.
- Model feature schema and serialization compatibility.
- Readiness behavior.

**Follow-up:** Should the service return a default prediction?

### Q8. A secret is found in an image layer. What do you do?
**Answer:**
- Revoke or rotate the secret immediately.
- Assess whether the image was pushed or accessed.
- Rebuild without the secret.
- Use secure runtime secret injection.
- Follow incident and image-retention policy.

**Follow-up:** Why may deleting the current file not remove it from earlier image layers?

## 3. Project questions

### Q9. Describe your Containerized ML system.
**Answer:**
- I created separate API and UI images.
- I configured the UI to call the API using Compose service DNS.
- I added an API health check and environment configuration.
- I built and ran the services.
- I tested predictions, validation errors and logs.

**Follow-up:** How would you deploy the same images to a server?

### Q10. What is the purpose of .dockerignore?
**Answer:**
- It excludes unnecessary files from the build context.
- It helps reduce image build size.
- It prevents accidental inclusion of local environments and secrets.
- It should be reviewed so required artifacts are not excluded.

**Follow-up:** Does .dockerignore replace secret management?

## 4. Rapid revision
- Image: container template.
- Container: running image instance.
- Compose: multi-service definition.
- Service DNS: service-name networking.
- Internal port: container-to-container port.
- Health check: configured service test.
- .dockerignore: excludes build-context files.
