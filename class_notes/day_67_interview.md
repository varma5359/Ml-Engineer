# Day 67: Interview Questions — Docker and Containerizing ML Applications

**Phase:** 7 — Deployment and MLOps

## 1. Core questions

### Q1. What is Docker?
**Answer:**
- A platform for building and running containers.
- It packages application code and runtime dependencies.
- It helps reduce environment differences.
- It does not remove the need for security and deployment configuration.

### Q2. What is the difference between an image and a container?
**Answer:**
- An image is a template.
- A container is an instance created from an image.
- Multiple containers can be created from one image.
- Container state is not automatically a durable data store.

### Q3. What is a Dockerfile?
**Answer:**
- A text file containing image build instructions.
- It defines base image, working directory, copied files and commands.
- Docker uses it to build an image.
- It should avoid copying secrets or unnecessary files.

### Q4. What is port mapping?
**Answer:**
- It connects a host port to a container port.
- Example: `-p 8000:8000`.
- The application inside the container must listen on the correct interface and port.
- EXPOSE alone does not publish a port.

### Q5. Why bind a FastAPI app to 0.0.0.0 in a container?
**Answer:**
- It listens on the container's network interfaces.
- Binding only to 127.0.0.1 can make it inaccessible from the host mapping.
- Port mapping then forwards traffic to the container port.
- Network access should still be restricted appropriately.

### Q6. What is .dockerignore?
**Answer:**
- It excludes files from the Docker build context.
- It can prevent virtual environments, Git history and secrets from being copied.
- It reduces build context size.
- It is not a substitute for secret management.

## 2. Scenario questions

### Q7. The container runs but the API cannot be reached. What do you check?
**Answer:**
- Is the app bound to 0.0.0.0?
- Is the correct port mapped?
- Is the process running?
- Check container logs and firewall settings.
- Verify the route locally inside and outside the container.

### Q8. The image works on one machine but not another. What do you investigate?
**Answer:**
- CPU architecture and base image.
- Dependency versions.
- Environment variables and mounted artifacts.
- File paths and permissions.
- Runtime configuration and resource limits.

### Q9. Why should secrets not be copied into an image?
**Answer:**
- Images may be pushed to registries and shared.
- Secrets can remain in image layers and history.
- Use a secret manager or secure runtime injection.
- Rotate any secret that was accidentally exposed.

### Q10. How would you reduce image size?
**Answer:**
- Use a slim maintained base image.
- Copy only required files.
- Use .dockerignore.
- Avoid unnecessary build tools in the runtime image.
- Consider multi-stage builds when dependencies require compilation.

## 3. Project questions

### Q11. Describe your Containerized Prediction Service.
**Answer:**
- I prepared an API and model artifact.
- I listed runtime dependencies.
- I wrote a Dockerfile and .dockerignore.
- I built a tagged image and ran a container with port mapping.
- I tested health and prediction endpoints and inspected logs.

### Q12. Does Docker guarantee reproducibility?
**Answer:**
- It improves environment consistency but does not guarantee complete reproducibility.
- Base images and dependencies should be pinned.
- Data and model artifacts need versioning.
- Build provenance and runtime configuration should be recorded.

### Q13. How do you persist data from a container?
**Answer:**
- Use volumes or external storage.
- Container writable layers are not a reliable long-term data store.
- Model artifacts may be baked into an immutable image or mounted from controlled storage.
- Choose based on update and security requirements.

## 4. Rapid revision
- Image: template.
- Container: running instance.
- Dockerfile: build instructions.
- Port mapping: host-to-container connection.
- .dockerignore: excludes build context files.
- Bind API to 0.0.0.0 in container.
- Never bake secrets into images.
