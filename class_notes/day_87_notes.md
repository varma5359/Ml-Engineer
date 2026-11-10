# Day 87: Capstone — Docker and Deployment Preparation

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved mini-project / deliverable:** Containerized ML system

## 1. Learning objectives
- Explain why the capstone services should be containerized.
- Create Dockerfiles for API and UI services.
- Configure service communication and environment variables.
- Build and run a multi-service application with Docker Compose.
- Verify health, logs, networking and artifact availability.

## 2. What are we containerizing?
The capstone has two application components:
- FastAPI inference service.
- Streamlit prediction interface.

Both use the same agreed request/response contract. The API loads the approved model artifact. The UI calls the API over the Compose network.

## 3. Suggested folder structure
```text
customer_churn_capstone/
├── api/
│   ├── main.py
│   └── schemas.py
├── app/
│   └── ui.py
├── artifacts/
│   └── churn_pipeline.joblib
├── requirements-api.txt
├── requirements-ui.txt
├── Dockerfile.api
├── Dockerfile.ui
├── compose.yaml
└── .dockerignore
```

Adjust paths to your actual project. Avoid placing private data or credentials in the image build context.

## 4. API requirements
Example `requirements-api.txt`:
```text
fastapi
uvicorn[standard]
pydantic
pandas
scikit-learn
joblib
```

Pin versions that have been tested with the serialized model. A model artifact may not be compatible across arbitrary scikit-learn versions.

## 5. Create API Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /service

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY artifacts/ ./artifacts/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

For production, consider a non-root user, pinned base image digest, vulnerability scanning and read-only filesystem where compatible.

## 6. Create UI Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /service

COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt

COPY app/ ./app/

EXPOSE 8501
CMD ["streamlit", "run", "app/ui.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

## 7. Configure the UI to reach the API
Inside Docker Compose, the UI should call the API service by its service name, not localhost.

For example:
```python
import os
API_URL = os.getenv("API_URL", "http://api:8000/predict")
```

In a local non-container run, override the URL to the local API address. Do not commit credentials or sensitive configuration.

## 8. Create compose.yaml
```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]
      interval: 20s
      timeout: 5s
      retries: 3

  ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    ports:
      - "8501:8501"
    environment:
      API_URL: "http://api:8000/predict"
    depends_on:
      api:
        condition: service_healthy
```

Compose health-check support and dependency behavior depend on the Compose implementation. Also ensure the API has a health endpoint that checks required model readiness, not merely that the process exists.

## 9. Add .dockerignore
```text
.git
.venv
__pycache__
*.pyc
.env
.ipynb_checkpoints
data/private/
notebooks/
```

Only exclude files that are not required by the build. If the model artifact is copied into the API image, ensure it is intentionally included and approved for distribution.

## 10. Build and start
```bash
docker compose build
docker compose up
```

Open:
- UI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

Inspect:
```bash
docker compose ps
docker compose logs api
docker compose logs ui
```

Stop:
```bash
docker compose down
```

Do not use volume deletion flags unless you intentionally want to remove persisted data.

## 11. Deployment preparation checklist
- Model artifact and dependencies are compatible.
- API listens on 0.0.0.0 inside the container.
- UI uses service DNS for API calls.
- Ports and health checks are correct.
- Secrets are injected securely at runtime.
- Images are tagged and scanned.
- Logs do not expose sensitive customer data.
- Deployment and rollback steps are documented.

## 12. Approved deliverable — Containerized ML system

**Objective:** Run the capstone UI and inference API as containerized services.

### Tasks
1. Prepare API and UI dependency files.
2. Write separate Dockerfiles.
3. Configure the UI API URL using an environment variable.
4. Create Docker Compose configuration.
5. Add a health check that reflects API readiness.
6. Build and start both services.
7. Test one valid and one invalid prediction.
8. Inspect logs and stop the stack.
9. Document image tags, artifact version and deployment assumptions.

**Suggested structure**
```text
customer_churn_capstone/
├── Dockerfile.api
├── Dockerfile.ui
├── compose.yaml
├── .dockerignore
└── docs/deployment.md
```

**Validation:** Confirm the UI can reach the API by service name, model artifact is present, invalid requests are rejected, and no secrets/private data are baked into images.

## 13. AI-assisted implementation guide
| Level | Prompt |
|---|---|
| Hint | "Give me one hint for API_URL inside a Compose network." |
| Plan | "Break multi-service containerization into build, network and test steps." |
| Example | "Show a small Compose file with API and UI services." |
| Guided implementation | "Review my API Dockerfile before helping me write the UI Dockerfile." |
| Explanation | "Explain why localhost inside the UI container is not the API container." |
| Debugging | "Help me debug this service-name connection error from Streamlit." |
| Review | "Review my container setup for secrets, ports, health checks and artifact compatibility." |

## 14. Daily quiz

**Q1. How should the UI container reach the API container in Compose?**  
A. Use the API service name and container port  
B. Use the UI's localhost  
C. Use a public IP always  
D. Read the model file directly  
**Correct: A.** Compose provides service-name DNS on its network.

**Q2. Why bind Uvicorn to 0.0.0.0 in a container?**  
A. To listen on container interfaces  
B. To change model labels  
C. To enable training  
D. To remove authentication  
**Correct: A.** Binding only to loopback can prevent external container networking.

**Q3. What does Docker Compose help manage?**  
A. Multiple related services  
B. Only pandas DataFrames  
C. Only model metrics  
D. Only Git branches  
**Correct: A.** Compose defines and runs a multi-container application.

**Q4. What should an API health check verify?**  
A. Only that a Dockerfile exists  
B. Required service and model readiness  
C. That the UI title is correct  
D. That a dataset has no rows  
**Correct: B.** Readiness should reflect the ability to serve predictions.

**Q5. Where should runtime secrets be provided?**  
A. Hardcoded in Dockerfile  
B. Secure runtime secret configuration  
C. In the model prediction response  
D. In a public Git commit  
**Correct: B.** Secrets should not be baked into image layers or source control.

## 15. Interview preparation

### Q1. How do containers communicate in Docker Compose?
**Answer:** Compose creates a network and provides service-name DNS. The UI container can call the API using its service name and internal container port. Host port mappings are for access from outside the Compose network.

**Follow-up:** Why does localhost inside one container not refer to another container?

### Q2. What is the difference between a Docker health check and application readiness?
**Answer:** A Docker health check executes a configured test and reports container health. Application readiness should verify that the service can actually handle requests, including required model and dependency availability. The check should represent the true serving condition.

**Follow-up:** What should happen if the model artifact cannot load?

## 16. Completion checklist
- [ ] Create API and UI Dockerfiles.
- [ ] Configure Compose networking.
- [ ] Build and run both services.
- [ ] Verify health and prediction endpoints.
- [ ] Inspect logs and stop containers safely.
- [ ] Complete Containerized ML system.

**Next lesson:** Day 88 — Capstone: Experiment tracking and versioning.
