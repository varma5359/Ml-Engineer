# Day 67: Docker and Containerizing ML Applications

**Pathway:** Machine Learning Engineer  
**Phase:** 7 — Deployment and MLOps  
**Level:** Beginner to Intermediate  
**Estimated time:** 2–3 hours  
**Approved mini-project:** Containerized Prediction Service

## 1. Learning objectives
- Explain containers and images.
- Distinguish an image from a running container.
- Write a Dockerfile for a Python prediction API.
- Build and run an image with port mapping.
- Identify basic container security and reproducibility practices.

## 2. Why Docker for ML?
An ML application depends on Python, libraries, model artifacts and system packages. A container packages the application and its runtime dependencies into a portable image.

**Flow:** Source code + dependencies + model artifact → Docker image → running container → API endpoint.

A container shares the host kernel; it is not the same as a full virtual machine.

## 3. Important Docker terms

| Term | Meaning |
|---|---|
| Image | Read-only template used to create containers |
| Container | Running or stopped instance of an image |
| Dockerfile | Instructions for building an image |
| Registry | Stores and distributes images |
| Volume | Persistent or shared storage |
| Port mapping | Connects a host port to a container port |

## 4. Prepare a small FastAPI application
Suggested project:
```text
prediction_service/
├── app/
│   └── main.py
├── artifacts/
│   └── model.joblib
├── requirements.txt
└── Dockerfile
```

Example `app/main.py`:
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Containerized ML API")

class InputData(BaseModel):
    value: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    # Replace this illustrative calculation with a trusted model pipeline.
    return {"prediction": data.value * 2}
```

This endpoint demonstrates containerization only; it is not a trained ML model. For the project, replace it with your fitted pipeline and matching schema.

## 5. Create requirements.txt
```text
fastapi
uvicorn[standard]
pydantic
scikit-learn
joblib
pandas
```

For a reproducible build, pin versions tested by the project and consider a lock file.

## 6. Write a Dockerfile
Create a file named `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY artifacts/ ./artifacts/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Explanation:**
- `FROM`: selects the base image.
- `WORKDIR`: sets the working directory.
- `COPY`: copies files into the image.
- `RUN`: executes a build-time command.
- `EXPOSE`: documents the application port; it does not publish it.
- `CMD`: default process when the container starts.

## 7. Add a .dockerignore file
```text
.git
.venv
__pycache__
*.pyc
.env
.ipynb_checkpoints
data/private/
```

Do not copy credentials, local virtual environments or private training data into the image.

## 8. Build the image
From the project directory:
```bash
docker build -t ml-prediction-service:1.0 .
```

The final dot means the current directory is the build context.

## 9. Run the container
```bash
docker run --name ml-api -p 8000:8000 ml-prediction-service:1.0
```

Open `http://localhost:8000/health`. The host port 8000 maps to the container port 8000.

In another terminal:
```bash
docker ps
docker logs ml-api
```

Stop and remove:
```bash
docker stop ml-api
docker rm ml-api
```

## 10. Test the API
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"ok"}
```

On Windows PowerShell, `curl` may be an alias; use `curl.exe` or Invoke-RestMethod if needed.

## 11. Container best practices
- Use a small, maintained base image.
- Pin dependencies and preferably pin base image digests for controlled releases.
- Run as a non-root user where practical.
- Do not bake secrets into images.
- Scan images and dependencies.
- Use health/readiness checks in the deployment platform.
- Tag images with meaningful versions; avoid relying only on `latest`.
- Keep model artifact version and image version traceable.

## 12. Approved mini-project — Containerized Prediction Service

**Scenario:** A team wants the same prediction API to run consistently on developer machines and a deployment server.

**Requirements**
1. Prepare a working API and trusted model artifact.
2. Create requirements and Dockerfile.
3. Add .dockerignore.
4. Build a tagged image.
5. Run the container with port mapping.
6. Test health and prediction endpoints.
7. Inspect logs and stop/remove the container.
8. Document image tag, artifact version and environment assumptions.

**Validation:** Build from a clean context, verify endpoints, test invalid input and confirm no secrets or private datasets were copied into the image.

## 13. Common mistakes
| Mistake | Fix |
|---|---|
| App binds to 127.0.0.1 inside container | Bind to 0.0.0.0 |
| EXPOSE assumed to publish port | Use `-p host:container` |
| Missing model file | Include or mount the correct artifact |
| Copy .env into image | Use secure runtime secret injection |
| Use unpinned dependencies | Pin tested versions |
| Container exits immediately | Inspect command and logs |

## 14. Completion checklist
- [ ] Explain image and container.
- [ ] Write a Dockerfile.
- [ ] Build and run a container.
- [ ] Test the API through a mapped port.
- [ ] Complete Containerized Prediction Service.

**Next lesson:** Day 68 — MLflow experiment tracking fundamentals.
