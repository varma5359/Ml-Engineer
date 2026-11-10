# Day 70: DVC and Dataset/Model Versioning

**Pathway:** Machine Learning Engineer  
**Phase:** 7 — Deployment and MLOps  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Approved mini-project:** Versioned ML Dataset

## 1. Learning objectives
- Explain why datasets need version control.
- Distinguish Git source control from DVC data versioning.
- Initialize DVC and track a dataset.
- Record pipeline stages and reproduce them.
- Explain remote storage and model artifact versioning.

## 2. Why use DVC?
Git is excellent for source code and small text files, but large datasets and model artifacts are often better handled with specialized storage.

DVC (Data Version Control) works with Git to track data and ML pipeline metadata while storing large file contents in a cache and optional remote storage.

**Flow:** Data file → DVC tracking metadata → Git commit → DVC remote for large content.

DVC does not replace Git. Both are used together.

## 3. Install and initialize
```bash
python -m pip install dvc
git init
dvc init
```

If the repository already has Git and DVC initialized, do not repeat initialization. Check:
```bash
git status
dvc status
```

## 4. Track a dataset
Assume the file is `data/customers.csv`.

```bash
dvc add data/customers.csv
```

DVC creates a small pointer file such as `data/customers.csv.dvc` and adds the actual data to its local cache.

Commit the pointer and ignore changes:
```bash
git add data/customers.csv.dvc data/.gitignore
git commit -m "Track customer dataset with DVC"
```

Do not add the large raw dataset directly to Git if DVC is intended to manage it.

## 5. Configure remote storage
A remote can be local, S3-compatible or another supported backend. Example local remote:

```bash
mkdir dvc-storage
dvc remote add -d localstorage ../dvc-storage
dvc remote list
```

Push data:
```bash
dvc push
```

A local remote is useful for a demonstration. For team collaboration, use approved shared storage with access controls and backups. Never commit cloud credentials.

## 6. Retrieve a tracked version
After checking out a Git commit that contains the desired .dvc pointer:
```bash
dvc pull
```

DVC uses the pointer metadata to retrieve the corresponding data from the configured remote/cache.

## 7. Create a simple reproducible pipeline
Create `src/preprocess.py`:
```python
import pandas as pd

data = pd.read_csv("data/customers.csv")
data = data.drop_duplicates()
data = data.fillna(0)
data.to_csv("data/cleaned.csv", index=False)
print("Rows after cleaning:", len(data))
```

Run it through DVC:
```bash
dvc stage add -n preprocess -d src/preprocess.py -d data/customers.csv -o data/cleaned.csv python src/preprocess.py
```

This records dependencies, outputs and command in `dvc.yaml`. Adapt the cleaning logic to the dataset; filling every missing value with zero is only an illustrative operation and may be statistically inappropriate.

Reproduce:
```bash
dvc repro
dvc status
```

## 8. Track a model artifact
A trained model can also be stored with DVC:
```bash
dvc add models/model.joblib
git add models/model.joblib.dvc models/.gitignore
git commit -m "Version trained model artifact"
dvc push
```

Keep model metadata and evaluation reports alongside a clear Git revision.

## 9. Useful commands
| Command | Purpose |
|---|---|
| `dvc add file` | Start tracking a file |
| `dvc push` | Upload cached data to remote |
| `dvc pull` | Download data for current Git revision |
| `dvc status` | Check pipeline/data changes |
| `dvc repro` | Reproduce pipeline stages |
| `dvc dag` | Display pipeline dependency graph |
| `dvc remote list` | List configured remotes |

## 10. Approved mini-project — Versioned ML Dataset

**Scenario:** A team updates a training dataset monthly and must reproduce which data version produced each model.

**Requirements**
1. Create a Git repository and initialize DVC.
2. Add a dataset with DVC.
3. Configure an approved remote.
4. Commit DVC pointer metadata to Git.
5. Create one preprocessing stage in `dvc.yaml`.
6. Reproduce the stage using `dvc repro`.
7. Track a model artifact or processed dataset.
8. Demonstrate retrieving a previous data version.
9. Document Git commit, DVC file and remote relationship.

**Suggested structure**
```text
versioned_ml_dataset/
├── data/
├── src/preprocess.py
├── models/
├── dvc.yaml
├── dvc.lock
└── README.md
```

**Validation:** Clone/check out the project in a clean location, run `dvc pull`, reproduce the pipeline and compare output checksums. Confirm the remote is accessible and credentials are not in Git.

## 11. Common mistakes
- Commit large data directly to Git unintentionally.
- Forget to commit .dvc pointer or dvc.yaml.
- Run dvc pull without the correct remote access.
- Change data without updating DVC metadata.
- Assume Git alone contains the dataset.
- Store remote credentials in source files.

## 12. Completion checklist
- [ ] Explain Git and DVC responsibilities.
- [ ] Track a dataset.
- [ ] Configure and use a remote.
- [ ] Reproduce a pipeline stage.
- [ ] Complete Versioned ML Dataset.

**Next lesson:** Day 71 — CI/CD for ML workflows.
