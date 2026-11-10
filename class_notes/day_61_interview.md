# Day 61: Interview Questions — Model Selection and Experiment Tracking

**Phase:** 6 — Advanced ML and Practical Applications

## 1. Core questions

### Q1. What is model selection?
**Answer:**
- It is the process of choosing a model and configuration for a defined task.
- It considers validation performance and generalization.
- It also considers latency, cost, maintainability and interpretability.
- The decision should meet business requirements, not just maximize one score.

### Q2. Explain train, validation and test sets.
**Answer:**
- Training data fits model parameters.
- Validation data supports tuning and candidate selection.
- Test data estimates final performance after decisions are complete.
- Repeated test-set tuning makes the test result optimistic.

### Q3. What is experiment tracking?
**Answer:**
- Recording model runs in a structured and searchable way.
- It stores parameters, metrics, artifacts and run metadata.
- It supports comparison and reproducibility.
- MLflow is one commonly used tool.

### Q4. What should an ML experiment record contain?
**Answer:**
- Run ID and timestamp.
- Dataset version and code revision.
- Model type and hyperparameters.
- Validation metrics.
- Training duration and environment.
- Artifacts and notes.

### Q5. What is MLflow?
**Answer:**
- MLflow is an open-source platform for managing ML lifecycle workflows.
- Its tracking component records parameters, metrics and artifacts.
- It can log models and support model lifecycle workflows.
- Backend and artifact storage should be configured appropriately for team usage.

### Q6. Why record dataset and code versions?
**Answer:**
- A model result depends on both data and code.
- Without versions, the exact experiment may be difficult to reproduce.
- Version references help investigate changes.
- They support audits and collaboration.

## 2. Practical scenarios

### Q7. Model A has F1 0.84 and Model B has F1 0.83. Which do you choose?
**Answer:**
- Check cross-validation variation and statistical stability.
- Compare precision and recall trade-offs.
- Review latency, size, cost and interpretability.
- Select the model that best meets business constraints.
- A tiny score difference may not justify added complexity.

### Q8. A developer reports the best score but cannot identify the dataset used. What is the issue?
**Answer:**
- The experiment is not adequately traceable.
- Record dataset version or immutable checksum.
- Record code revision, split strategy and parameters.
- Re-run with reproducible inputs before relying on the result.

### Q9. Why should test metrics not be used for repeated tuning?
**Answer:**
- Repeated decisions become indirectly optimized to the test set.
- The test score no longer gives an independent final estimate.
- Use validation or cross-validation for model selection.
- Evaluate the test set once after choices are complete.

### Q10. Two experiments use different target definitions. Can their scores be compared?
**Answer:**
- Not directly.
- Target definitions change the task.
- Align business objective, population and label window first.
- Otherwise report them as separate experiments.

## 3. Project questions

### Q11. How would you organize an experiment comparison report?
**Answer:**
1. State objective and primary metric.
2. Document dataset version and split.
3. List candidate models and parameter settings.
4. Show validation metrics and variation.
5. Include runtime and artifact links.
6. Explain selection and limitations.

### Q12. What is the difference between a parameter and a metric?
**Answer:**
- A parameter is an input setting to a training run, such as max_depth.
- A metric measures a result, such as F1 or RMSE.
- Parameters describe how the run was configured.
- Metrics describe observed performance.

### Q13. Does MLflow automatically make an experiment reproducible?
**Answer:**
- No. It records information but cannot guarantee all dependencies and inputs are captured.
- Record code, data, environment and random seeds.
- Use reproducible packaging and versioned artifacts.
- Validate by rerunning the experiment.

## 4. Rapid revision
- Model selection: performance plus constraints.
- Validation: tuning and candidate selection.
- Test: final evaluation.
- Experiment tracking: parameters, metrics, artifacts and metadata.
- MLflow: tracking and model lifecycle tool.
- Dataset/code version: critical reproducibility context.
