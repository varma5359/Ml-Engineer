# Day 81: Capstone — Candidate Model Training

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved mini-project / deliverable:** Model training suite

## 1. Learning objectives
- Select candidate algorithms appropriate for tabular churn data.
- Train multiple models using a shared preprocessing pipeline.
- Compare models using consistent validation data.
- Record hyperparameters, runtime and evaluation metrics.
- Explain trade-offs between performance, complexity and interpretability.

## 2. Why train candidate models?
A single algorithm is rarely guaranteed to be best for a new dataset. Candidate training compares a small, justified set of approaches under the same data split and evaluation rules.

For this capstone, consider:
- Logistic Regression: simple, fast and interpretable baseline.
- Random Forest: captures nonlinear interactions and handles mixed feature relationships after preprocessing.
- Gradient Boosting: sequentially builds learners to improve errors from earlier stages.

Avoid training many models without a reason. The objective is a defensible comparison, not the largest leaderboard.

## 3. Reuse the preprocessing pipeline
Use the fitted design from Day 79. Create an unfitted pipeline for each candidate so each model learns its own preprocessing only from the training fold.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

candidates = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=250, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, random_state=42
    )
}
```

If the preprocessing output is sparse because of one-hot encoding, some estimators may require dense input. Check estimator compatibility and memory requirements. Do not densify high-cardinality data without considering memory.

## 4. Train each candidate
```python
trained_models = {}

for name, estimator in candidates.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])
    pipeline.fit(X_train, y_train)
    trained_models[name] = pipeline
    print(name, "trained")
```

Every candidate receives the same training data and preprocessing definition. For robust comparison, use cross-validation on the training portion or a separate validation set.

## 5. Evaluate candidate models
```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import pandas as pd

rows = []
for name, model in trained_models.items():
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    rows.append({
        "model": name,
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, score) if score is not None else None,
        "pr_auc": average_precision_score(y_test, score) if score is not None else None
    })

candidate_results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
print(candidate_results.round(3))
```

During model development, replace X_test/y_test above with a validation set. Keep the final test set untouched until selection and tuning are complete.

## 6. Compare more than a single score
Consider:
- Precision and recall at a meaningful threshold.
- PR-AUC and ROC-AUC for ranking.
- Confusion matrix.
- Segment performance.
- Fit and prediction time.
- Model size and memory.
- Interpretability and operational complexity.

## 7. Record training configuration
```python
experiment_record = {
    "dataset_version": "record-your-data-version",
    "split_seed": 42,
    "target": "Churn",
    "positive_label": 1,
    "candidate_models": list(candidates.keys())
}
print(experiment_record)
```

Replace the placeholder dataset version with a real Git/DVC revision or checksum. Record dependency versions and code revision as well.

## 8. Approved deliverable — Model training suite

**Objective:** Train and compare multiple candidate models using the same approved feature and split strategy.

### Tasks
1. Reuse the Day 79 preprocessing design and Day 80 baseline.
2. Train at least three candidates, including a simple linear model and tree-based alternatives.
3. Use a consistent validation strategy.
4. Record metrics, hyperparameters and training time.
5. Compare model complexity and performance.
6. Identify a provisional candidate and explain why it deserves further evaluation.
7. Save training configuration and results.

**Suggested structure**
```text
customer_churn_capstone/
├── src/train_candidates.py
├── reports/candidate_comparison.csv
├── reports/training_notes.md
└── artifacts/candidates/
```

**Validation:** Confirm candidates use the same data split and target mapping. Avoid using final test results to choose the candidate. Explain missing metrics and estimator constraints.

## 9. AI-assisted implementation guide
| Level | Prompt |
|---|---|
| Hint | "Suggest one additional candidate model appropriate for tabular churn data." |
| Plan | "Break my model training suite into reusable functions." |
| Example | "Show a small loop that trains two sklearn classifiers." |
| Guided implementation | "Help me add the random forest candidate to my existing pipeline." |
| Explanation | "Explain why class_weight may help and what it does not solve." |
| Debugging | "Inspect this estimator compatibility error and guide me through the fix." |
| Review | "Review my candidate comparison for fair validation and reproducibility." |

## 10. Daily quiz

**Q1. Why train multiple candidate models?**  
A. To guarantee perfect accuracy  
B. To compare justified approaches under consistent conditions  
C. To avoid validation  
D. To remove the target  
**Correct: B.** Model selection requires evidence from comparable experiments.

**Q2. Which model is commonly used as a simple linear classification baseline?**  
A. Logistic Regression  
B. K-Means  
C. PCA  
D. DBSCAN  
**Correct: A.** Logistic regression is a common classification baseline.

**Q3. What is the main danger of comparing models on different splits?**  
A. Different UI output  
B. Data differences confound the comparison  
C. Models cannot be serialized  
D. Training becomes impossible  
**Correct: B.** A score difference may come from the sample rather than the model.

**Q4. Which metric is often useful for ranking under class imbalance?**  
A. PR-AUC  
B. R²  
C. MAE  
D. MSE  
**Correct: A.** Precision-recall behavior focuses on the positive class.

**Q5. What should guide the choice of final candidate?**  
A. Highest training accuracy only  
B. Newest algorithm  
C. Validation quality, business needs and operational trade-offs  
D. Most lines of code  
**Correct: C.** The best choice must fit the task and deployment constraints.

## 11. Interview preparation

### Q1. How would you compare Logistic Regression and Random Forest for churn?
**Answer:** Use the same data split, feature pipeline and validation metrics. Logistic Regression provides a simple interpretable reference, while Random Forest can capture nonlinear patterns and interactions. Compare precision, recall, PR-AUC, segment behavior, latency and complexity before selecting a candidate.

**Follow-up:** What if Random Forest improves PR-AUC but has much worse latency?

### Q2. Why is class_weight not a complete solution to class imbalance?
**Answer:** It changes the penalty applied to classes during training, but it does not create new information, guarantee calibrated probabilities or select the correct business threshold. Evaluate precision-recall behavior and consider other approaches only when justified.

**Follow-up:** How would you compare class weighting with resampling?

## 12. Completion checklist
- [ ] Train at least three candidate models.
- [ ] Use a consistent validation strategy.
- [ ] Compare classification metrics.
- [ ] Record parameters, versions and runtime.
- [ ] Explain the provisional candidate.
- [ ] Complete Model training suite.

**Next lesson:** Day 82 — Capstone: Evaluation and model selection.
