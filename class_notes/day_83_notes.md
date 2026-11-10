# Day 83: Capstone — Tuning and Validation

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved mini-project / deliverable:** Optimized model

## 1. Learning objectives
- Explain hyperparameter tuning and its purpose.
- Use cross-validation for model selection.
- Configure a bounded GridSearchCV or RandomizedSearchCV.
- Select a model using a justified scoring metric.
- Validate the selected configuration without test leakage.

## 2. What is hyperparameter tuning?
A hyperparameter is a setting chosen before or during model fitting, such as tree depth or regularization strength. Tuning searches a defined set of settings to find a configuration that performs well under a validation strategy.

Tuning does not guarantee better real-world performance. Large searches can overfit the validation process and consume unnecessary compute.

## 3. Why cross-validation?
K-fold cross-validation divides training data into folds. Each fold is used for validation once while the remaining folds train the model.

For classification, StratifiedKFold aims to preserve class proportions in each fold. If data contains time dependence or repeated customers, use a suitable time-aware or group-aware strategy instead.

## 4. Define a parameter grid
Example with a Random Forest pipeline:
```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42, n_jobs=-1, class_weight="balanced"
    ))
])

param_grid = {
    "classifier__n_estimators": [150, 250],
    "classifier__max_depth": [None, 8, 14],
    "classifier__min_samples_leaf": [2, 5]
}
```

The double underscore syntax addresses a nested pipeline parameter: step name followed by parameter name.

## 5. Run GridSearchCV
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)
print("Best CV PR-AUC:", search.best_score_)
best_model = search.best_estimator_
```

The grid contains 2 × 3 × 2 = 12 configurations. With five folds, that requires 60 fits. Estimate compute cost before expanding the search.

## 6. Inspect tuning results
```python
import pandas as pd

tuning_results = pd.DataFrame(search.cv_results_)
columns = [
    "rank_test_score", "mean_test_score", "std_test_score",
    "param_classifier__n_estimators",
    "param_classifier__max_depth",
    "param_classifier__min_samples_leaf"
]
print(tuning_results[columns].sort_values("rank_test_score").head(10))
```

Do not select solely by the highest mean score if the difference is tiny or unstable. Consider variation, complexity and practical constraints.

## 7. Validate the tuned candidate
- Compare tuned candidate against the baseline and untuned model.
- Inspect cross-validation variation.
- Check performance on a separate validation set if one is part of the approved design.
- Freeze the model and threshold before the final test evaluation.
- Do not repeatedly inspect test metrics to choose parameters.

## 8. Avoid common tuning errors
- Fitting preprocessing before cross-validation.
- Using the test set as GridSearchCV input.
- Choosing a scoring metric unrelated to the business task.
- Searching huge grids without compute estimates.
- Ignoring variance across folds.
- Forgetting to record the selected configuration.

## 9. Approved deliverable — Optimized model

**Objective:** Tune the selected candidate using a reproducible validation strategy.

### Tasks
1. Select the candidate algorithm from Day 81–82.
2. Define a small justified hyperparameter search space.
3. Choose a suitable cross-validation strategy.
4. Select a scoring metric aligned with churn and class balance.
5. Run GridSearchCV or RandomizedSearchCV.
6. Record best parameters, mean score and fold variation.
7. Compare tuned and untuned model on validation evidence.
8. Save the fitted candidate and configuration.
9. Document compute cost and limitations.

**Suggested structure**
```text
customer_churn_capstone/
├── src/tune_model.py
├── reports/tuning_report.md
├── config/search_space.json
└── artifacts/optimized_model/
```

**Validation:** Ensure preprocessing is inside the estimator pipeline, folds are appropriate for the data, and the final test set was not used in search.

## 10. AI-assisted implementation guide
| Level | Prompt |
|---|---|
| Hint | "Give me one hint for choosing a small hyperparameter search space." |
| Plan | "Plan a cross-validation tuning workflow for my selected churn model." |
| Example | "Show how double underscore accesses a Pipeline parameter." |
| Guided implementation | "Help me configure GridSearchCV one parameter at a time." |
| Explanation | "Explain why preprocessing must be inside the cross-validation pipeline." |
| Debugging | "Help me understand this failed fit warning from GridSearchCV." |
| Review | "Review my tuning protocol for leakage, cost and metric alignment." |

## 11. Daily quiz

**Q1. What is a hyperparameter?**  
A. A setting selected for the learning algorithm  
B. A target label  
C. A customer ID  
D. A test prediction  
**Correct: A.** Hyperparameters configure how the estimator learns.

**Q2. Why use StratifiedKFold for classification?**  
A. It sorts features alphabetically  
B. It aims to preserve class proportions in folds  
C. It removes all positive cases  
D. It serializes the model  
**Correct: B.** Stratification helps maintain class distribution.

**Q3. What does classifier__max_depth mean in a Pipeline?**  
A. A new target column  
B. max_depth parameter of the classifier step  
C. A dataset path  
D. A model registry stage  
**Correct: B.** Double underscore addresses nested estimator parameters.

**Q4. Where should GridSearchCV fit preprocessing?**  
A. On the complete dataset before splitting  
B. Inside the pipeline passed to cross-validation  
C. Only on the test set  
D. Nowhere  
**Correct: B.** This ensures transformations are fitted within each training fold.

**Q5. What should remain untouched during tuning?**  
A. Training set  
B. Final test set  
C. Search grid  
D. Feature names  
**Correct: B.** The final test set is reserved for final evaluation.

## 12. Interview preparation

### Q1. Explain GridSearchCV.
**Answer:** GridSearchCV evaluates specified hyperparameter combinations using cross-validation. It selects the configuration with the best average score under the chosen metric and refits the best estimator on the full training input.

**Follow-up:** When would RandomizedSearchCV be preferable?

### Q2. How do you prevent data leakage during cross-validation?
**Answer:** Put preprocessing and the estimator in one pipeline and pass that pipeline into the cross-validation search. Each fold fits transformations only on its training portion and evaluates on its held-out fold.

**Follow-up:** How would you adapt cross-validation for time-dependent data?

## 13. Completion checklist
- [ ] Define a justified search space.
- [ ] Select suitable cross-validation.
- [ ] Run bounded hyperparameter search.
- [ ] Record best parameters and fold variation.
- [ ] Confirm no test-set tuning.
- [ ] Complete Optimized model.

**Next lesson:** Day 84 — Capstone: Explainability and business interpretation.
