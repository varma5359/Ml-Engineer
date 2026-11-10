# Day 32: Supervised Learning Comparison and Model Selection

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Compare supervised algorithms fairly.
- Select metrics appropriate to the target and business cost.
- Understand cross-validation at a high level.
- Consider interpretability, latency and maintenance.
- Explain why the most accurate model is not always the best production choice.

## 1. What Is Model Selection?
Model selection is the process of comparing candidate algorithms and choosing one that meets the business and technical requirements on unseen data.

## 2. Common Models
| Algorithm | Typical strength | Important limitation |
|---|---|---|
| Linear/logistic regression | Simple and interpretable baseline | Limited by chosen feature representation |
| KNN | Simple similarity-based method | Sensitive to scaling and prediction cost |
| Decision tree | Rule-like explanations | Can overfit |
| Random forest | Often robust for tabular patterns | Less transparent than one tree |
| SVM | Effective in some high-dimensional settings | Scaling and parameter sensitivity |

## 3. Three Fair Comparison Rules
1. Use the same train/validation split or cross-validation folds.
2. Use the same relevant metric definitions.
3. Keep preprocessing leakage-safe and fit it within each fold.

## 4. Syntax: cross-validation
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
print(scores)
print(scores.mean())
```
Cross-validation estimates performance across several training/validation folds. Choose a scoring metric suited to the problem.

## 5. Scenario Questions
1. Model A has 94% accuracy and Model B has 90%, but B has much higher fraud recall. Which is better? It depends on false-negative cost and investigation capacity.
2. A model is accurate but too slow for live predictions. What matters? Latency and infrastructure constraints.
3. Two models have similar scores. What else helps choose? Stability, explainability, maintenance, cost and monitoring.
