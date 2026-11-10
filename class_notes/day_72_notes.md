# Day 72: Model Monitoring and Data Drift

**Pathway:** Machine Learning Engineer  
**Phase:** 7 — Deployment and MLOps  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Approved mini-project:** Data Drift Detector

## 1. Learning objectives
- Explain why deployed models need monitoring.
- Distinguish data drift, concept drift and prediction drift.
- Compare reference and production feature distributions.
- Calculate simple numerical drift indicators.
- Explain alert thresholds and investigation steps.

## 2. Why monitor a deployed model?
A model is trained on historical data, but production data can change. Customer behavior, products, devices and business processes evolve. Monitoring helps identify changes that may reduce model reliability.

Monitoring should combine statistical signals with domain context. A drift alert is a reason to investigate, not automatic proof that model accuracy has fallen.

## 3. Types of change

| Term | Meaning | Example |
|---|---|---|
| Data drift | Input feature distribution changes | Customer age distribution shifts |
| Concept drift | Relationship between inputs and target changes | Customers respond differently to price |
| Prediction drift | Model output distribution changes | More records receive high-risk scores |
| Data quality issue | Input violates expected schema/quality | Missing feature rate rises |

Data drift can occur without concept drift, and concept drift may not be obvious from input distributions alone.

## 4. Establish a reference dataset
Use a representative, validated training or baseline dataset as a reference. Record its version, time window, schema and population. Compare production windows with a compatible reference.

## 5. Create example data
```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
reference = pd.DataFrame({
    "age": rng.normal(38, 10, 1000).clip(18, 85),
    "monthly_charge": rng.normal(70, 15, 1000).clip(10, 150)
})
production = pd.DataFrame({
    "age": rng.normal(45, 12, 1000).clip(18, 85),
    "monthly_charge": rng.normal(82, 18, 1000).clip(10, 150)
})
print(reference.describe())
print(production.describe())
```

The data is synthetic and deliberately shifted to demonstrate detection.

## 6. Compare basic statistics
```python
comparison = pd.DataFrame({
    "reference_mean": reference.mean(),
    "production_mean": production.mean(),
    "reference_std": reference.std(),
    "production_std": production.std()
})
comparison["mean_difference"] = (
    comparison["production_mean"] - comparison["reference_mean"]
)
print(comparison.round(2))
```

A mean difference alone can miss distribution changes. Inspect quantiles, missingness and distribution shape too.

## 7. Visualize distributions
```python
import matplotlib.pyplot as plt

for column in reference.columns:
    plt.figure(figsize=(7, 3))
    plt.hist(reference[column], bins=30, alpha=0.5, label="Reference")
    plt.hist(production[column], bins=30, alpha=0.5, label="Production")
    plt.title(f"Distribution comparison: {column}")
    plt.legend()
    plt.show()
```

## 8. Calculate Population Stability Index (PSI)
PSI compares proportions across bins. It is a monitoring heuristic; thresholds such as 0.1 or 0.25 are conventions sometimes used in practice, not universal statistical laws.

```python
import numpy as np

def calculate_psi(reference_values, production_values, bins=10):
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        raise ValueError("Not enough unique reference values for PSI bins")

    ref_counts, _ = np.histogram(reference_values, bins=edges)
    prod_counts, _ = np.histogram(production_values, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    prod_pct = prod_counts / max(prod_counts.sum(), 1)

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    prod_pct = np.clip(prod_pct, epsilon, None)

    return np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))

for col in reference.columns:
    print(col, round(calculate_psi(reference[col], production[col]), 3))
```

**Caution:** This simple implementation uses reference quantile bins and does not fully handle values outside the reference range. A production implementation should define stable bin edges, handle out-of-range values and missing values explicitly, and validate the method.

## 9. Monitoring workflow
1. Define reference population and feature schema.
2. Collect production features with appropriate privacy controls.
3. Check missingness, ranges and schema.
4. Compare distributions over a fixed time window.
5. Alert on meaningful changes.
6. Investigate source changes and business events.
7. Compare delayed ground-truth performance when labels arrive.
8. Record actions and decisions.

## 10. Approved mini-project — Data Drift Detector

**Scenario:** A subscription service wants to know whether incoming customer features differ from the model's training population.

**Requirements**
1. Load a reference and a production-like dataset.
2. Validate matching feature schema.
3. Compare missing rates, means, standard deviations and quantiles.
4. Plot numerical distributions.
5. Implement PSI or another justified drift statistic.
6. Define configurable warning/alert thresholds.
7. Generate a report with features requiring investigation.
8. Explain why drift does not automatically mean model failure.

**Suggested structure**
```text
data_drift_detector/
├── data/reference.csv
├── data/production_sample.csv
├── src/drift.py
├── reports/
└── README.md
```

**Validation:** Test identical distributions, deliberately shifted data, missing columns and new categories. Document sample size and threshold limitations.

## 11. Common mistakes
- Treating drift as proof of lower accuracy.
- Comparing different populations or time windows.
- Ignoring schema and missingness.
- Choosing alert thresholds without validation.
- Using production labels that are not yet available.
- Logging sensitive raw features unnecessarily.

## 12. Completion checklist
- [ ] Explain data, concept and prediction drift.
- [ ] Compare reference and production data.
- [ ] Calculate and interpret a drift statistic.
- [ ] Explain alert limitations.
- [ ] Complete Data Drift Detector.

**Next lesson:** Day 73 — Model performance monitoring.
