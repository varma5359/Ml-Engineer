# Day 36: Hierarchical Clustering and Dendrograms

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised learning  
**Level:** Beginner to Intermediate

## 1. Learning objectives
- Define the concept without memorized jargon.
- Explain why it is used and when it is unsuitable.
- Understand the core syntax and important parameters.
- Interpret the output in a business context.
- Identify assumptions, common failures and validation needs.

## 2. Definition from zero

**Hierarchical Clustering and Dendrograms:** Hierarchical clustering builds nested groups by repeatedly merging (agglomerative) or splitting (divisive) clusters. A dendrogram visualizes the order and distance at which groups are merged.

Start by identifying the data available, the output you need and the business decision that output will support. An algorithm is not useful merely because it runs; its result must answer the actual question.

## 3. Why do we need it?

This technique provides a structured way to learn patterns or organize data that would be difficult to handle with manually written rules. Its usefulness depends on data quality, assumptions, scale and the cost of mistakes.

**If misunderstood or skipped:** the model may learn misleading patterns, create unstable groups, miss important cases or give stakeholders an output they cannot act on.

## 4. Syntax and important parameters

```python
from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(X_scaled, method='ward')
dendrogram(Z)
plt.show()
```

### How to read the code
- Imports load the library components used in the workflow.
- The model/transformer constructor sets the algorithm and its hyperparameters.
- `fit` learns patterns or parameters from the supplied training data.
- `predict` or `fit_predict` produces output labels/values.
- Always inspect output shape, labels and quality before using results.

### Parameter awareness
Understand what each parameter controls before tuning it. Defaults are not guaranteed to suit your dataset. Check the installed library version and official API documentation for version-specific behaviour.

## 5. Worked-example approach
Use a small sample with known structure. Run the algorithm, inspect its output and change one parameter at a time. Record what changed and why. For unsupervised methods, inspect group sizes, profiles and outliers; for supervised methods, compare predictions with known labels using suitable metrics.

## 6. Real-world scenario

**Scenario:** A product team wants to understand similarities among products before choosing a number of categories. A dendrogram lets analysts inspect group structure at several cut levels.

### Senior engineer's approach
1. Clarify the business objective and the exact decision to be supported.
2. Check row grain, features, data types, missing values and units.
3. Check whether scaling or encoding is required.
4. Establish a simple baseline or reference approach.
5. Fit using training data and evaluate on suitable held-out data.
6. Review failure cases and important segments.
7. Communicate limitations and monitor results after deployment.

## 7. What can go wrong?

Hierarchical clustering can be computationally expensive for large datasets and is sensitive to distance and linkage choices. Ward linkage has specific Euclidean assumptions.

Additional checks: verify the algorithm's assumptions, parameter sensitivity, runtime/memory needs and whether the output is stable when the data changes slightly.

## 8. Common beginner mistakes
| Mistake | Why it matters | Better practice |
|---|---|---|
| Selecting a model before defining the problem | Wrong output or metric | Clarify business goal first |
| Ignoring scale and data types | Distances and coefficients can be distorted | Inspect and preprocess appropriately |
| Treating output labels as business truth | Algorithm labels may be arbitrary | Profile and validate with domain knowledge |
| Tuning against test data | Final evaluation becomes biased | Tune on validation/CV; preserve final test |
| Ignoring model limitations | Poor production decisions | Document assumptions and monitor |

## 9. Revision questions
1. Define the method in your own words.
2. What are the inputs and outputs?
3. Which parameters influence its behaviour?
4. What assumptions should be checked?
5. How would you validate usefulness for the business?

## 10. Interview preparation
Use the matching interview file for detailed scenario questions and answer frameworks.
