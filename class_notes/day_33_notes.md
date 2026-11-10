# Day 33: Clustering and Unsupervised Learning

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain supervised and unsupervised learning.
- Explain clustering in simple language.
- Understand why clustering does not require a target label.
- Recognize common clustering use cases.
- Distinguish clustering from classification.
- Understand observations, features, clusters and cluster labels.
- Recognize why scaling and feature selection matter.
- Explain why clusters require business interpretation.
- Identify common limitations of clustering.

## 2. What is unsupervised learning?

**Unsupervised learning** is a machine-learning approach where the training data does not include a target label that tells the algorithm the correct answer for each row.

The algorithm looks for structure in the input data.

Examples:
- Group customers with similar purchasing behaviour.
- Identify products that have similar usage patterns.
- Discover groups of similar documents.
- Find unusual sensor behaviour.
- Explore customer segments when predefined segment labels do not exist.

Unsupervised learning helps explore data, but the groups it discovers still need human interpretation and validation.

## 3. What is clustering?

**Clustering** is an unsupervised learning task that groups observations based on similarity.

Imagine a retail company has customer data containing:
- Monthly spending.
- Number of purchases.
- Average order value.
- Time since last purchase.

The company does not have existing customer segment labels. A clustering algorithm can group customers with similar behaviour.

Possible groups might include:
- Frequent, high-spend customers.
- Occasional, low-spend customers.
- New customers.
- Customers who have not purchased recently.

These names are assigned after inspecting the cluster characteristics. The algorithm itself does not know the business meaning of a cluster.

## 4. Clustering versus classification

| Topic | Classification | Clustering |
|---|---|---|
| Learning type | Supervised | Unsupervised |
| Target labels | Known labels provided | No target labels required |
| Goal | Predict a known category | Discover groups in data |
| Example | Predict churn yes/no | Discover customer segments |
| Evaluation | Compare with known labels and task metrics | Use internal structure metrics and domain validation |

A cluster number such as 0 or 1 is not the same as a known business class. Cluster labels are identifiers assigned by the algorithm.

## 5. Common clustering use cases

### Customer segmentation
Group customers based on spending, frequency, tenure or engagement.

### Product grouping
Group products based on sales behaviour, price, category attributes or usage.

### Document grouping
Group similar articles, support tickets or customer feedback.

### Sensor analysis
Group similar operating patterns or identify unusual behaviour.

### Geographic analysis
Group locations based on demand, customer density or service activity.

## 6. Understand the clustering vocabulary

| Term | Meaning |
|---|---|
| Observation | One row being grouped |
| Feature | Input variable used to describe an observation |
| Cluster | A group of observations considered similar |
| Centroid | A representative centre used by centroid-based methods |
| Distance | A measure of how different two observations are |
| Cluster label | An identifier assigned to a group |
| Noise point | An observation not assigned to a dense cluster by some algorithms |
| Cluster profile | Summary describing the observations in a cluster |

## 7. Why feature selection matters

Clustering depends on the features supplied. If irrelevant columns are included, they can distort similarity.

Example customer features:
- Monthly spend.
- Purchase frequency.
- Average order value.
- Customer ID.
- Random internal reference number.

Customer ID usually does not represent customer behaviour. Including it in distance calculations can create meaningless groups.

Before clustering:
1. Understand the business question.
2. Select meaningful features.
3. Check missing values and invalid records.
4. Consider feature scaling.
5. Inspect feature distributions.
6. Document the observation unit.

## 8. Why scaling matters in clustering

Many clustering methods use distances. If one feature has a much larger range than another, it can dominate the distance.

Example:
- Purchase frequency ranges from 1 to 20.
- Annual spending ranges from 1,000 to 500,000.

Without scaling, spending may dominate similarity simply because of its numerical range.

Scaling should be selected based on the clustering method and feature distributions.

## 9. First look at customer data

### 9.1 Import libraries

### 9.1. Import Pandas and plotting libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Ready")
```

**Expected output**

```text
Ready
```

**Explanation**

These libraries support basic exploration before clustering.



### 9.2 Create a customer sample

### 9.2. Create customer behaviour data

```python
df = pd.DataFrame({
    "customer_id": ["C01", "C02", "C03", "C04", "C05", "C06"],
    "monthly_spend": [1200, 1500, 8000, 7500, 3000, 3200],
    "purchase_frequency": [2, 3, 12, 11, 5, 6],
    "average_order_value": [600, 500, 900, 850, 600, 530]
})

print(df)
```

**Expected output**

```text
A six-row customer table
```

**Explanation**

This is a small illustrative dataset. The customer ID is retained for traceability but should not automatically be used as a clustering feature.



### 9.3 Select behaviour features

### 9.3. Exclude the identifier

```python
X = df[[
    "monthly_spend",
    "purchase_frequency",
    "average_order_value"
]]

print(X)
```

**Expected output**

```text
A feature table containing three customer behaviour columns
```

**Explanation**

Only meaningful behavioural features are selected for this initial clustering example.



### 9.4 Inspect feature ranges

### 9.4. Review summary statistics

```python
print(X.describe().T)
```

**Expected output**

```text
Summary statistics for the three behaviour features
```

**Explanation**

The feature scales differ, so scaling should be considered before using a distance-based clustering method.



### 9.5 Visualize two selected features

### 9.5. Create a customer scatter plot

```python
sns.scatterplot(
    data=df,
    x="monthly_spend",
    y="purchase_frequency"
)
plt.xlabel("Monthly spend")
plt.ylabel("Purchase frequency")
plt.title("Customer behaviour")
plt.show()
```

**Expected output**

```text
A scatter plot of spending against purchase frequency
```

**Explanation**

The plot helps inspect whether groups may exist. A visible pattern does not prove that a particular number of clusters is correct.



## 10. Common clustering approaches

### K-Means
Groups observations around cluster centres. It requires the number of clusters to be specified.

### Hierarchical clustering
Builds a hierarchy of groups. A dendrogram can help inspect possible grouping levels.

### DBSCAN
Groups dense regions and can identify observations that do not belong to a dense cluster.

### Gaussian Mixture Models
Models observations as coming from a mixture of probability distributions and can provide soft cluster membership.

Different algorithms make different assumptions. Select a method based on data shape, scale, noise and business needs.

## 11. How do you evaluate clusters?

Clustering does not usually have known correct labels. Evaluation therefore combines:
- Internal metrics such as silhouette score.
- Separation and compactness of groups.
- Stability across samples and settings.
- Cluster size and profile.
- Domain expert review.
- Whether the clusters support a useful business decision.

A high internal score does not guarantee that the clusters are commercially useful.

## 12. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Treat cluster IDs as known classes | Interpret each cluster after profiling |
| Include customer IDs in distance features | Select behavioural or meaningful attributes |
| Ignore feature scaling | Inspect ranges and scale when appropriate |
| Assume every algorithm finds the same groups | Understand each algorithm's assumptions |
| Choose cluster count only by visual preference | Compare internal metrics and business usefulness |
| Call clusters customer types without validation | Profile and validate group characteristics |
| Ignore tiny clusters | Inspect whether they represent rare segments or noise |
| Claim clusters cause behaviour | Describe patterns, not causation |

## 13. Interview-ready explanations

**What is unsupervised learning?**  
It learns structure from input data without supplied target labels.

**What is clustering?**  
It groups observations based on similarity in selected features.

**How is clustering different from classification?**  
Classification predicts predefined labels using labelled examples. Clustering discovers groups without predefined labels.

**Why should features be scaled before K-Means?**  
K-Means uses distances, so features with larger numerical ranges can dominate cluster assignment.

**How do you evaluate a clustering result?**  
Use internal metrics, cluster profiles, stability and domain validation. The best numerical score is not automatically the most useful business segmentation.

## 14. Scenario questions

1. A retail company has no customer segment labels. Which ML approach could help explore segments?  
   **Answer:** Unsupervised clustering using meaningful customer behaviour features.

2. K-Means groups customers mainly by customer ID. What is wrong?  
   **Answer:** The ID is not a behavioural measurement and should not be used as a distance feature.

3. Two clusters have very similar profiles. What do you investigate?  
   **Answer:** Whether the selected cluster count is too high, features are weak or the segmentation has little business value.

4. One cluster contains only a few customers. Is it automatically wrong?  
   **Answer:** No. It may be a rare segment or noise. Inspect its profile and source records.

5. A clustering metric improves but the business cannot explain the groups. Is the result ready?  
   **Answer:** Not necessarily. The clusters need domain interpretation and a useful action or decision.

## 15. Lesson completion checklist

- [ ] I can explain unsupervised learning.
- [ ] I can explain clustering.
- [ ] I can distinguish clustering from classification.
- [ ] I understand cluster and centroid vocabulary.
- [ ] I can select meaningful clustering features.
- [ ] I understand why scaling matters.
- [ ] I can describe common clustering algorithms.
- [ ] I can explain why cluster validation needs domain interpretation.

**Next lesson:** Day 34 — K-Means Clustering and Centroid-Based Grouping.
