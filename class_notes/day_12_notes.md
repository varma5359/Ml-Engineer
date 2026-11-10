# Day 12: Encoding Categorical Variables and Scaling Numerical Data

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## Learning Objectives
- Explain categorical and numerical features.
- Convert categories into numeric representations.
- Compare one-hot and ordinal encoding.
- Explain standardization and min-max scaling.
- Avoid preprocessing leakage.

## 1. What Is Encoding?
Encoding converts categorical values into a numeric representation so an estimator can process them.

### One-hot encoding
Creates a separate indicator column for each category.

```python
import pandas as pd
df = pd.DataFrame({"city": ["Hyd", "Vizag", "Hyd"]})
print(pd.get_dummies(df, columns=["city"], dtype=int))
```
Output contains city_Hyd and city_Vizag indicator columns.

**Three examples**
1. City: Hyderabad, Vizag, Chennai → separate city indicator columns.
2. Payment method: cash, card, UPI → indicator columns.
3. Product type: phone, laptop, tablet → indicator columns.

### Ordinal encoding
Maps categories to numbers when a meaningful order exists, such as low=0, medium=1, high=2. Do not use arbitrary numeric codes for unordered cities.

## 2. What Is Scaling?
Scaling changes numeric feature magnitudes. It matters for algorithms that use distances or gradient-based optimization.

### Standardization
Transforms values using (value - training mean) / training standard deviation.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Min-max scaling
Maps values to a selected range, often 0 to 1.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Three scaling scenarios**
1. KNN comparing income and age: scale numeric features so income does not dominate only because of units.
2. SVM using measurements in different units: scaling is usually important.
3. Tree-based model: scaling is often unnecessary for tree split decisions.

## 3. Important Rule: Avoid Leakage
Split the data first. Fit the encoder/scaler on training data, then use transform() on validation/test data. Do not learn preprocessing statistics from the full dataset.

## 4. Scenario Questions
1. City is encoded as 1, 2 and 3. What is the risk? The model may infer a false order.
2. KNN distance is dominated by salary. What can help? Scale numeric features.
3. A new city appears after deployment. What should you plan for? Configure unknown-category handling and monitor unseen values.
