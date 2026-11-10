# Day 12: Interview Questions — Encoding Categorical Variables and Scaling Numerical Data

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. Why do we encode categorical variables?
**Answer:** Many estimators expect numerical inputs. Encoding converts category labels into a numerical representation.
**Follow-up:** Should every category receive an arbitrary integer?  
**Answer:** No. That may create an artificial order or distance for nominal categories.

### Q2. What is one-hot encoding?
**Answer:** It creates indicator columns for categories in a nominal feature.
**Follow-up:** What is one drawback?  
**Answer:** High-cardinality features can create many columns and increase dimensionality.

### Q3. When is ordinal encoding suitable?
**Answer:** When the categories have a meaningful order, such as Low, Medium and High, and the mapping is defined.
**Follow-up:** Should cities be ordinal-encoded as 1, 2 and 3?  
**Answer:** Not without a meaningful order; city names are nominal categories.

### Q4. Why scale numerical features?
**Answer:** Distance-based and many gradient-based methods can be influenced by feature magnitude. Scaling puts numerical features on a more comparable scale.
**Follow-up:** Do all algorithms require scaling?  
**Answer:** No. Tree-based models are generally less sensitive to monotonic scaling.

### Q5. Compare StandardScaler and MinMaxScaler.
**Answer:** StandardScaler centers features and scales their spread using training statistics. MinMaxScaler maps the training minimum and maximum to a chosen range, often zero to one.
**Follow-up:** Is MinMaxScaler robust to extreme values?  
**Answer:** No. Extreme minima or maxima can compress the other observations.

## Scenario questions

### Q6. A developer maps Hyderabad=1, Chennai=2 and Vizag=3. What is the concern?
**Answer:** The codes may imply a ranking that does not exist. Use an appropriate nominal encoding.
**Follow-up:** Does a tree model automatically make arbitrary codes harmless?  
**Answer:** No. The codes can still impose artificial split structure.

### Q7. A KNN model is dominated by income rather than age. What do you investigate?
**Answer:** Inspect units and feature ranges. Consider scaling numerical features using training-only fitted statistics and compare validation performance.
**Follow-up:** Does scaling fix invalid data?  
**Answer:** No. It does not fix incorrect units, leakage or poor feature definitions.

### Q8. A new city appears in production. What should happen?
**Answer:** Use a documented unknown-category policy, such as OneHotEncoder with handle_unknown="ignore", and monitor the new category.
**Follow-up:** Does ignore mean the model has learned the new city?  
**Answer:** No. It prevents an encoding error but does not create learned information about that category.

### Q9. A scaler was fitted before train-test splitting. What is wrong?
**Answer:** Held-out information influenced the scaling statistics. Split first, fit on training data and transform test data using the fitted scaler.
**Follow-up:** How do you prevent this?  
**Answer:** Use a Pipeline or ColumnTransformer inside the model evaluation workflow.

### Q10. A tree-based model performs similarly with and without scaling. Is that surprising?
**Answer:** No. Tree splits are generally less sensitive to monotonic feature scaling than distance-based methods.
**Follow-up:** Should scaling be added automatically?  
**Answer:** No. Match preprocessing to the estimator and validate.

## Code interpretation

### Q11. What does OneHotEncoder(handle_unknown="ignore") do?
**Answer:** It learns categories during fit and avoids raising an error for unseen categories during transform.
**Follow-up:** Does transform learn new categories?  
**Answer:** No. New categories require an updated fit if they should become learned columns.

### Q12. Difference between fit_transform() and transform()?
**Answer:** fit_transform learns transformation parameters and applies them. transform applies parameters already learned.
**Follow-up:** Which should be used on test data?  
**Answer:** transform() using the object fitted on training data.

### Q13. Why use ColumnTransformer with Pipeline?
**Answer:** ColumnTransformer applies different transformations to selected column groups. Pipeline connects those transformations to the estimator and helps keep training and inference consistent.

**Senior developer principle:** Preprocessing is part of the model. Training and production must use the same fitted transformation.
