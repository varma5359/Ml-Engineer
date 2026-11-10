# Day 43: Interview Questions — Unsupervised Learning Project Integration

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What are the main stages in an end-to-end clustering workflow?
**Answer:** Define the business question and observation unit, select features, validate and preprocess data, fit clustering, evaluate, profile groups, assign new observations, save the workflow and monitor it.

### Q2. Why should customer ID be excluded from clustering features?
**Answer:** It is an identifier rather than a behavioural measurement. Keep it separately to join cluster assignments back to records.

### Q3. Why save the complete preprocessing and clustering Pipeline?
**Answer:** Future records must receive the same imputation and scaling transformations used during training. Saving only KMeans would lose those learned preprocessing steps.

### Q4. How do you assign new customers to existing K-Means clusters?
**Answer:** Use the fitted pipeline's predict method. It applies the fitted preprocessing and assigns each new observation to the nearest learned centroid.

### Q5. Why should the scaler not be refitted on every incoming batch?
**Answer:** The feature representation would change between batches. Reuse the scaler fitted with the historical clustering workflow.

### Q6. What does a large nearest-centroid distance indicate?
**Answer:** The observation is far from all learned centroids in transformed feature space. It may represent unusual behaviour or a new population pattern, but it is not an anomaly probability.

## Scenario questions

### Q7. Incoming customers are assigned to one cluster much more often than historical customers. What do you investigate?
**Answer:** Feature distribution changes, missing values, source-system changes, population changes and whether the historical segmentation still represents current customers.

### Q8. A new input is missing one required feature. What should happen?
**Answer:** Validate the schema and apply only a documented missing-value policy. Do not silently substitute an arbitrary value.

### Q9. A customer has a very large nearest-centroid distance. What do you do?
**Answer:** Review the record, feature values and source quality. Determine whether it is a valid new behaviour or a data issue.

### Q10. A business wants to call cluster 2 "high-value customers". What evidence is needed?
**Answer:** Review original feature distributions, revenue or margin measures, retention and the business definition of value. Cluster ID alone does not establish value.

### Q11. Why store model metadata?
**Answer:** Metadata records the feature schema, algorithm settings, observation unit, preprocessing and model context so the workflow can be reproduced and maintained.

**Senior developer principle:** A clustering workflow is not complete when labels are created. It is complete when inputs, preprocessing, interpretation, assignment and monitoring are consistent and documented.
