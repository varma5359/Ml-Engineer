# Day 27: Interview Questions — K-Nearest Neighbors and Distance Metrics

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is K-Nearest Neighbors?
**Answer:** KNN predicts a new observation using the labels or target values of nearby training observations. For classification it uses neighbour voting; for regression it combines neighbour target values.
**Follow-up:** Why is it called a lazy learner?  
**Answer:** It stores training observations and performs much of its computation when predictions are requested.

### Q2. What does K control?
**Answer:** It controls how many neighbours contribute to a prediction. Small K can be sensitive to noise; large K can smooth over local patterns.
**Follow-up:** Is there one best K for every dataset?  
**Answer:** No. Select it using validation or cross-validation.

### Q3. Why is feature scaling important for KNN?
**Answer:** KNN uses distances. Features with larger numerical ranges can dominate distance calculations if they are not scaled appropriately.
**Follow-up:** Should scaling be fitted on the full dataset?  
**Answer:** No. Fit on training data only and transform held-out data.

### Q4. Compare Euclidean and Manhattan distance.
**Answer:** Euclidean distance measures straight-line distance. Manhattan distance sums absolute differences across dimensions.
**Follow-up:** Which one is always better?  
**Answer:** Neither. Compare based on data representation and validation.

### Q5. What is distance weighting?
**Answer:** It gives closer neighbours more influence than farther neighbours. Uniform weighting gives each selected neighbour equal influence.

## Scenario questions

### Q6. KNN performs poorly because income ranges into millions while age ranges below 100. What do you do?
**Answer:** Inspect feature units and scale numeric features inside a training pipeline.
**Follow-up:** Does scaling fix irrelevant features?  
**Answer:** No. Irrelevant or noisy features can still distort distance.

### Q7. K=1 performs well on training but poorly on validation. What may be happening?
**Answer:** The model may be sensitive to individual training observations and overfit noise.
**Follow-up:** What might you try?  
**Answer:** Compare larger K values and suitable distance/weighting options on validation data.

### Q8. The dataset has hundreds of features. What is the concern?
**Answer:** In high-dimensional spaces, distances may become less informative. Consider feature selection, dimensionality reduction or another model family.
**Follow-up:** Would scaling alone solve it?  
**Answer:** No. Scaling does not remove irrelevant dimensions.

### Q9. KNN prediction is slow after deployment. What may explain it?
**Answer:** KNN may need to compare a new observation with many stored training rows. Large datasets increase prediction cost and memory needs.
**Follow-up:** What alternatives could you investigate?  
**Answer:** Efficient neighbour-search structures, approximate search or a different estimator with faster inference.

### Q10. A new observation is far outside the training feature range. What is the concern?
**Answer:** Its nearest training neighbours may still be poor analogues. Validate inputs and recognize that predictions outside the training distribution can be unreliable.

## Code interpretation

### Q11. What does n_neighbors=5 mean?
**Answer:** Five nearest training observations contribute to the prediction.
**Follow-up:** Does KNN learn five model coefficients?  
**Answer:** No. It retains the training observations and uses them during prediction.

### Q12. What is the difference between weights="uniform" and weights="distance"?
**Answer:** Uniform gives each neighbour equal influence. Distance weighting gives closer neighbours more influence.

**Senior developer principle:** KNN quality depends heavily on feature representation, scaling and the meaning of distance.
