# Day 31: Interview Questions — Support Vector Machines (SVM)

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is SVM?
**Answer:** A supervised algorithm that finds a decision boundary with a useful margin between classes. Kernel methods allow it to represent nonlinear boundaries.
**Follow-up:** Is SVM used only for classification?  
**Answer:** No. Support Vector Regression is also available.

### Q2. What are support vectors?
**Answer:** Training observations that are important in defining the decision boundary, typically observations closest to the boundary.
**Follow-up:** Are support vectors automatically the most important business records?  
**Answer:** No. They are important to the learned boundary, not necessarily to the business process.

### Q3. What is the margin?
**Answer:** It is the separation around the decision boundary. SVM tries to find a useful margin while managing classification errors.
**Follow-up:** Does a larger margin always mean better test performance?  
**Answer:** No. It depends on the data and regularization settings.

### Q4. What is the difference between linear and RBF kernels?
**Answer:** A linear kernel creates a linear boundary in the input representation. RBF can model nonlinear boundaries by using similarity in feature space.
**Follow-up:** Is RBF always better?  
**Answer:** No. Compare using validation and consider complexity.

### Q5. What does C control?
**Answer:** It controls how strongly training classification errors are penalized. Larger C fits training observations more strictly; smaller C allows a softer margin.
**Follow-up:** What is a risk of very high C?  
**Answer:** It may overfit.

### Q6. What does gamma control for an RBF kernel?
**Answer:** It controls how local each training observation's influence is. High gamma can create complex local boundaries; low gamma creates smoother boundaries.
**Follow-up:** What is a risk of very high gamma?  
**Answer:** Overfitting.

## Scenario questions

### Q7. SVM performs poorly because income has a much larger range than age. What do you do?
**Answer:** Scale features using a pipeline fitted on training data.
**Follow-up:** Does scaling fix irrelevant features?  
**Answer:** No. Irrelevant or noisy features can still distort the model.

### Q8. Training accuracy is high but validation accuracy is low with high gamma. What may be happening?
**Answer:** The model may be overfitting with overly local decision regions.
**Follow-up:** What would you try?  
**Answer:** Test lower gamma or different C values using validation or cross-validation.

### Q9. A linear SVM underfits but RBF performs better on validation. What might that indicate?
**Answer:** The class relationship may be nonlinear and RBF captures it better.
**Follow-up:** Should you choose RBF immediately?  
**Answer:** Also consider stability, inference cost and the business need for interpretability.

### Q10. SVM training is slow on a very large dataset. What do you consider?
**Answer:** Benchmark computational cost, simplify the model, use a suitable implementation or compare with algorithms that scale better for the task.

## Practical questions

### Q11. Why scale features for SVM?
**Answer:** The margin and kernel calculations depend on feature geometry. Large feature ranges can dominate if inputs are not scaled.

### Q12. Does SVC return probabilities by default?
**Answer:** No. Probability estimates must be enabled explicitly, and their calibration should be assessed if they drive decisions.

**Senior developer principle:** SVM performance depends on feature representation, scaling and parameter choices. Validate the complete workflow.
