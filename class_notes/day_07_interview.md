# Day 07: Interview Questions — Linear Algebra: Scalars, Vectors, Matrices and Operations

**Pathway:** Machine Learning Engineer  
**Focus:** Data representation, matrix operations and input-shape reasoning

## 1. Core concept questions

### Q1. What is a scalar, vector and matrix?

**Answer:** A scalar is one numerical value, a vector is an ordered one-dimensional collection of values, and a matrix is a two-dimensional arrangement with rows and columns.

**Example:** One model setting can be a scalar, one customer's feature values can be a vector, and a table of many customers can be represented as a matrix.

**Follow-up:** What does a matrix shape of `(1000, 12)` commonly represent?

**Answer:** 1,000 observations and 12 feature columns, assuming rows represent observations and columns represent features.

### Q2. What is a dot product and where is it used in ML?

**Answer:** A dot product combines corresponding values from two equal-length vectors by multiplying matching positions and adding the results. It is used to calculate weighted scores, including the weighted feature contribution in a linear model.

**Follow-up:** What must be true about the vector lengths?

**Answer:** They must have the same number of elements for a standard vector dot product.

### Q3. What is the difference between element-wise multiplication and matrix multiplication?

**Answer:** Element-wise multiplication combines values at matching positions. Matrix multiplication combines rows and columns and requires the inner dimensions to match.

**Follow-up:** Which NumPy operator performs matrix multiplication?

**Answer:** The `@` operator. NumPy also provides functions such as `matmul()`.

### Q4. What does transpose do?

**Answer:** Transpose swaps the rows and columns of a matrix. It changes the orientation and shape, but does not change the total number of elements.

**Follow-up:** Should you transpose a feature matrix whenever a model raises a shape error?

**Answer:** No. First understand what each axis represents and what shape the model expects. An unnecessary transpose can change the meaning of the input.

## 2. Practical ML scenarios

### Q5. Your feature matrix has shape (500, 8). How would you explain it?

**Answer:** It commonly represents 500 observations and eight features. I would verify the dataset schema to confirm that each row is one observation and each column is one feature.

**Follow-up:** Does this shape identify the target column?

**Answer:** No. The target must be identified from the problem definition and schema.

### Q6. A model expects four features but an API request supplies three values. What should happen?

**Answer:** The request should fail validation with a clear message explaining the expected feature count or names. The application should not silently invent a missing feature.

**Follow-up:** Why is silently filling the missing value dangerous?

**Answer:** It may produce a plausible-looking but invalid prediction. Missing-feature handling must follow the trained preprocessing contract.

### Q7. A matrix multiplication raises a dimension mismatch error. How do you investigate?

**Answer:** Print the shapes, identify what each axis represents and confirm that the inner dimensions match for the intended operation. Then correct the orientation or feature selection only when it is mathematically and semantically justified.

**Follow-up:** Is reshaping enough to make the calculation correct?

**Answer:** No. It may remove the error while changing the meaning of the observations or features.

### Q8. A prediction changes after the feature order changes. Why?

**Answer:** The model's learned parameters correspond to particular feature positions. If the application swaps two values, the model may apply each parameter to the wrong feature.

**Follow-up:** How can a team reduce this risk?

**Answer:** Preserve feature names and order in a preprocessing pipeline or model input schema, and validate incoming requests against that schema.

### Q9. Your feature matrix has 500 rows but your target contains 498 values. What would you do?

**Answer:** Stop training and trace the preprocessing steps. Check whether rows were removed or reordered in only one object. Restore correct observation alignment using a stable row identifier, then validate both row counts and row correspondence.

**Follow-up:** Is matching row counts sufficient?

**Answer:** No. The feature row and target value must refer to the same observation.

### Q10. Why is linear algebra relevant even when you use scikit-learn rather than writing equations?

**Answer:** Libraries still represent data, model parameters and calculations using numerical arrays and matrices. Understanding shapes, feature order and operations helps diagnose errors and validate model inputs.

**Follow-up:** Does an ML Engineer need to manually implement every matrix operation?

**Answer:** No. Libraries handle many operations. The engineer needs to understand their meaning and validate the inputs and outputs.

## 3. Code interpretation questions

### Q11. What is the output?

```python
import numpy as np
x = np.array([2, 3])
w = np.array([4, 5])
print(x @ w)
```

**Answer:** `23`. The dot product combines corresponding values into one result.

**Follow-up:** What happens if the first vector has three elements and the second has two?

**Answer:** The dot product fails because the vector lengths do not match.

### Q12. What does this expression return?

```python
import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.T.shape)
```

**Answer:** `(3, 2)`. Transpose changes the original two-row, three-column matrix into a three-row, two-column matrix.

**Follow-up:** Does transpose change the number of values?

**Answer:** No.

### Q13. What is the difference between these expressions?

```python
A * B
A @ B
```

**Answer:** For compatible NumPy arrays, `A * B` performs element-wise multiplication. `A @ B` performs matrix multiplication.

**Follow-up:** Why should a developer not replace one with the other just to make code run?

**Answer:** They represent different calculations and can produce different meanings and shapes.

## 4. Interview answer framework

When discussing a shape or matrix error:

1. State what each row and column represents.
2. Inspect the input shapes and data types.
3. Confirm feature names, feature order and target alignment.
4. Identify the intended operation.
5. Correct the smallest underlying issue.
6. Validate using a small known example.
7. Add input checks so the issue is caught earlier.

**Senior developer principle:** A numerical operation can be technically valid and still be semantically wrong. Always validate what the dimensions represent.
