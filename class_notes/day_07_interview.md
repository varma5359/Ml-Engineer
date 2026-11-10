# Day 07: Interview Questions — Linear Algebra: Scalars, Vectors, Matrices and Operations

**Pathway:** Machine Learning Engineer

---

## Question 1: What is a scalar, vector and matrix?

**Answer:** A scalar is one number, a vector is an ordered collection of numbers, and a matrix is a rectangular arrangement of numbers in rows and columns.

**Example:** One model weight is a scalar; one customer's features form a vector; many customers' features form a matrix.

**Follow-up:** What does a matrix shape of `(1000, 12)` often represent in tabular ML?

**Answer:** 1,000 observations and 12 feature columns, assuming rows are observations and columns are features.

## Question 2: What is a dot product and where is it used in ML?

**Answer:** It multiplies corresponding vector elements and sums the products. It is used to calculate weighted sums, such as a linear model's score.

**Follow-up:** What must be true about the vector lengths?

**Answer:** They must be equal for the standard dot product.

## Question 3: What is the difference between element-wise multiplication and matrix multiplication?

**Answer:** Element-wise multiplication multiplies corresponding entries. Matrix multiplication combines rows and columns and requires the left matrix's column count to match the right matrix's row count.

**Follow-up:** In NumPy, which operator performs matrix multiplication?

**Answer:** `@` (or `np.matmul`).

## Question 4: If A has shape `(5, 3)` and B has shape `(3, 2)`, what is the shape of `A @ B`?

**Answer:** `(5, 2)`. The inner dimensions 3 and 3 match; the output uses outer dimensions 5 and 2.

**Follow-up:** Can `(5, 3) @ (4, 2)` be performed?

**Answer:** No, because 3 does not equal 4.

## Question 5: What is transpose?

**Answer:** Transpose swaps rows and columns. A matrix with shape `(2, 3)` becomes shape `(3, 2)` after transpose.

**Follow-up:** Why might transpose appear in ML code?

**Answer:** It changes data orientation to meet the required mathematical shape for a calculation, but the engineer must understand the meaning of each axis before transposing.

## Question 6: How does linear algebra support linear regression?

**Answer:** A linear regression prediction is a weighted sum of input features plus a bias. It can be represented as a dot product between feature values and learned coefficients, plus the intercept.

**Follow-up:** What does a coefficient mean?

**Answer:** In a simple model, it represents the predicted change in the output for a one-unit increase in that feature, holding other features fixed. Interpretation depends on scaling, units and model assumptions.

## Question 7: Why is feature order important when passing a vector into a trained model?

**Answer:** The model's learned weights correspond to specific feature positions. If age and income are swapped, the model applies the age weight to income and vice versa, producing invalid predictions.

**Follow-up:** How can an ML pipeline reduce this risk?

**Answer:** Preserve feature names and order in a consistent preprocessing/model pipeline and validate incoming schema.

## Question 8: A matrix operation raises a dimension mismatch error. How do you investigate it?

**Answer:** Inspect both shapes, identify what each axis means, verify the intended operation and confirm the inner dimensions match for matrix multiplication. Then correct the data orientation only when mathematically justified.

**Follow-up:** Is reshaping enough to make an operation conceptually correct?

**Answer:** No. It may remove the error while scrambling the meaning of observations and features. Validate the mathematical interpretation as well as the shape.