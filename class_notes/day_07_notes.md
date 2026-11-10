# Day 07: Linear Algebra — Scalars, Vectors, Matrices and Operations

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, Mathematics and Statistics Foundations  
**Level:** Beginner  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

By the end of this lesson, you should be able to:

- Explain a scalar, vector and matrix in simple language.
- Connect rows and columns of a dataset to matrix representation.
- Read the shape of a vector and matrix.
- Distinguish element-wise operations from matrix multiplication.
- Explain a dot product as a weighted combination of values.
- Recognize when matrix dimensions are incompatible.
- Explain why feature order and shape matter when preparing model inputs.

## 2. Why does a Machine Learning Engineer need linear algebra?

Machine learning software works with numerical representations of observations, features, weights and predictions. Linear algebra gives us a consistent way to represent and combine these values.

You do not need to become a mathematician before building ML applications. You do need to understand what the data structures represent and why a numerical operation is valid.

Consider a house-price dataset:

| House | Area (sq. ft.) | Bedrooms | Age (years) |
|---|---:|---:|---:|
| H1 | 1200 | 2 | 8 |
| H2 | 1500 | 3 | 5 |
| H3 | 900 | 2 | 12 |

Each house is one observation. The columns are features. The three feature values for one house can be represented as a vector. All houses together can be represented as a matrix.

## 3. Scalar — one numerical value

A **scalar** is a single numerical value. It has no row or column collection attached to it.

Examples:
- One customer's age: 25.
- A model's learning-rate setting: 0.01.
- A product price: 500.
- A prediction threshold selected by a team: 0.70.

### 3.1 Create a scalar in Python

```python
learning_rate = 0.01

print(learning_rate)
print(type(learning_rate))
```

**Expected output**

```text
0.01
<class 'float'>
```

**Explanation**

- `learning_rate` is the variable name.
- `0.01` is the stored numeric value.
- Python identifies the value as a float.
- In an ML library, the learning rate controls the size of updates during certain optimization procedures. Its suitable value depends on the algorithm and experiment.

### 3.2 Compare two scalar values

```python
minimum_age = 18
customer_age = 25

is_eligible = customer_age >= minimum_age

print(is_eligible)
```

**Expected output**

```text
True
```

**Explanation**

Both values are scalars. Python compares them and returns one Boolean result.

### 3.3 Scalar multiplication of a value

```python
monthly_charge = 500
months = 12

annual_charge = monthly_charge * months

print(annual_charge)
```

**Expected output**

```text
6000
```

**Explanation**

A scalar operation combines individual numerical values. The example annualizes a constant monthly charge; in a real business dataset, verify that the charge does not vary across months before making that assumption.

## 4. Vector — an ordered collection of values

A **vector** is an ordered one-dimensional collection of numerical values.

In a tabular ML problem, one vector can represent the feature values for one observation.

Example:

```text
One house:
[area, bedrooms, age]
[1200, 3, 8]
```

The order matters. If the model expects area first, bedrooms second and age third, the values must be supplied in that same order.

### 4.1 Create a vector using NumPy

```python
import numpy as np

house = np.array([1200, 3, 8])

print(house)
print(type(house))
```

**Expected output**

```text
[1200    3    8]
<class 'numpy.ndarray'>
```

**Explanation**

- `np.array()` creates a NumPy array.
- The array contains three values.
- It is one-dimensional because the values form one sequence.

### 4.2 Inspect the vector shape

```python
print(house.shape)
print(house.ndim)
print(house.size)
```

**Expected output**

```text
(3,)
1
3
```

**Explanation**

- `shape` reports the length of each dimension.
- `ndim` reports the number of dimensions.
- `size` reports the total number of elements.
- `(3,)` means one dimension containing three values. It is not the same as a two-dimensional array with one row and three columns.

### 4.3 Create a customer feature vector

```python
customer = np.array([32, 4500, 7])

print(customer)
```

**Expected output**

```text
[  32 4500    7]
```

**Explanation**

This example represents age, monthly spend and order count. In a real project, document the meaning and units of every position. A vector by itself does not preserve human-readable column labels.

### 4.4 Access values by position

```python
print(customer[0])
print(customer[1])
print(customer[2])
```

**Expected output**

```text
32
4500
7
```

**Explanation**

Python uses zero-based indexing. Position zero is the first value. Position one is the second value.

### 4.5 Change a vector value

```python
customer[1] = 4700

print(customer)
```

**Expected output**

```text
[  32 4700    7]
```

**Explanation**

NumPy arrays are mutable. The value at index one has been replaced. Changing an input value intentionally is different from changing the feature order.

## 5. Matrix — rows and columns of values

A **matrix** is a rectangular arrangement of numerical values with rows and columns.

In a typical tabular ML dataset:
- Each row represents one observation.
- Each column represents one feature.
- The matrix contains the feature values for many observations.

### 5.1 Create a feature matrix

```python
import numpy as np

X = np.array([
    [1200, 2, 8],
    [1500, 3, 5],
    [900, 2, 12]
])

print(X)
```

**Expected output**

```text
[[1200    2    8]
 [1500    3    5]
 [ 900    2   12]]
```

**Explanation**

The matrix has three rows and three columns. The rows represent houses; the columns represent area, bedrooms and age.

### 5.2 Inspect matrix shape

```python
print(X.shape)
print(X.ndim)
print(X.size)
```

**Expected output**

```text
(3, 3)
2
9
```

**Explanation**

The matrix contains three observations and three features. It has two dimensions and nine total values.

### 5.3 Access one observation

```python
first_house = X[0]

print(first_house)
```

**Expected output**

```text
[1200    2    8]
```

**Explanation**

Index zero selects the first row, which represents the first house.

### 5.4 Access one feature column

```python
areas = X[:, 0]

print(areas)
```

**Expected output**

```text
[1200 1500  900]
```

**Explanation**

The colon selects all rows. Index zero selects the first column, which represents area.

### 5.5 Select a single feature value

```python
second_house_bedrooms = X[1, 1]

print(second_house_bedrooms)
```

**Expected output**

```text
3
```

**Explanation**

The first index selects the second row. The second index selects the second column.

## 6. Understand vector and matrix shapes

Shapes help us understand the structure of numerical data and whether operations can be performed.

| Shape | Common tabular interpretation |
|---|---|
| `(5,)` | One-dimensional sequence with five values |
| `(1, 5)` | One row and five columns |
| `(100, 5)` | 100 observations and five features |
| `(1000, 20)` | 1,000 observations and 20 features |
| `(20, 1)` | 20 rows and one column |

### 6.1 Compare a vector and one-row matrix

```python
vector = np.array([10, 20, 30])
one_row_matrix = np.array([[10, 20, 30]])

print(vector.shape)
print(one_row_matrix.shape)
```

**Expected output**

```text
(3,)
(1, 3)
```

**Explanation**

Both contain three values, but their structures differ. A vector is one-dimensional; the second object is two-dimensional with one row and three columns.

### 6.2 Convert a vector into a column-shaped matrix

```python
values = np.array([10, 20, 30])

column_matrix = values.reshape(3, 1)

print(column_matrix)
print(column_matrix.shape)
```

**Expected output**

```text
[[10]
 [20]
 [30]]
(3, 1)
```

**Explanation**

`reshape(3, 1)` changes the shape to three rows and one column. It does not add new values.

### 6.3 Convert a vector into a row-shaped matrix

```python
row_matrix = values.reshape(1, 3)

print(row_matrix)
print(row_matrix.shape)
```

**Expected output**

```text
[[10 20 30]]
(1, 3)
```

**Explanation**

The same three values are arranged as one row. Choose the shape that matches the receiving operation's requirements.

## 7. Basic vector operations

### 7.1 Add two vectors element by element

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)
```

**Expected output**

```text
[5 7 9]
```

**Explanation**

NumPy adds values at matching positions. The vectors have the same shape, so the operation is compatible.

### 7.2 Subtract one vector from another

```python
current_sales = np.array([100, 150, 200])
previous_sales = np.array([90, 160, 180])

difference = current_sales - previous_sales

print(difference)
```

**Expected output**

```text
[ 10 -10  20]
```

**Explanation**

The result compares corresponding positions. A positive result indicates an increase; a negative result indicates a decrease.

### 7.3 Multiply a vector by a scalar

```python
prices = np.array([100, 200, 300])

print(prices * 2)
```

**Expected output**

```text
[200 400 600]
```

**Explanation**

The scalar two is applied to every vector element. This is element-wise scalar multiplication.

### 7.4 Element-wise multiplication of vectors

```python
quantity = np.array([2, 3, 4])
unit_price = np.array([100, 200, 150])

line_totals = quantity * unit_price

print(line_totals)
```

**Expected output**

```text
[200 600 600]
```

**Explanation**

Each quantity is multiplied by the unit price at the same position. This operation is not matrix multiplication.

## 8. Dot product — combining values with weights

A **dot product** combines corresponding values from two equal-length vectors by multiplying matching positions and adding the results.

In a simple linear model, feature values can be combined with model weights to produce a score. The model may also include a separate bias or intercept.

### 8.1 Calculate a dot product

```python
features = np.array([2, 3])
weights = np.array([4, 5])

score = np.dot(features, weights)

print(score)
```

**Expected output**

```text
23
```

**Explanation**

The first feature is paired with the first weight, and the second feature is paired with the second weight. The products are combined into one score.

### 8.2 Use the @ operator for vector dot product

```python
score = features @ weights

print(score)
```

**Expected output**

```text
23
```

**Explanation**

For these one-dimensional vectors, `@` performs the dot product. It is a convenient operator for matrix multiplication and vector dot products.

### 8.3 Understand why vector lengths must match

```python
features = np.array([2, 3, 4])
weights = np.array([0.5, 0.2])

print(features @ weights)
```

**Expected result**

```text
ValueError: shapes (3,) and (2,) not aligned
```

**Explanation**

There are three feature values but only two weights. Each feature needs its corresponding weight in this operation. Do not remove or add values randomly; confirm the intended feature set and model configuration.

### 8.4 Apply a dot product to a customer score

```python
customer_features = np.array([2, 5, 1])
model_weights = np.array([0.4, 0.2, 0.8])

customer_score = customer_features @ model_weights

print(customer_score)
```

**Expected output**

```text
2.6
```

**Explanation**

The three customer features are combined with three weights. This is an educational weighted score, not a trained risk model or a calibrated probability.

## 9. Matrix operations

### 9.1 Add two matrices

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[10, 20], [30, 40]])

print(A + B)
```

**Expected output**

```text
[[11 22]
 [33 44]]
```

**Explanation**

Matching positions are added. Both matrices have the same shape.

### 9.2 Subtract two matrices

```python
print(B - A)
```

**Expected output**

```text
[[ 9 18]
 [27 36]]
```

**Explanation**

The operation subtracts the value in A from the corresponding value in B.

### 9.3 Multiply a matrix by a scalar

```python
print(A * 3)
```

**Expected output**

```text
[[ 3  6]
 [ 9 12]]
```

**Explanation**

Every matrix element is multiplied by three. This is element-wise scalar multiplication.

### 9.4 Element-wise multiplication of matrices

```python
print(A * B)
```

**Expected output**

```text
[[ 10  40]
 [ 90 160]]
```

**Explanation**

The multiplication happens at matching positions. The `*` operator on NumPy arrays does not perform matrix multiplication.

## 10. Matrix multiplication

Matrix multiplication combines rows from the first matrix with columns from the second matrix. The number of columns in the first matrix must equal the number of rows in the second matrix.

### 10.1 Multiply two compatible matrices

```python
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

result = A @ B

print(result)
```

**Expected output**

```text
[[19 22]
 [43 50]]
```

**Explanation**

The operation combines rows of A with columns of B. The output contains two rows and two columns.

### 10.2 Inspect shapes before multiplication

```python
print(A.shape)
print(B.shape)
print(result.shape)
```

**Expected output**

```text
(2, 2)
(2, 2)
(2, 2)
```

**Explanation**

The input dimensions are compatible. The output uses the row count from the first matrix and the column count from the second matrix.

### 10.3 Multiply a feature matrix by a weight column

```python
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

weights = np.array([
    [0.5],
    [0.2]
])

scores = X @ weights

print(scores)
print(scores.shape)
```

**Expected output**

```text
[[0.9]
 [2.3]
 [3.7]]
(3, 1)
```

**Explanation**

There are three observations and two features. The weight matrix contains one weight for each feature. The output contains one score per observation.

### 10.4 Observe an incompatible matrix operation

```python
A = np.ones((3, 4))
B = np.ones((2, 3))

print(A @ B)
```

**Expected result**

```text
ValueError: matmul input operand mismatch in its core dimension
```

**Explanation**

The first matrix has four columns, while the second matrix has two rows. These inner dimensions do not match. Inspect shapes and confirm the intended orientation before changing the data.

## 11. Transpose

The **transpose** of a matrix swaps its rows and columns.

### 11.1 Transpose a matrix

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(A.T)
```

**Expected output**

```text
[[1 4]
 [2 5]
 [3 6]]
```

**Explanation**

The original matrix has two rows and three columns. The transpose has three rows and two columns.

### 11.2 Inspect before and after transpose

```python
print(A.shape)
print(A.T.shape)
```

**Expected output**

```text
(2, 3)
(3, 2)
```

**Explanation**

Transpose changes orientation, not the number of elements. It does not automatically know whether rows represent observations or features.

## 12. Identity matrix

An **identity matrix** is a square matrix with ones along its main diagonal and zeros elsewhere. It is a useful reference object in linear algebra.

### 12.1 Create an identity matrix

```python
identity = np.eye(3)

print(identity)
```

**Expected output**

```text
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

**Explanation**

`np.eye(3)` creates a three-by-three identity matrix. The diagonal values are one and all other values are zero.

## 13. Linear algebra and ML model inputs

A typical tabular ML application uses a feature matrix and often a separate target vector.

### 13.1 Create a small feature matrix

```python
X = np.array([
    [25, 4500],
    [30, 5200],
    [35, 6100]
])

print(X.shape)
```

**Expected output**

```text
(3, 2)
```

**Explanation**

There are three observations and two features. In a real dataset, the feature names and units should be stored alongside the array or in a schema.

### 13.2 Create the target vector

```python
y = np.array([0, 1, 0])

print(y.shape)
```

**Expected output**

```text
(3,)
```

**Explanation**

There is one target value for each observation. The labels are illustrative binary outcomes.

### 13.3 Validate row counts

```python
print(X.shape[0])
print(y.shape[0])
print(X.shape[0] == y.shape[0])
```

**Expected output**

```text
3
3
True
```

**Explanation**

The feature matrix and target vector have the same number of observations. This is a necessary check, but row order and observation alignment must also be preserved.

### 13.4 Explain why feature order matters

Imagine a trained model was fitted using this order:

1. Age
2. Monthly spend
3. Number of orders

If a new application sends number of orders first and age second, the model will interpret values incorrectly. A valid numeric array can still be semantically wrong.

**Developer habit:** Store and validate the expected feature names and order as part of the model's input contract.

## 14. Common beginner mistakes

| Mistake | Why it is a problem | Better approach |
|---|---|---|
| Confusing `(3,)` with `(1, 3)` | They are different dimensional structures | Inspect shape and use the required representation |
| Confusing `*` with `@` | Element-wise and matrix multiplication are different operations | Select the operation based on its meaning |
| Ignoring feature order | Model weights may be applied to the wrong values | Preserve a documented feature schema |
| Reshaping until an error disappears | The result may have the wrong observation meaning | Understand each axis before reshaping |
| Ignoring units | Area, age and price may have different scales and meanings | Document units and preprocessing |
| Assuming a score is a probability | A weighted score may not be calibrated | Use an appropriate model and probability validation |
| Forgetting row alignment | Features may be paired with the wrong target | Preserve row identity through processing |

## 15. Interview-ready explanations

### What is the difference between a scalar, vector and matrix?

A scalar is one numerical value. A vector is an ordered one-dimensional collection. A matrix is a two-dimensional arrangement with rows and columns. In tabular ML, one observation can be represented as a feature vector and many observations as a feature matrix.

### What is the difference between element-wise multiplication and matrix multiplication?

Element-wise multiplication combines values at matching positions. Matrix multiplication combines rows and columns and requires compatible inner dimensions. In NumPy, `*` is element-wise multiplication and `@` performs matrix multiplication.

### Why is shape validation important in machine learning?

The model expects inputs in a particular structure. Incorrect row counts, feature counts or orientation can cause errors or, more dangerously, incorrect predictions. Shape checks should be combined with feature-name and row-alignment checks.

### Why does feature order matter?

The model's learned parameters correspond to particular feature positions. Changing the order without changing the model's expected schema can cause each value to be interpreted as a different feature.

## 16. Scenario questions

1. A dataset has shape `(800, 12)`. What does this commonly represent?  
   **Answer:** 800 observations and 12 feature columns, assuming rows are observations.

2. A model expects four features but the API receives three values. What should happen?  
   **Answer:** Reject the request with a clear input-validation error. Do not silently add an arbitrary value.

3. A matrix multiplication fails. What do you inspect first?  
   **Answer:** The shapes and the meaning of the axes for both matrices.

4. A prediction changes after the feature order changes. Why?  
   **Answer:** The model may be applying learned parameters to the wrong feature positions.

5. The feature matrix has 500 rows and the target has 498 values. Can training proceed?  
   **Answer:** No. Investigate filtering, dropped rows and alignment before training.

## 17. Lesson completion checklist

- [ ] I can explain scalar, vector and matrix.
- [ ] I can represent one observation as a vector.
- [ ] I can represent a tabular dataset as a matrix.
- [ ] I can inspect dimensions and shapes.
- [ ] I can distinguish element-wise and matrix multiplication.
- [ ] I can explain a dot product as a weighted combination.
- [ ] I can transpose a matrix and inspect the new shape.
- [ ] I can identify incompatible matrix dimensions.
- [ ] I understand why feature order and row alignment matter.

**Next lesson:** Day 08 — Mean, Median, Variance, Standard Deviation, Probability and Distributions.
