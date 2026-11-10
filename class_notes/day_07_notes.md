# Day 07: Linear Algebra: Scalars, Vectors, Matrices and Operations

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate  
**Learning focus:** Understand the concepts, apply them correctly, and explain decisions in an interview.

---

## 1. Learning Objectives

- Explain scalars, vectors, matrices and dimensions.
- Represent observations and features using vectors and matrices.
- Perform addition, subtraction, scalar multiplication and dot products.
- Understand matrix multiplication shape rules.
- Interpret transpose and identity matrix.
- Connect linear algebra to linear regression and neural-network calculations.

## 2. Why Linear Algebra Matters in ML

Machine learning data is often stored as numerical arrays. Many algorithms calculate weighted combinations of input features, compare vectors, transform matrices and update parameters.

A simple view:

```text
Rows of data + feature columns
              |
              v
       Numerical matrix X
              |
              v
    Mathematical operations
              |
              v
       Predictions / scores
```

## 3. Scalar

A scalar is a single numerical value.

Examples:

- One customer age: 29
- One model coefficient: 2.5
- One learning rate: 0.01
- One prediction error: -3.2

## 4. Vector

A vector is an ordered collection of numbers. It can represent one observation's features or a list of model weights.

Example customer feature vector:

```text
x = [age, monthly_income, loan_count]
x = [30, 50000, 2]
```

Here the vector has 3 components, so its length/dimension is 3.

### Vector addition

```text
[2, 3] + [4, 1] = [6, 4]
```

Corresponding components are added.

### Scalar multiplication

```text
3 × [2, 4] = [6, 12]
```

Each component is multiplied by the scalar.

### Dot product

The dot product multiplies corresponding components and sums the results.

```text
[2, 3] · [4, 5]
= (2 × 4) + (3 × 5)
= 8 + 15
= 23
```

In ML, a weighted sum of features can be written as a dot product.

## 5. Matrix

A matrix is a rectangular arrangement of numbers in rows and columns.

Example: three customers and two features (age, monthly spend):

```text
X = [ [25, 1200],
      [31, 2500],
      [40, 1800] ]
```

Its shape is `(3, 2)`: 3 observations and 2 features.

### Matrix dimensions

- Rows often represent observations.
- Columns often represent features.
- A target vector is kept separately in supervised learning.

This is a common convention, not a universal rule for every tensor or model input.

## 6. Matrix Operations

### 6.1 Addition

Matrices can be added element by element when their shapes match.

```text
[ [1, 2],     [ [5, 6],     [ [6, 8],
  [3, 4] ]  +   [7, 8] ]  =   [ [10, 12] ]
```

### 6.2 Scalar multiplication

```text
2 × [ [1, 2],     [ [2, 4],
      [3, 4] ]  =   [6, 8] ]
```

### 6.3 Transpose

Transpose swaps rows and columns.

```text
A = [ [1, 2, 3],
      [4, 5, 6] ]

A-transpose = [ [1, 4],
                [2, 5],
                [3, 6] ]
```

If A has shape `(2, 3)`, its transpose has shape `(3, 2)`.

### 6.4 Matrix multiplication

For `A @ B`, the number of columns in A must equal the number of rows in B.

```text
A shape = (2, 3)
B shape = (3, 4)
Result shape = (2, 4)
```

The inner dimensions (3 and 3) must match. The result keeps the outer dimensions (2 and 4).

**Do not confuse matrix multiplication with element-wise multiplication.** In NumPy, `A * B` is element-wise for compatible shapes; `A @ B` performs matrix multiplication.

## 7. Identity Matrix

An identity matrix has 1s on its main diagonal and 0s elsewhere.

```text
I = [ [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1] ]
```

For compatible matrices, multiplying by an identity matrix leaves the matrix unchanged, similar to how multiplying a number by 1 leaves it unchanged.

## 8. Linear Algebra in Linear Regression

A simple linear regression model with one feature predicts:

```text
y_prediction = weight × x + bias
```

For multiple features, the prediction is a weighted sum:

```text
prediction = (weight_1 × feature_1)
           + (weight_2 × feature_2)
           + ... + bias
```

This can be represented using a vector dot product plus a bias.

### Worked example

A simplified rent model uses area and bedrooms:

- area coefficient = 20
- bedroom coefficient = 3,000
- bias = 2,000
- area = 700
- bedrooms = 2

Prediction:

- Area contribution = 20 × 700 = 14,000
- Bedroom contribution = 3,000 × 2 = 6,000
- Add bias = 2,000
- Estimated rent = 22,000

The coefficients are illustrative. A real model learns coefficients from training data and must be evaluated on unseen examples.

## 9. NumPy Demonstration

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Addition:")
print(A + B)

print("Element-wise multiplication:")
print(A * B)

print("Matrix multiplication:")
print(A @ B)

print("Transpose:")
print(A.T)
```

Output:

```text
Addition:
[[ 6  8]
 [10 12]]

Element-wise multiplication:
[[ 5 12]
 [21 32]]

Matrix multiplication:
[[19 22]
 [43 50]]

Transpose:
[[1 3]
 [2 4]]
```

## 10. Industry Scenario — Predicting Delivery Time

A delivery company has two numerical features: distance and number of traffic-delay units. A simplified model calculates an estimated time as a weighted sum.

```python
import numpy as np

features = np.array([8, 3])       # distance, traffic indicator
weights = np.array([4.0, 6.0])
bias = 5.0

estimated_minutes = features @ weights + bias
print(estimated_minutes)
```

Output: `55.0`

Interpretation: the illustrative weighted model estimates 55 minutes. The weights are not trained or validated here; this example only demonstrates a dot product plus bias.

## 11. Common Mistakes

- Multiplying matrices with incompatible inner dimensions.
- Confusing `*` element-wise multiplication with `@` matrix multiplication in NumPy.
- Forgetting that transposing changes shape.
- Mixing feature order between training and inference.
- Interpreting model coefficients without considering feature units and scaling.
- Assuming a mathematical result guarantees a useful business model.

## 12. Quick Revision

- Scalar = one value; vector = ordered values; matrix = rows and columns.
- Dot product creates a weighted sum.
- Matrix multiplication requires matching inner dimensions.
- Transpose swaps axes; identity matrix behaves like multiplicative 1.
- Linear models use weighted combinations of input features.