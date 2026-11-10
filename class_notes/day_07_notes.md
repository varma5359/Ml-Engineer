# Day 07: Linear Algebra: Scalars, Vectors, Matrices and Operations

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define scalar, vector and matrix.
- Represent observations and features numerically.
- Understand dimensions and shapes.
- Perform addition, subtraction, scalar multiplication and dot products.
- Understand matrix multiplication and transpose.
- Connect linear algebra with ML models.

## 1. What Is Linear Algebra?
Linear algebra is the branch of mathematics that studies vectors, matrices and operations between them. Machine learning uses it to represent data and calculate predictions efficiently.

## 2. Scalar
**Definition:** A scalar is a single numerical value.

**Examples**
1. A customer's age: 25.
2. A model learning rate: 0.01.
3. A product price: 500.

```python
learning_rate = 0.01
print(learning_rate)
```

## 3. Vector
**Definition:** A vector is an ordered one-dimensional collection of numbers. In ML, one vector can represent the features of one observation.

**Example:** A house may be represented as [area, bedrooms, age] = [1200, 3, 8].

```python
import numpy as np
house = np.array([1200, 3, 8])
print(house.shape)
```
Output: (3,).

**Three examples**
1. Customer features: [age, monthly_spend, order_count].
2. Image pixel intensities: [0.2, 0.7, 0.1].
3. Model weights: [0.5, -0.2, 1.1].

## 4. Matrix
**Definition:** A matrix is a rectangular arrangement of numbers with rows and columns. A dataset can be represented as a matrix where rows are observations and columns are features.

```python
X = np.array([[25, 3000], [30, 4500], [22, 2500]])
print(X.shape)
```
Output: (3, 2), meaning 3 rows and 2 columns.

**Three examples**
1. Customer dataset: each row is a customer, each column is a feature.
2. Grayscale image: rows and columns represent pixel positions.
3. Neural-network weights: a matrix connects one layer's inputs to another layer's units.

## 5. Basic Operations
**Addition/subtraction:** Arrays must have compatible shapes.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)
print(b - a)
```
Output: [5 7 9] and [3 3 3].

**Scalar multiplication**
```python
a = np.array([2, 4, 6])
print(3 * a)
```
Output: [6 12 18].

**Dot product:** Multiply corresponding vector elements and add the products.
```python
x = np.array([2, 3])
w = np.array([4, 5])
print(np.dot(x, w))
print(x @ w)
```
Output: 23.

**Three practical dot-product examples**
1. Combine house features with model weights to produce a price score.
2. Combine customer features with learned weights to produce a risk score.
3. Combine image features with weights in a neural-network layer.

## 6. Matrix Multiplication and Shape
For A with shape (m, n) and B with shape (n, p), A @ B is valid and produces shape (m, p). The inner dimensions must match.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5], [6]])
print(A @ B)
```
Output: [[17], [39]].

## 7. Transpose and Identity Matrix
Transpose swaps rows and columns: `A.T`. An identity matrix has ones on its main diagonal and zeros elsewhere: `np.eye(n)`.

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.T)
print(np.eye(2))
```

## 8. Scenario Questions
1. A feature vector has 4 values. What does that represent? Four numerical features for one observation.
2. A matrix has shape (500, 8). How many observations and features? 500 observations and 8 features.
3. Why does A @ B fail for some matrices? The inner dimensions do not match.
