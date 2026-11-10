# Day 05: NumPy Arrays, Indexing and Slicing

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate

## Learning Objectives
- Explain NumPy and its ndarray.
- Create 1D and 2D arrays.
- Understand shape, dimensions, size and dtype.
- Access values using indexing and slicing.
- Perform vectorized operations and basic aggregation.
- Recognize broadcasting and shape errors.

## 1. What Is NumPy?
NumPy (Numerical Python) is a Python library for efficient numerical computation. Its central structure is the `ndarray`, a multi-dimensional array usually storing values of a consistent data type.

Why ML developers use NumPy:
- Store numerical features.
- Perform element-wise operations efficiently.
- Work with vectors and matrices.
- Calculate statistics.
- Exchange data with Pandas and scikit-learn.

## 2. Creating Arrays
**Syntax:** `np.array(sequence)`

**Example 1 — 1D array**
```python
import numpy as np
scores = np.array([78, 85, 92, 88])
print(scores)
print(type(scores))
```
Output: [78 85 92 88] and numpy.ndarray.

**Example 2 — 2D array**
```python
marks = np.array([[80, 90], [70, 85], [95, 88]])
print(marks)
```
Output is a 3-row, 2-column array.

**Example 3 — Array creation functions**
```python
print(np.zeros(3))
print(np.ones(3))
print(np.arange(1, 5))
print(np.linspace(0, 1, 3))
```
Output: [0. 0. 0.], [1. 1. 1.], [1 2 3 4], [0. 0.5 1.].

## 3. Array Properties
| Property | Meaning |
|---|---|
| shape | Number of elements along each dimension |
| ndim | Number of dimensions |
| size | Total number of elements |
| dtype | Data type of array elements |

**Example 1**
```python
a = np.array([10, 20, 30])
print(a.shape, a.ndim, a.size, a.dtype)
```
Shape (3,), dimensions 1, size 3.

**Example 2**
```python
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape)
print(b.ndim)
```
Output: (2, 3) and 2.

**Example 3**
```python
c = np.array([1, 2, 3], dtype=float)
print(c.dtype)
```
Output: float64 (dtype may vary by platform).

## 4. Indexing
**Definition:** Indexing selects an individual element. Python/NumPy indexing starts at zero.

**Syntax:** `array[index]`; for 2D, `array[row_index, column_index]`.

**Example 1**
```python
a = np.array([10, 20, 30, 40])
print(a[0])
print(a[-1])
```
Output: 10 and 40.

**Example 2**
```python
a = np.array([[10, 20], [30, 40]])
print(a[1, 0])
```
Output: 30.

**Example 3 — Update**
```python
a = np.array([10, 20, 30])
a[1] = 99
print(a)
```
Output: [10 99 30].

## 5. Slicing
**Definition:** Slicing extracts a range of elements. The start index is included; the stop index is excluded.

**Syntax:** `array[start:stop:step]`.

**Example 1**
```python
a = np.array([10, 20, 30, 40, 50])
print(a[1:4])
```
Output: [20 30 40].

**Example 2**
```python
a = np.array([10, 20, 30, 40, 50])
print(a[:3])
print(a[2:])
```
Output: [10 20 30] and [30 40 50].

**Example 3 — 2D slicing**
```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a[:2, 1:])
```
Output: [[2 3], [5 6]].

## 6. Vectorized Operations
Vectorization applies an operation to array elements without writing a Python loop for every element.

**Example 1**
```python
prices = np.array([100, 200, 300])
print(prices * 1.10)
```
Output: [110. 220. 330.].

**Example 2**
```python
a = np.array([10, 20, 30])
b = np.array([1, 2, 3])
print(a + b)
print(a * b)
```
Output: [11 22 33] and [10 40 90].

**Example 3 — Filtering**
```python
scores = np.array([45, 70, 85, 30])
print(scores[scores >= 50])
```
Output: [70 85].

## 7. Aggregation
```python
values = np.array([10, 20, 30, 40])
print(values.sum())
print(values.mean())
print(values.min())
print(values.max())
```
Output: 100, 25.0, 10, 40.

## 8. Broadcasting and Shape Awareness
Broadcasting allows compatible arrays of different shapes to participate in element-wise operations. For example, adding a scalar to an array adds it to every element. Incompatible shapes raise a ValueError.

**Scenario questions**
1. A feature matrix has shape (100, 5). How many records and features? 100 records and 5 features.
2. Why does array addition fail for incompatible shapes? NumPy cannot align dimensions under broadcasting rules.
3. Why use boolean indexing? To select values meeting a condition without manually looping.
