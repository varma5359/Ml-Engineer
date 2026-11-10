# Day 05: Interview Questions — NumPy Arrays, Indexing and Slicing

**Pathway:** Machine Learning Engineer  
**Focus:** Numerical arrays, dimensions, vectorization and shape debugging

## 1. Core concepts

### Q1. What is NumPy and why is it important in machine learning?

**Answer:** NumPy is a Python library for numerical computing. Its central structure, the ndarray, supports multi-dimensional numerical data, indexing, aggregation and vectorized operations. It is commonly used by scientific Python and machine-learning libraries.

**Follow-up:** What is an ndarray?

**Answer:** It is NumPy's n-dimensional array object, typically storing elements with a common data type.

### Q2. What is the difference between a Python list and a NumPy array?

**Answer:** A Python list is a general-purpose collection that can contain mixed object types. A NumPy array is designed for numerical computation, commonly stores a consistent dtype, supports multi-dimensional shapes and performs element-wise operations.

**Follow-up:** Does a NumPy array always have to contain numbers?

**Answer:** No. NumPy supports other data types too, but numeric ML operations generally require properly prepared numerical arrays.

### Q3. Explain shape, ndim, size and dtype.

**Answer:** `shape` gives the length along each dimension, `ndim` gives the number of dimensions, `size` gives the total number of elements and `dtype` gives the element data type.

**Example:** For an array with shape `(100, 5)`, there are 100 rows, five columns, two dimensions and 500 elements.

**Follow-up:** Does shape tell you which column is the target?

**Answer:** No. Shape describes dimensions only. The dataset schema and problem definition identify the target.

### Q4. What is the difference between indexing and slicing?

**Answer:** Indexing selects a particular element by position. Slicing selects a range or subset. Python and NumPy use zero-based indexing, and the slice stop position is excluded.

**Follow-up:** What does `x[-1]` select?

**Answer:** The final element.

### Q5. What is vectorization?

**Answer:** Vectorization applies an operation to an entire array rather than writing an explicit Python loop for each element. NumPy uses optimized numerical routines for many such operations.

**Follow-up:** Is vectorization always possible or always faster?

**Answer:** No. It depends on the operation, data size and implementation. Prefer clear correct code and measure performance when it matters.

### Q6. What does axis=0 mean in a 2D aggregation?

**Answer:** An aggregation along axis zero combines values down the rows and produces one result per column. Axis one combines values across columns and produces one result per row.

**Follow-up:** How would you confirm the resulting shape?

**Answer:** Print the result's shape and compare it with the expected remaining dimensions.

## 2. Practical and debugging questions

### Q7. A matrix has shape (500, 8). How do you select the fourth column?

**Answer:** Use `matrix[:, 3]`, assuming the columns are zero-indexed and the fourth column is the intended feature.

**Follow-up:** How do you select rows at positions 10 through 19?

**Answer:** Use `matrix[10:20, :]`. The stop position 20 is excluded.

### Q8. You receive a broadcasting error when adding two arrays. How do you debug it?

**Answer:** Print both arrays' shapes and inspect what each axis represents. NumPy can broadcast compatible dimensions, but incompatible shapes raise an error. Reshape or transpose only when it matches the intended data layout.

**Follow-up:** Would you reshape randomly until the error disappears?

**Answer:** No. A shape-compatible result can still be mathematically wrong. Confirm the meaning of each axis first.

### Q9. A NumPy array contains values that look numeric but its dtype is object. What would you do?

**Answer:** Inspect the actual values and identify text markers, mixed types or malformed entries. Decide how each invalid value should be handled, convert valid values to a suitable numeric type and check the result's dtype and missing-value behaviour.

**Follow-up:** Why not cast the entire array without inspection?

**Answer:** Conversion may fail or lose information. The correct cleaning decision depends on the meaning of the invalid values.

### Q10. What is the difference between a view and a copy?

**Answer:** A view can share underlying data with the original array, so changes through the view may affect the original. A copy has independent array data.

**Follow-up:** How do you request an independent copy of a slice?

**Answer:** Use `.copy()`.

### Q11. What is broadcasting?

**Answer:** Broadcasting allows operations between arrays with compatible shapes, such as adding a scalar to each element or adding a row-shaped array to each row of a matrix.

**Follow-up:** What happens when dimensions are incompatible?

**Answer:** NumPy raises a broadcasting-related ValueError.

## 3. ML workflow scenarios

### Q12. Your feature matrix has shape (1, 20), but your model expects a two-dimensional batch. Is it valid?

**Answer:** It is already two-dimensional: one observation with 20 features. Whether the estimator accepts it depends on the API and whether the target operation expects a single row or a batch. Inspect the model's input contract rather than blindly reshaping.

**Follow-up:** What does shape (20,) mean?

**Answer:** A one-dimensional array containing 20 elements. It is not the same shape as a two-dimensional one-row matrix.

### Q13. Your feature matrix has 500 rows, but your target has 498 values. What would you do?

**Answer:** Stop before fitting the model. Trace the preprocessing and filtering steps applied to features and target. Check whether rows were dropped or reordered in only one object. Restore alignment using a reliable row identifier or consistent indexing, then verify the row counts and sample correspondence.

**Follow-up:** Is matching row counts enough?

**Answer:** No. The same observations must appear in the same order or be aligned correctly.

### Q14. A student uses Python's and to combine two NumPy masks and gets an ambiguous truth-value error. What is the correction?

**Answer:** Use element-wise `&` and put each comparison inside parentheses. Use `|` for element-wise OR.

**Follow-up:** When is Python's and appropriate?

**Answer:** For scalar Boolean conditions, not for combining arrays of Boolean values.

### Q15. A numerical operation produces unexpected values after reshaping. What would you inspect?

**Answer:** Check the original and new shapes, the number of elements, the order of values and the intended meaning of rows and columns. Confirm that the reshape preserves the desired observation-feature layout.

**Follow-up:** Does reshape automatically infer the business meaning of each column?

**Answer:** No. It changes the arrangement of elements, not their meaning.

### Q16. When would you use Pandas instead of NumPy?

**Answer:** Use Pandas when working with labelled tabular data, column names, mixed data types, filtering and missing-value inspection. Use NumPy for numerical arrays and vectorized numerical operations. They are complementary and commonly used together.

**Follow-up:** Can a DataFrame be converted to NumPy?

**Answer:** Yes, using methods such as `to_numpy()`, while checking how mixed column types are represented.

## 4. Code interpretation questions

### Q17. What is the output?

```python
import numpy as np
values = np.array([10, 20, 30, 40])
print(values[1:3])
```

**Answer:** `[20 30]`. The start is included and the stop is excluded.

**Follow-up:** What does `values[::-1]` return?

**Answer:** The elements in reverse order.

### Q18. What is the output?

```python
import numpy as np
values = np.array([1, 2, 3])
print(values + 5)
```

**Answer:** `[6 7 8]`. The scalar is added to every array element.

**Follow-up:** Would the same expression with a Python list behave identically?

**Answer:** No. List plus list concatenates; adding an integer to a list is not the same vectorized operation.

### Q19. What does this return?

```python
import numpy as np
matrix = np.array([[1, 2], [3, 4], [5, 6]])
print(matrix.sum(axis=0))
```

**Answer:** `[9 12]`. It sums down the rows, producing one value per column.

**Follow-up:** What would axis=1 return?

**Answer:** `[3 7 11]`, one sum per row.

### Q20. What does this filter select?

```python
import numpy as np
ages = np.array([18, 25, 32, 16, 40])
print(ages[(ages >= 20) & (ages <= 35)])
```

**Answer:** `[25 32]`. It selects values meeting both conditions.

**Follow-up:** Why are parentheses used around each comparison?

**Answer:** They make the intended comparison grouping explicit and avoid operator-precedence problems.

## 5. Interview answer framework

When debugging NumPy code:

1. Inspect the input values and dtypes.
2. Print the shapes and number of dimensions.
3. Confirm what each axis represents.
4. Reproduce the issue with a small array.
5. Check indexing, slicing and broadcasting assumptions.
6. Compare the output with a manually verified small example.
7. Add a test for the corrected behaviour.

**Senior developer principle:** Do not treat a shape error as merely a syntax problem. It often indicates that the program's assumptions about observations, features or dimensions are inconsistent.
