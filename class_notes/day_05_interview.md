# Day 05: Interview Questions — NumPy Arrays, Indexing and Slicing

**Pathway:** Machine Learning Engineer

---

## Question 1: What is NumPy and why is it important in machine learning?

**Answer:** NumPy is a Python library for efficient numerical arrays and mathematical operations. ML data is often represented as numeric vectors and matrices, and NumPy supports those operations efficiently.

**Follow-up:** What is an `ndarray`?

**Answer:** It is NumPy's n-dimensional array object, typically storing elements of a common dtype.

## Question 2: What is the difference between a Python list and a NumPy array?

**Answer:** A Python list is a general-purpose collection and can hold mixed object types. A NumPy array is designed for numerical operations, usually has a consistent dtype and supports vectorized computation and multi-dimensional shapes.

**Follow-up:** Can a NumPy array contain strings?

**Answer:** Yes, NumPy supports string and object arrays, but numeric ML calculations generally require properly prepared numerical arrays.

## Question 3: Explain `shape`, `ndim`, `size` and `dtype`.

**Answer:** `shape` gives the length along each dimension, `ndim` gives the number of dimensions, `size` gives the total number of elements, and `dtype` gives the element data type.

**Example:** For an array of shape `(100, 5)`, there are 100 observations and 5 columns, 2 dimensions and 500 elements.

**Follow-up:** Does shape `(100, 5)` tell you which column is the target?

**Answer:** No. Shape tells the dimensions only. You must know the dataset schema and target definition.

## Question 4: What is the difference between indexing and slicing?

**Answer:** Indexing selects a particular element. Slicing selects a range or subset of elements.

**Example:** `x[2]` selects one value; `x[1:4]` selects positions 1, 2 and 3.

**Follow-up:** Is the stop position included in a Python slice?

**Answer:** No. The start is included and the stop is excluded.

## Question 5: A matrix has shape `(500, 8)`. How do you select the fourth feature column?

**Answer:** `matrix[:, 3]`, assuming feature columns are zero-indexed and the fourth column is indeed the desired feature.

**Follow-up:** How do you select rows 10 through 19?

**Answer:** `matrix[10:20, :]`.

## Question 6: What is vectorization?

**Answer:** Vectorization means applying an operation to an entire array or vector rather than writing an explicit Python loop for each element. NumPy performs these operations in optimized compiled code in many cases.

**Follow-up:** Is vectorization always possible?

**Answer:** Not every algorithm or dependency is naturally vectorized. Use the clearest correct approach and measure performance where it matters.

## Question 7: What does `axis=0` mean in `array.sum(axis=0)`?

**Answer:** It reduces along axis 0, combining values down the rows and producing one sum for each column.

**Follow-up:** What does `axis=1` produce for a 2-D matrix?

**Answer:** One sum for each row, combining values across columns.

## Question 8: You get a broadcasting error when adding two arrays. How do you debug it?

**Answer:** Print both arrays' shapes. Compare dimensions from the right; corresponding dimensions must match or one must be 1. Reshape or transpose only when it matches the intended mathematical meaning.

**Follow-up:** Would you reshape randomly until the error disappears?

**Answer:** No. A shape-compatible result can still be mathematically wrong. Confirm what each axis represents before reshaping.