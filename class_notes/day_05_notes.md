# Day 05: NumPy Arrays, Indexing and Slicing

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, Mathematics and Statistics Foundations  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

By the end of this lesson, you should be able to:

- Explain why NumPy is used for numerical computing.
- Import NumPy using the common alias.
- Create one-dimensional and two-dimensional arrays.
- Inspect an array's shape, number of dimensions, size and data type.
- Access and update array elements using indexing.
- Extract portions of arrays using slicing.
- Perform element-wise operations and basic aggregations.
- Understand the difference between Python lists and NumPy arrays.
- Recognize common shape, indexing and data-type mistakes.
- Explain how arrays represent numerical features in ML workflows.

## 2. Why do we need NumPy?

Python lists are general-purpose containers. They can hold different types of objects, but numerical work often needs efficient operations over many values. NumPy provides the `ndarray`, a multi-dimensional array designed for numerical computation.

In machine learning, numerical data is commonly represented as:
- A one-dimensional array of values.
- A two-dimensional table of observations and features.
- Higher-dimensional arrays for images, sequences and other data.

### 2.1 A small numerical dataset

### 2.1. Store measurements in a Python list

```python
customer_ages = [25, 30, 35, 40]

print(customer_ages)
```

**Expected output**

```text
[25, 30, 35, 40]
```

**Explanation**

This is a valid Python list. It is flexible and useful for general-purpose collections.



### 2.2 Install or access NumPy

NumPy is already available in many notebook environments. If it is missing in a local environment, install it in that environment using the package manager used by the project.

### 2.3 Import NumPy

### 2.2. Import the NumPy library

```python
import numpy as np

print(np.__version__)
```

**Expected output**

```text
A version string such as 1.x.x or 2.x.x
```

**Explanation**

The alias `np` is the widely used convention. The exact version depends on the environment. Record versions when reproducibility matters.



## 3. Creating NumPy arrays

### 3.1 Convert a Python list into an array

### 3.1. Create a one-dimensional array

```python
import numpy as np

customer_ages = [25, 30, 35, 40]
age_array = np.array(customer_ages)

print(age_array)
print(type(age_array))
```

**Expected output**

```text
[25 30 35 40]
<class 'numpy.ndarray'>
```

**Explanation**

The `np.array()` function creates an ndarray from the list. The result is a NumPy array object.



### 3.2 Create an array directly

### 3.2. Create an array from numeric values

```python
import numpy as np

monthly_sales = np.array([12000, 15000, 18000, 22000])

print(monthly_sales)
```

**Expected output**

```text
[12000 15000 18000 22000]
```

**Explanation**

The values are supplied directly as a list-like sequence. NumPy stores them in an array.



### 3.3 Create a floating-point array

### 3.3. Create decimal-valued measurements

```python
import numpy as np

monthly_charges = np.array([450.5, 520.0, 610.75])

print(monthly_charges)
```

**Expected output**

```text
[450.5  520.   610.75]
```

**Explanation**

The values contain decimal components, so NumPy chooses a floating-point data type.



### 3.4 Create an array with a specified dtype

### 3.4. Specify an integer dtype

```python
import numpy as np

scores = np.array([80, 85, 90], dtype=int)

print(scores)
print(scores.dtype)
```

**Expected output**

```text
[80 85 90]
int64 (platform-dependent)
```

**Explanation**

The `dtype` argument requests a data type. The exact integer width can vary by platform; inspect `scores.dtype` in the active environment.



### 3.5 Create an array with float dtype

### 3.5. Convert integer-looking values to floats

```python
import numpy as np

scores = np.array([80, 85, 90], dtype=float)

print(scores)
print(scores.dtype)
```

**Expected output**

```text
[80. 85. 90.]
float64 (commonly)
```

**Explanation**

The values are represented as floating-point values. This can be useful when a later operation produces decimal results.



### 3.6 Create a two-dimensional array

### 3.6. Create a 2D array

```python
import numpy as np

customer_data = np.array([
    [25, 4500],
    [30, 5200],
    [35, 6100]
])

print(customer_data)
```

**Expected output**

```text
[[  25 4500]
 [  30 5200]
 [  35 6100]]
```

**Explanation**

Each inner list becomes a row. This example has three observations and two numeric columns.



### 3.7 Create an array of zeros

### 3.7. Create an array filled with zeros

```python
import numpy as np

empty_scores = np.zeros(4)

print(empty_scores)
```

**Expected output**

```text
[0. 0. 0. 0.]
```

**Explanation**

The argument gives the requested number of elements. By default, NumPy creates floating-point zeros.



### 3.8 Create an array of ones

### 3.8. Create an array filled with ones

```python
import numpy as np

initial_weights = np.ones(3)

print(initial_weights)
```

**Expected output**

```text
[1. 1. 1.]
```

**Explanation**

This creates three values equal to one. The function is often used when initializing numerical structures.



### 3.9 Create a range of values

### 3.9. Use arange()

```python
import numpy as np

row_numbers = np.arange(0, 5)

print(row_numbers)
```

**Expected output**

```text
[0 1 2 3 4]
```

**Explanation**

The stop value is excluded. `arange(start, stop)` creates regularly spaced values.



### 3.10 Create values with a step

### 3.10. Use arange() with a step

```python
import numpy as np

even_numbers = np.arange(0, 10, 2)

print(even_numbers)
```

**Expected output**

```text
[0 2 4 6 8]
```

**Explanation**

The third argument is the step size. It increases the next value by two each time.



### 3.11 Create evenly spaced values

### 3.11. Use linspace()

```python
import numpy as np

sample_points = np.linspace(0, 1, 5)

print(sample_points)
```

**Expected output**

```text
[0.   0.25 0.5  0.75 1.  ]
```

**Explanation**

The third argument specifies how many values to create between the start and end. Unlike the default use of `arange()`, `linspace()` includes the endpoint by default.



## 4. Understanding array dimensions and shape

### 4.1 What is a dimension?

A one-dimensional array is a sequence. A two-dimensional array has rows and columns. Higher-dimensional arrays contain additional axes.

### 4.2 Inspect the number of dimensions

### 4.1. Check ndim

```python
import numpy as np

one_dimensional = np.array([10, 20, 30])
two_dimensional = np.array([[10, 20], [30, 40]])

print(one_dimensional.ndim)
print(two_dimensional.ndim)
```

**Expected output**

```text
1
2
```

**Explanation**

The `ndim` attribute reports how many dimensions or axes the array has.



### 4.3 Inspect shape

### 4.2. Check the shape of a 1D array

```python
import numpy as np

feature_values = np.array([25, 30, 35, 40])

print(feature_values.shape)
```

**Expected output**

```text
(4,)
```

**Explanation**

The shape is a tuple. `(4,)` means one dimension containing four elements.



### 4.4 Inspect the shape of a 2D array

### 4.3. Check rows and columns

```python
import numpy as np

customer_data = np.array([
    [25, 4500],
    [30, 5200],
    [35, 6100]
])

print(customer_data.shape)
```

**Expected output**

```text
(3, 2)
```

**Explanation**

The shape means three rows and two columns. In ML, rows often represent observations and columns represent features.



### 4.5 Inspect total number of elements

### 4.4. Check size

```python
import numpy as np

customer_data = np.array([
    [25, 4500],
    [30, 5200],
    [35, 6100]
])

print(customer_data.size)
```

**Expected output**

```text
6
```

**Explanation**

The `size` attribute returns the total number of elements: three rows multiplied by two columns.



### 4.6 Inspect data type

### 4.5. Check dtype

```python
import numpy as np

monthly_charges = np.array([450.5, 520.0, 610.75])

print(monthly_charges.dtype)
```

**Expected output**

```text
float64 (commonly)
```

**Explanation**

The `dtype` attribute tells us how NumPy represents the elements. The precise dtype can depend on the values and platform.



## 5. Indexing one-dimensional arrays

NumPy arrays use zero-based indexing, like Python lists.

### 5.1 Read the first element

### 5.1. Access index zero

```python
import numpy as np

monthly_sales = np.array([12000, 15000, 18000, 22000])

print(monthly_sales[0])
```

**Expected output**

```text
12000
```

**Explanation**

Index zero selects the first element.



### 5.2 Read another element

### 5.2. Access index two

```python
import numpy as np

monthly_sales = np.array([12000, 15000, 18000, 22000])

print(monthly_sales[2])
```

**Expected output**

```text
18000
```

**Explanation**

Index two selects the third element.



### 5.3 Use negative indexing

### 5.3. Access the final element

```python
import numpy as np

monthly_sales = np.array([12000, 15000, 18000, 22000])

print(monthly_sales[-1])
```

**Expected output**

```text
22000
```

**Explanation**

Negative index `-1` selects the final element.



### 5.4 Update an element

### 5.4. Change a value by index

```python
import numpy as np

monthly_sales = np.array([12000, 15000, 18000, 22000])

monthly_sales[1] = 16000

print(monthly_sales)
```

**Expected output**

```text
[12000 16000 18000 22000]
```

**Explanation**

NumPy arrays are mutable. The element at index one is replaced.



## 6. Slicing one-dimensional arrays

The slicing pattern is similar to Python lists: `array[start:stop:step]`. The stop index is excluded.

### 6.1 Select a range

### 6.1. Select elements from index one to three

```python
import numpy as np

values = np.array([10, 20, 30, 40, 50])

print(values[1:4])
```

**Expected output**

```text
[20 30 40]
```

**Explanation**

The slice begins at index one and stops before index four.



### 6.2 Select the first elements

### 6.2. Slice from the beginning

```python
import numpy as np

values = np.array([10, 20, 30, 40, 50])

print(values[:3])
```

**Expected output**

```text
[10 20 30]
```

**Explanation**

When the start is omitted, slicing begins at the first element.



### 6.3 Select through the end

### 6.3. Slice to the end

```python
import numpy as np

values = np.array([10, 20, 30, 40, 50])

print(values[2:])
```

**Expected output**

```text
[30 40 50]
```

**Explanation**

When the stop is omitted, the slice continues through the last element.



### 6.4 Select every second value

### 6.4. Use a slicing step

```python
import numpy as np

values = np.array([10, 20, 30, 40, 50, 60])

print(values[::2])
```

**Expected output**

```text
[10 30 50]
```

**Explanation**

The step of two selects every second element.



### 6.5 Reverse an array

### 6.5. Use a negative step

```python
import numpy as np

values = np.array([10, 20, 30, 40])

print(values[::-1])
```

**Expected output**

```text
[40 30 20 10]
```

**Explanation**

The negative step traverses the array from the end towards the beginning.



## 7. Indexing two-dimensional arrays

For a two-dimensional array, use `array[row_index, column_index]`.

### 7.1 Create a small table

```python
import numpy as np

customer_data = np.array([
    [25, 4500],
    [30, 5200],
    [35, 6100]
])

print(customer_data)
```

**Expected output**

```text
[[  25 4500]
 [  30 5200]
 [  35 6100]]
```

**Explanation**

The first column can represent age and the second column can represent monthly charges. This is a small educational matrix, not a replacement for a labelled DataFrame.

### 7.2 Access the first row

### 7.1. Select row zero

```python
import numpy as np

customer_data = np.array([[25, 4500], [30, 5200], [35, 6100]])

print(customer_data[0])
```

**Expected output**

```text
[  25 4500]
```

**Explanation**

The first index identifies the row. Omitting the column index returns the whole row.



### 7.3 Access one value

### 7.2. Select row one and column zero

```python
import numpy as np

customer_data = np.array([[25, 4500], [30, 5200], [35, 6100]])

print(customer_data[1, 0])
```

**Expected output**

```text
30
```

**Explanation**

The first index selects the second row; the second index selects the first column.



### 7.4 Access the final column

### 7.3. Select the second column

```python
import numpy as np

customer_data = np.array([[25, 4500], [30, 5200], [35, 6100]])

print(customer_data[:, 1])
```

**Expected output**

```text
[4500 5200 6100]
```

**Explanation**

The colon means all rows. Index one selects the second column.



### 7.5 Access the first column

### 7.4. Select the first column

```python
import numpy as np

customer_data = np.array([[25, 4500], [30, 5200], [35, 6100]])

print(customer_data[:, 0])
```

**Expected output**

```text
[25 30 35]
```

**Explanation**

The colon selects all rows; zero selects the first column.



### 7.6 Select multiple rows and columns

### 7.5. Slice a rectangular section

```python
import numpy as np

customer_data = np.array([
    [25, 4500, 2],
    [30, 5200, 4],
    [35, 6100, 6],
    [40, 7000, 8]
])

print(customer_data[1:3, 0:2])
```

**Expected output**

```text
[[  30 5200]
 [  35 6100]]
```

**Explanation**

The row slice selects rows one and two. The column slice selects columns zero and one.



### 7.7 Update a 2D element

### 7.6. Update a selected cell

```python
import numpy as np

customer_data = np.array([[25, 4500], [30, 5200]])

customer_data[0, 1] = 4700

print(customer_data)
```

**Expected output**

```text
[[  25 4700]
 [  30 5200]]
```

**Explanation**

The value in row zero, column one is replaced.



## 8. Vectorized operations

Vectorization means applying an operation to an entire array without writing an explicit Python loop for each element.

### 8.1 Add a value to every element

### 8.1. Increase every value

```python
import numpy as np

prices = np.array([100, 200, 300])

updated_prices = prices + 10

print(updated_prices)
```

**Expected output**

```text
[110 210 310]
```

**Explanation**

NumPy applies the addition element by element. This is called a vectorized operation.



### 8.2 Multiply every element

### 8.2. Multiply an array

```python
import numpy as np

prices = np.array([100, 200, 300])

discounted_base = prices * 0.9

print(discounted_base)
```

**Expected output**

```text
[ 90. 180. 270.]
```

**Explanation**

The multiplication is applied to every element. NumPy returns floating-point results here.



### 8.3 Add two arrays element by element

### 8.3. Combine two arrays

```python
import numpy as np

online_orders = np.array([10, 20, 30])
store_orders = np.array([5, 8, 12])

total_orders = online_orders + store_orders

print(total_orders)
```

**Expected output**

```text
[15 28 42]
```

**Explanation**

Values at matching positions are added. The arrays must have compatible shapes.



### 8.4 Compare a Python list and NumPy array

### 8.4. Observe list concatenation

```python
python_list = [10, 20, 30]

print(python_list + [1])
```

**Expected output**

```text
[10, 20, 30, 1]
```

**Explanation**

For lists, `+` concatenates collections; it does not add one to each numeric value.



### 8.5. Observe NumPy element-wise addition

```python
import numpy as np

numpy_array = np.array([10, 20, 30])

print(numpy_array + 1)
```

**Expected output**

```text
[11 21 31]
```

**Explanation**

For NumPy arrays, adding a scalar applies the operation to every element.



### 8.5 Why this matters

A model often receives many feature values. Vectorized operations let us express numerical transformations clearly and use optimized numerical routines.

## 9. Aggregation functions

### 9.1 sum()

### 9.1. Calculate a total

```python
import numpy as np

monthly_sales = np.array([12000, 15000, 18000])

print(np.sum(monthly_sales))
```

**Expected output**

```text
45000
```

**Explanation**

The `np.sum()` function calculates the total of the array values.



### 9.2 mean()

### 9.2. Calculate an average

```python
import numpy as np

monthly_sales = np.array([12000, 15000, 18000])

print(np.mean(monthly_sales))
```

**Expected output**

```text
15000.0
```

**Explanation**

The mean returns the arithmetic average of the values.



### 9.3 min() and max()

### 9.3. Find minimum and maximum

```python
import numpy as np

scores = np.array([0.81, 0.92, 0.87, 0.79])

print(np.min(scores))
print(np.max(scores))
```

**Expected output**

```text
0.79
0.92
```

**Explanation**

The functions return the smallest and largest values.



### 9.4 median()

### 9.4. Calculate the median

```python
import numpy as np

monthly_sales = np.array([1000, 1200, 1300, 1500, 9000])

print(np.median(monthly_sales))
```

**Expected output**

```text
1300.0
```

**Explanation**

The median is the middle value after ordering the observations. It is less affected by an extreme value than the mean in this example.



### 9.5 Standard deviation

### 9.5. Inspect variation

```python
import numpy as np

model_scores = np.array([0.80, 0.82, 0.84, 0.86])

print(np.std(model_scores))
```

**Expected output**

```text
0.02236067977499788
```

**Explanation**

Standard deviation describes how spread out the values are around their mean. NumPy's default population standard deviation uses `ddof=0`; use the appropriate setting for the question being asked.



## 10. Operations across rows and columns

For a 2D array, the `axis` argument controls the direction of an aggregation.

### 10.1 Sum all values

### 10.1. Sum a complete 2D array

```python
import numpy as np

sales = np.array([[10, 20], [30, 40]])

print(np.sum(sales))
```

**Expected output**

```text
100
```

**Explanation**

Without an axis argument, the function aggregates all elements.



### 10.2 Sum each column

### 10.2. Sum down the rows using axis zero

```python
import numpy as np

sales = np.array([[10, 20], [30, 40]])

print(np.sum(sales, axis=0))
```

**Expected output**

```text
[40 60]
```

**Explanation**

Axis zero collapses the row dimension, leaving one result for each column.



### 10.3 Sum each row

### 10.3. Sum across columns using axis one

```python
import numpy as np

sales = np.array([[10, 20], [30, 40]])

print(np.sum(sales, axis=1))
```

**Expected output**

```text
[30 70]
```

**Explanation**

Axis one collapses the column dimension, leaving one result for each row.



### 10.4 Mean for each column

### 10.4. Calculate column means

```python
import numpy as np

customer_data = np.array([[20, 100], [30, 200], [40, 300]])

print(np.mean(customer_data, axis=0))
```

**Expected output**

```text
[ 30. 200.]
```

**Explanation**

The result contains one mean per column. Ensure that columns are comparable and represent the intended features before interpreting the result.



## 11. Boolean filtering

A comparison on an array creates a Boolean array. That Boolean array can be used to select matching values.

### 11.1 Create a Boolean condition

### 11.1. Check values greater than 20

```python
import numpy as np

ages = np.array([18, 25, 32, 16, 40])

condition = ages > 20

print(condition)
```

**Expected output**

```text
[False  True  True False  True]
```

**Explanation**

Each element is compared with 20. The result contains one Boolean value for each input value.



### 11.2 Filter using the condition

### 11.2. Select ages above 20

```python
import numpy as np

ages = np.array([18, 25, 32, 16, 40])

print(ages[ages > 20])
```

**Expected output**

```text
[25 32 40]
```

**Explanation**

The Boolean mask selects only the positions where the condition is true.



### 11.3 Filter with two conditions

### 11.3. Combine two conditions with element-wise and

```python
import numpy as np

ages = np.array([18, 25, 32, 16, 40])

selected = ages[(ages >= 20) & (ages <= 35)]

print(selected)
```

**Expected output**

```text
[25 32]
```

**Explanation**

Use `&` for element-wise logical AND in NumPy. Put each comparison inside parentheses. Python's scalar `and` is not used to combine array masks.



### 11.4 Filter with either condition

### 11.4. Combine two conditions with element-wise or

```python
import numpy as np

ages = np.array([18, 25, 32, 16, 40])

selected = ages[(ages < 20) | (ages > 35)]

print(selected)
```

**Expected output**

```text
[18 16 40]
```

**Explanation**

The `|` operator combines the Boolean arrays element by element. Parentheses are important around comparisons.



## 12. Reshaping arrays

Reshaping changes an array's dimensions without changing the order of its elements. The requested shape must contain the same number of elements.

### 12.1 Reshape a 1D array into 2D

### 12.1. Create two rows and three columns

```python
import numpy as np

values = np.array([1, 2, 3, 4, 5, 6])

matrix = values.reshape(2, 3)

print(matrix)
```

**Expected output**

```text
[[1 2 3]
 [4 5 6]]
```

**Explanation**

The original array has six elements. The new shape also contains six positions.



### 12.2 Reshape into three rows and two columns

### 12.2. Change the layout

```python
import numpy as np

values = np.array([1, 2, 3, 4, 5, 6])

matrix = values.reshape(3, 2)

print(matrix)
```

**Expected output**

```text
[[1 2]
 [3 4]
 [5 6]]
```

**Explanation**

The elements are placed into the new shape in order.



### 12.3 Invalid reshape

```python
import numpy as np

values = np.array([1, 2, 3, 4, 5, 6])

matrix = values.reshape(4, 2)
```

**Expected result**

```text
ValueError: cannot reshape array of size 6 into shape (4,2)
```

**Explanation**

A shape of four rows and two columns requires eight elements. The source contains only six. The requested shape is incompatible.

## 13. Broadcasting basics

Broadcasting allows NumPy to perform operations between arrays with compatible shapes, including an array and a scalar.

### 13.1 Add a scalar to a 1D array

### 13.1. Broadcast a scalar

```python
import numpy as np

feature_values = np.array([10, 20, 30])

print(feature_values + 5)
```

**Expected output**

```text
[15 25 35]
```

**Explanation**

The scalar value five is applied to every element.



### 13.2 Add a row-shaped array to a matrix

### 13.2. Broadcast across rows

```python
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6]])
offset = np.array([10, 20, 30])

print(matrix + offset)
```

**Expected output**

```text
[[11 22 33]
 [14 25 36]]
```

**Explanation**

The one-dimensional array is compatible with the matrix's column dimension and is applied to each row.



### 13.3 Incompatible shapes

```python
import numpy as np

first = np.array([1, 2, 3])
second = np.array([10, 20])

print(first + second)
```

**Expected result**

```text
ValueError: operands could not be broadcast together with shapes (3,) (2,)
```

**Explanation**

The arrays have incompatible shapes for this operation. Inspect `.shape` before combining arrays.

## 14. Copying and views

Slicing an array often returns a view that refers to the same underlying data. A change through a view can affect the original array.

### 14.1 Create a slice view

### 14.1. Modify a slice

```python
import numpy as np

original = np.array([10, 20, 30, 40])

selected = original[1:3]
selected[0] = 999

print(original)
print(selected)
```

**Expected output**

```text
[ 10 999  30  40]
[999  30]
```

**Explanation**

In this example, the slice shares data with the original array. Updating the selected view changes the original array.



### 14.2 Create an independent copy

### 14.2. Use copy()

```python
import numpy as np

original = np.array([10, 20, 30, 40])

selected = original[1:3].copy()
selected[0] = 999

print(original)
print(selected)
```

**Expected output**

```text
[10 20 30 40]
[999  30]
```

**Explanation**

The `copy()` method creates independent array data for this selection. Updating the copy does not change the original.



## 15. Common NumPy mistakes

### 15.1 Confusing list and array addition

- Python list `+` concatenates lists.
- NumPy array `+` performs element-wise addition when shapes are compatible.

### 15.2 Incorrect index

An index outside the valid range raises `IndexError`. Inspect the shape and remember zero-based indexing.

### 15.3 Incorrect shape

An operation may fail when arrays have incompatible dimensions. Print both shapes before combining them.

### 15.4 Unexpected data type

A numeric-looking value may be stored as text. Inspect `dtype` and clean or convert values deliberately.

### 15.5 Using scalar and array logical operators interchangeably

Use `and` and `or` for scalar Boolean conditions. Use `&` and `|` with parentheses to combine NumPy Boolean masks.

### 15.6 Assuming slicing always copies data

A slice can be a view. Use `copy()` when you require an independent array.

## 16. NumPy in an ML workflow

A common ML workflow uses tabular data for inspection and preparation, then numerical arrays for model operations. Scikit-learn estimators commonly accept array-like inputs and often interoperate with NumPy.

### 16.1 Represent a feature matrix

```python
import numpy as np

features = np.array([
    [25, 4500],
    [30, 5200],
    [35, 6100]
])

print(features.shape)
```

**Expected output**

```text
(3, 2)
```

**Explanation:** There are three observations and two feature columns in this simple example.

### 16.2 Represent a target vector

```python
import numpy as np

target = np.array([0, 1, 0])

print(target.shape)
```

**Expected output**

```text
(3,)
```

**Explanation:** The target has one label for each observation. The number of rows in the feature matrix and the number of target values should match for ordinary supervised learning.

### 16.3 Check row alignment

```python
import numpy as np

features = np.array([[25, 4500], [30, 5200], [35, 6100]])
target = np.array([0, 1, 0])

print(features.shape[0] == target.shape[0])
```

**Expected output**

```text
True
```

**Explanation:** This checks that the number of feature rows equals the number of target labels. It is a basic shape check, not a full data validation process.

## 17. Quick revision table

| Concept | Meaning |
|---|---|
| ndarray | NumPy's multi-dimensional array |
| `np.array()` | Create an array |
| `dtype` | Element data type |
| `ndim` | Number of dimensions |
| `shape` | Length of each dimension |
| `size` | Total number of elements |
| Indexing | Access a value by position |
| Slicing | Select a range of values |
| Vectorization | Apply operations to arrays without explicit Python loops |
| Boolean mask | Select elements using conditions |
| `reshape()` | Change array dimensions |
| Broadcasting | Perform operations on compatible shapes |
| View | A selection that can share underlying data |
| `copy()` | Create independent array data |

## 18. Interview-ready explanations

**What is NumPy and why is it used in machine learning?**  
NumPy is a numerical computing library. Its ndarray supports multi-dimensional numerical data, efficient element-wise operations, indexing, aggregation and shape-aware processing. It is widely used as a foundation for scientific Python and ML tools.

**What is the difference between a Python list and a NumPy array?**  
A list is a general-purpose collection and can hold mixed object types. A NumPy array is designed for numerical operations, commonly stores a consistent dtype and supports vectorized operations and multi-dimensional shapes.

**What is the difference between shape, size and ndim?**  
Shape gives the length of each dimension, size gives the total number of elements, and ndim gives the number of dimensions.

**What is broadcasting?**  
Broadcasting allows NumPy to apply operations between arrays with compatible shapes, including applying a scalar to every array element. Incompatible shapes produce an error.

**Why should you inspect array shapes before model training?**  
The feature matrix and target vector must be aligned. Shape checks can catch mismatched row counts or unexpected dimensions before the estimator is called.

## 19. Lesson completion checklist

- [ ] I can import NumPy.
- [ ] I can create 1D and 2D arrays.
- [ ] I can inspect dtype, ndim, shape and size.
- [ ] I can use positive and negative indexing.
- [ ] I can slice rows and columns.
- [ ] I understand vectorized arithmetic.
- [ ] I can use basic aggregation functions.
- [ ] I can create and use Boolean masks.
- [ ] I can reshape compatible arrays.
- [ ] I understand basic broadcasting.
- [ ] I can explain why shape validation matters in ML.

**Next lesson:** Day 06 — Pandas Series, DataFrames and File Reading. We will use labelled tables to read, inspect and select real-world data before machine-learning preparation.
