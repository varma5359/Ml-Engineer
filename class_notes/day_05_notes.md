# Day 05: NumPy Arrays, Indexing and Slicing

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate  
**Learning focus:** Understand the concepts, apply them correctly, and explain decisions in an interview.

---

## 1. Learning Objectives

- Explain why NumPy is used in numerical computing and ML.
- Create one-dimensional and multi-dimensional arrays.
- Understand shape, dimensions, size and data type.
- Access and modify elements using indexing.
- Extract subsets using slicing.
- Perform vectorized arithmetic and aggregation.
- Understand broadcasting and common shape errors.

## 2. What Is NumPy?

NumPy (Numerical Python) is a Python library for efficient numerical computation. Its central object is the `ndarray`, a multi-dimensional array usually containing values of a consistent data type.

Why ML engineers use NumPy:

- Store numerical features efficiently.
- Perform arithmetic on whole arrays without writing a Python loop for every element.
- Calculate sums, averages and other statistics.
- Represent matrices and vectors used in ML mathematics.
- Exchange numerical data with libraries such as Pandas and scikit-learn.

## 3. Create NumPy Arrays

```python
import numpy as np

scores = np.array([78, 85, 92, 88])
print(scores)
print(type(scores))
```

Output:

```text
[78 85 92 88]
<class 'numpy.ndarray'>
```

`np.array()` converts a Python sequence into a NumPy array.

### Common creation functions

```python
import numpy as np

print(np.zeros(3))
print(np.ones(4))
print(np.arange(1, 6))
print(np.linspace(0, 1, 5))
```

Output:

```text
[0. 0. 0.]
[1. 1. 1. 1.]
[1 2 3 4 5]
[0.   0.25 0.5  0.75 1.  ]
```

| Function | Purpose |
|---|---|
| `np.array()` | Create an array from a sequence |
| `np.zeros(shape)` | Fill an array with zeros |
| `np.ones(shape)` | Fill an array with ones |
| `np.arange(start, stop, step)` | Generate values using a step; stop is excluded |
| `np.linspace(start, stop, num)` | Generate a requested number of evenly spaced values |
| `np.eye(n)` | Create an identity matrix |

## 4. Array Dimensions and Shape

```python
features = np.array([
    [25, 5000],
    [31, 6200],
    [40, 7100]
])

print(features.ndim)
print(features.shape)
print(features.size)
print(features.dtype)
```

Output (dtype may vary with values/platform details):

```text
2
(3, 2)
6
int64
```

- `ndim`: number of dimensions (axes).
- `shape`: number of elements along each axis; here 3 rows and 2 columns.
- `size`: total number of elements.
- `dtype`: data type used by the array.

### Visual idea

```text
2-D array shape = (3, 2)

             Column 0   Column 1
Row 0           25        5000
Row 1           31        6200
Row 2           40        7100
```

A 1-D array has one axis, a 2-D array has rows and columns, and a 3-D array adds another axis (for example, height × width × color channel for an image).

## 5. Indexing — Access Individual Values

Python and NumPy use zero-based indexing.

```python
values = np.array([10, 20, 30, 40, 50])

print(values[0])
print(values[2])
print(values[-1])
```

Output:

```text
10
30
50
```

### Index a 2-D array

```python
marks = np.array([
    [80, 75, 90],
    [88, 92, 81],
    [70, 85, 78]
])

print(marks[0, 1])
print(marks[2, 2])
```

Output:

```text
75
78
```

`marks[row_index, column_index]` selects one value.

### Modify a value

```python
values = np.array([10, 20, 30])
values[1] = 99
print(values)
```

Output: `[10 99 30]`

## 6. Slicing — Access a Range of Values

General 1-D form:

```python
array[start:stop:step]
```

The start index is included; the stop index is excluded.

```python
values = np.array([10, 20, 30, 40, 50, 60])

print(values[1:4])
print(values[:3])
print(values[3:])
print(values[::2])
print(values[::-1])
```

Output:

```text
[20 30 40]
[10 20 30]
[40 50 60]
[10 30 50]
[60 50 40 30 20 10]
```

### 2-D slicing

```python
data = np.array([
    [10, 100, 1],
    [20, 200, 2],
    [30, 300, 3],
    [40, 400, 4]
])

print(data[:, 0])
print(data[1:3, 1:3])
```

Output:

```text
[10 20 30 40]
[[200   2]
 [300   3]]
```

- `:` means select all elements along that axis.
- `data[:, 0]` selects every row from column 0.
- `data[1:3, 1:3]` selects rows 1 and 2, columns 1 and 2.

## 7. Vectorized Operations

NumPy performs element-wise operations on arrays.

```python
prices = np.array([100, 200, 300])
quantities = np.array([2, 1, 4])

line_totals = prices * quantities
print(line_totals)
print(line_totals.sum())
```

Output:

```text
[ 200  200 1200]
1600
```

No explicit loop is needed for the element-wise multiplication.

### Aggregation functions

```python
scores = np.array([70, 80, 90, 100])

print(np.sum(scores))
print(np.mean(scores))
print(np.min(scores))
print(np.max(scores))
print(np.std(scores))
```

Output:

```text
340
85.0
70
100
11.180339887498949
```

By default, `np.std()` calculates population standard deviation (`ddof=0`). Use `ddof=1` for the sample standard deviation when that is the intended statistic.

## 8. Axis in 2-D Arrays

`axis` tells NumPy which dimension to reduce.

```python
sales = np.array([
    [100, 120, 90],
    [80, 110, 130]
])

print(sales.sum(axis=0))
print(sales.sum(axis=1))
```

Output:

```text
[180 230 220]
[310 320]
```

- `axis=0`: sum down the rows, producing one result per column.
- `axis=1`: sum across columns, producing one result per row.

Remember it by looking at the dimension that disappears from the result.

## 9. Broadcasting

Broadcasting allows compatible arrays with different shapes to participate in element-wise operations.

```python
monthly_sales = np.array([
    [100, 120, 90],
    [80, 110, 130]
])

adjustment = np.array([10, 0, 5])
print(monthly_sales + adjustment)
```

Output:

```text
[[110 120  95]
 [ 90 110 135]]
```

The 1-D array of shape `(3,)` is applied to each row of the `(2, 3)` array. Broadcasting works only when dimensions are compatible (equal or one of them is 1, compared from the right).

## 10. Boolean Filtering

```python
scores = np.array([45, 67, 82, 91, 38])
passed = scores[scores >= 50]
print(passed)
```

Output: `[67 82 91]`

The condition creates a Boolean mask. Indexing with that mask selects values where the condition is true.

## 11. Industry Scenario — Prepare Customer Features

A bank has a small set of numeric customer records with columns age, monthly income and existing loan count.

```python
customers = np.array([
    [25, 30000, 1],
    [40, 65000, 2],
    [32, 45000, 0],
    [51, 80000, 1]
])

ages = customers[:, 0]
incomes = customers[:, 1]

print("Average age:", ages.mean())
print("Average income:", incomes.mean())
print("High income rows:")
print(customers[incomes >= 60000])
```

Expected output:

```text
Average age: 37.0
Average income: 55000.0
High income rows:
[[   40 65000     2]
 [   51 80000     1]]
```

This illustrates selecting columns and filtering rows. It does not establish that income or age is suitable for a particular lending decision; feature use requires governance, fairness review, legal compliance and business validation.

## 12. Common Shape and Data-Type Problems

| Problem | Why it happens | What to check |
|---|---|---|
| `IndexError` | Index is outside array bounds | Print `shape` and inspect indices |
| Broadcasting error | Shapes are incompatible | Compare dimensions from right to left |
| Unexpected decimals | Array type was promoted to float | Inspect `dtype` |
| Truncated integer values | Assigning a decimal to integer array can cast/truncate | Create an appropriate float array |
| Wrong column selected | Misunderstood row/column axis | Write down the array shape and intended axis |
| Mixed data causes string dtype | One or more values are non-numeric | Validate and convert before numerical operations |

## 13. Quick Revision

- NumPy's main structure is `ndarray`.
- `ndim`, `shape`, `size` and `dtype` describe an array.
- Indexing selects individual values; slicing selects ranges.
- Operations are commonly element-wise and vectorized.
- `axis=0` reduces rows; `axis=1` reduces columns.
- Broadcasting applies compatible shapes; always inspect shapes when results look wrong.

---

## Expanded Beginner-to-Developer Teaching Layer

# Day 05: NumPy arrays, indexing and slicing

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner foundations  
**Learning focus:** Build understanding from first principles, implement correctly and explain engineering decisions.

---

## 1. Start Here: What Will You Learn?

By the end of this lesson, you should be able to:
- Define the key terms in simple language.
- Explain why the concept is needed and which problem it solves.
- Write and read the core syntax without memorizing blindly.
- Apply the concept to a small practical example.
- Recognize common errors and explain how to investigate them.
- Discuss a realistic business scenario and its limitations.

## 2. Beginner Definition

**NumPy arrays, indexing and slicing**

NumPy is a Python library for efficient numerical arrays and vectorized calculations. An ndarray stores elements in a structured n-dimensional array, commonly with one data type.

### Explain it in everyday language

Think of this as a tool in a developer's toolbox. Before using it, identify the problem, understand what information goes in, what result should come out, and what assumptions are being made. The same technique can be useful in one situation and unsuitable in another.

### Key terms

| Term | Beginner-friendly meaning | Why it matters |
|---|---|---|
| Input | Information supplied to the process | Incorrect input produces unreliable results |
| Output | Result produced by the process | Must match the business requirement |
| Parameter | A value supplied to control an operation or learned by a model, depending on context | Know whether it is configured or learned |
| Validation | Checking whether the result is correct and suitable | Prevents silent errors |
| Generalization | Ability to work on new, unseen cases | Essential for real ML systems |

## 3. Why Do We Need It?

Use NumPy for numerical datasets, matrix operations, statistical calculations and efficient element-wise operations.

**Without this concept or step:** teams may produce incorrect outputs, waste compute, make poor business decisions or trust a result that does not work on new data. The exact risk depends on the application; investigate rather than assume.

## 4. Syntax, Structure and How to Read It

Avoid assuming arrays behave exactly like lists; check shape, dtype and broadcasting compatibility before combining arrays.

### Read the syntax carefully

1. Identify the library or Python feature being used.
2. Identify the input variable and its expected type/shape.
3. Identify the operation or method being called.
4. Check optional parameters and their defaults.
5. Inspect the output and verify that it matches the intended result.

### Practical example

```python
# A small, readable example related to the lesson
import numpy as np\nreadings = np.array([28.5, 29.0, 30.2, 27.8])\nprint(readings.mean())
```

**Expected result:** The example demonstrates the basic operation. Exact numeric output depends on the input values and library version; run it in the same environment to verify.

### Line-by-line explanation

- The import statement loads the required library or class.
- The input variable stores the data the operation will use.
- The method/function performs the requested operation.
- The print statement displays the result so that it can be checked.
- In a real application, validate inputs and handle expected exceptions rather than assuming every record is valid.

## 5. Worked Examples: From Simple to Practical

### Example A — Understand the smallest case

Start with a tiny input that can be checked manually. Predict what the output should be before running the code. This helps separate a coding error from a misunderstanding of the concept.

### Example B — Change one input

Change one value while keeping the rest constant. Observe which part of the output changes. This is a simple way to understand the effect of a parameter or feature.

### Example C — Check a boundary case

Test an empty input, a missing value, a very small value and a very large value where appropriate. Boundary testing often reveals bugs that normal examples do not show.

## 6. Industry Scenario: Think Like a Senior Developer

**Scenario:** A sensor platform receives thousands of temperature readings. NumPy can calculate averages, filter ranges and transform all readings without writing a Python loop for every element.

### How an experienced engineer approaches it

1. Clarify the business objective and what decision the output will support.
2. Identify the data available at the moment the prediction/operation must happen.
3. Check data quality, definitions, units and edge cases.
4. Choose the simplest method that meets the requirement.
5. Validate on representative cases, including difficult cases.
6. Explain uncertainty, limitations and the cost of mistakes.
7. Monitor the result after release; real-world data and business behaviour can change.

### What could go wrong?

undefined

### Senior developer's decision

Syntax: import numpy as np; a=np.array([10,20,30]); a[0]; a[1:]; a.shape; a.ndim; a.dtype; a.mean(); a[a>20]. Indexing starts at zero; slicing stop index is excluded.

## 7. Common Beginner Mistakes and Corrections

| Mistake | Why it causes trouble | Better approach |
|---|---|---|
| Copying syntax without understanding inputs | Code may run on the wrong data | Check type, shape, units and meaning first |
| Looking only at one successful example | Edge cases remain hidden | Test normal, boundary and invalid inputs |
| Ignoring warnings or unexpected output | Errors can silently affect downstream steps | Inspect output and add assertions/checks |
| Choosing a complex method immediately | Complexity makes debugging and maintenance harder | Establish a simple baseline first |
| Reporting a metric without context | A number alone may hide business failure | Explain the metric, test data and error costs |

## 8. Quick Revision Table

| Question | Key point |
|---|---|
| What is it? | NumPy is a Python library for efficient numerical arrays and vectorized calculations. An ndarray stores elements in a structured n-dimensional array, commonly with one data type. |
| Why use it? | Use NumPy for numerical datasets, matrix operations, statistical calculations and efficient element-wise operations. |
| Main practical use | A sensor platform receives thousands of temperature readings. NumPy can calculate averages, filter ranges and transform all readings without writing a Python loop for every element. |
| Main risk | undefined |
| Engineering principle | Validate assumptions and unseen/boundary cases |

## 9. Check Your Understanding

Try answering without looking at the notes:
1. Define the concept in your own words.
2. What problem does it solve?
3. What are the inputs and expected outputs?
4. When would you avoid using it?
5. What would you test before trusting the result?

## 10. Interview Preparation

Use the matching `day_05_interview.md` file for detailed interview questions, model answers, real-world troubleshooting scenarios and follow-up questions.

---

**Learning standard:** Understand the meaning first, then syntax, then practical use, then engineering trade-offs. A student should be able to explain the idea without reading code from the screen.
