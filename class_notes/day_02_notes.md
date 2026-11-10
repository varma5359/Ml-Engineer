# Day 02: Python Environment, Variables, Data Types and Operators

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, Mathematics and Statistics Foundations  
**Level:** Beginner  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

By the end of this lesson, you should be able to:

- Open a Python environment and run code in a notebook or script.
- Understand how a Python cell is executed.
- Create meaningful variable names.
- Store business and machine-learning information in variables.
- Identify integers, floats, strings, Booleans and missing values.
- Inspect values using `type()` and validate them using `isinstance()`.
- Convert compatible values between data types.
- Use arithmetic, comparison, logical and assignment operators.
- Understand operator precedence in everyday Python expressions.
- Recognize common data-type errors before beginning data preparation.

## 2. Why Python is used in machine learning

Machine learning work involves more than calling a model's `fit()` method. An engineer regularly:

- Reads values from files, databases and APIs.
- Stores file paths, column names and configuration values.
- Checks whether a feature contains numbers or text.
- Creates new values from existing columns.
- Compares model settings and experiment results.
- Checks whether a prediction meets a business condition.
- Converts data into a format accepted by a library.

Python variables and data types are the foundation for all these tasks.

### 2.1 A simple machine-learning workflow

```text
Business data
    ↓
Python variables and data types
    ↓
Data inspection and cleaning
    ↓
Feature preparation
    ↓
Model training
    ↓
Prediction and evaluation
```

At every stage, Python needs to know what kind of value it is handling.

## 3. Setting up the Python environment

### 3.1 What is a Python environment?

A Python environment is the place where Python code is written and executed. It includes the Python interpreter and the packages needed by a program.

Common choices:

| Environment | How it is used |
|---|---|
| Google Colab | Write and run notebook cells in a browser |
| Jupyter Notebook | Run code cell by cell |
| VS Code | Write Python scripts and notebooks |
| Python terminal | Run a Python file or interactive commands |

For the first lessons, a notebook is convenient because students can run one step, inspect the output and continue.

### 3.2 Starting a notebook

1. Open Google Colab or Jupyter Notebook.
2. Create a new Python notebook.
3. Rename it to `day_02_python_fundamentals.ipynb`.
4. Select a Python runtime or kernel.
5. Create a code cell.
6. Type the small example shown below.
7. Run only that cell and observe the output.

### 3.3 First Python cell

**Code cell 1 — Confirm that Python is running**

```python
print("Machine Learning Engineer - Day 02")
```

**Expected output**

```text
Machine Learning Engineer - Day 02
```

**Explanation**

- `print()` is a built-in Python function.
- The text inside quotation marks is a string.
- Python executes the statement when the cell runs.
- The output appears immediately below the cell.

**Why this matters:** If this cell does not run, solve the notebook/runtime issue before continuing. Do not confuse a notebook connection problem with a Python syntax problem.

### 3.4 Running cells in sequence

A notebook keeps variables created by earlier cells in the active kernel.

**Code cell 2 — Create a variable**

```python
course_name = "Machine Learning Engineer"
print(course_name)
```

**Expected output**

```text
Machine Learning Engineer
```

**Explanation**

The name `course_name` refers to the string value. After this cell runs, later cells can use the same name while the kernel remains active.

**Code cell 3 — Reuse the variable**

```python
print("Welcome to", course_name)
```

**Expected output**

```text
Welcome to Machine Learning Engineer
```

**Explanation**

This cell uses the variable created in Code cell 2. It does not need to create the value again.

**Common issue:** If Code cell 3 is run before Code cell 2, Python may report `NameError`. Run the cell that defines the variable first, or restart and run the notebook from the beginning.

## 4. What is a variable?

A variable is a name that refers to a value or object used by a program.

Think about a customer dataset. It may contain a customer's age, monthly spending, membership type and whether the customer left the service. We can store one example of each value in a Python variable.

### 4.1 First variable example

**Code cell 4 — Store one customer record**

```python
customer_age = 29
monthly_spend = 4500.75
membership_type = "Premium"
customer_left = False

print(customer_age)
print(monthly_spend)
print(membership_type)
print(customer_left)
```

**Expected output**

```text
29
4500.75
Premium
False
```

**Explanation**

- `customer_age` refers to the integer `29`.
- `monthly_spend` refers to the decimal value `4500.75`.
- `membership_type` refers to text.
- `customer_left` refers to a Boolean value.
- Each variable has a descriptive name so that another developer can understand its purpose.

### 4.2 Why do we need variables?

Without variables, a value may be repeated in many places.

**Code cell 5 — Repeating values directly**

```python
print(1200 + 800)
print((1200 + 800) / 2)
print((1200 + 800) * 0.10)
```

**Expected output**

```text
2000
1000.0
200.0
```

**Explanation**

The code works, but the same values are repeated. If the values change, each expression must be checked and updated.

**Code cell 6 — Store values once**

```python
january_orders = 1200
february_orders = 800

total_orders = january_orders + february_orders
average_orders = total_orders / 2
estimated_sample = total_orders * 0.10

print(total_orders)
print(average_orders)
print(estimated_sample)
```

**Expected output**

```text
2000
1000.0
200.0
```

**Explanation**

- The monthly values are stored once.
- `total_orders` reuses both variables.
- `average_orders` reuses the calculated total.
- The code is easier to read and update.

### 4.3 A variable is a name, not a permanent box

A useful Python mental model is that a variable name refers to an object. Assigning another value makes the name refer to the new object.

**Code cell 7 — Reassign a variable**

```python
training_accuracy = 0.82
print(training_accuracy)

training_accuracy = 0.88
print(training_accuracy)
```

**Expected output**

```text
0.82
0.88
```

**Explanation**

The second assignment replaces the value referred to by the name `training_accuracy`. It does not create a second variable with the same name.

## 5. Variable naming rules

Good variable names make ML code easier to understand, debug and maintain.

### 5.1 Valid names

Examples:

- `customer_age`
- `monthly_revenue`
- `model_name`
- `learning_rate`
- `prediction_result`
- `number_of_features`

### 5.2 Invalid names

| Name | Valid? | Reason |
|---|---|---|
| `customer_age` | Yes | Letters and underscore are allowed |
| `feature2` | Yes | A digit can appear after the first character |
| `2features` | No | A name cannot start with a digit |
| `customer age` | No | Spaces are not allowed |
| `model-name` | No | A hyphen is not allowed in an identifier |
| `class` | No | `class` is a reserved Python keyword |

### 5.3 Use descriptive names

**Code cell 8 — Unclear names**

```python
x = 28
y = 4500
z = "Premium"

print(x, y, z)
```

**Expected output**

```text
28 4500 Premium
```

**Explanation**

Python accepts these names, but a reader cannot immediately tell what each value represents.

**Code cell 9 — Clear names**

```python
customer_age = 28
monthly_spend = 4500
membership_type = "Premium"

print(customer_age, monthly_spend, membership_type)
```

**Expected output**

```text
28 4500 Premium
```

**Explanation**

The output is the same, but the program is easier to understand. Prefer descriptive names in data and model code.

### 5.4 Python is case-sensitive

**Code cell 10 — Compare differently capitalized names**

```python
score = 0.91
Score = 0.84
SCORE = 0.76

print(score)
print(Score)
print(SCORE)
```

**Expected output**

```text
0.91
0.84
0.76
```

**Explanation**

Python treats `score`, `Score` and `SCORE` as three separate names. Use a consistent naming style to avoid mistakes.

### 5.5 Recommended naming style

Use `snake_case` for ordinary Python variables.

```python
training_data_path = "data/train.csv"
target_column = "customer_left"
validation_score = 0.89
```

Use uppercase names by convention for values intended to act as constants.

```python
RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
```

Python does not enforce constants. Uppercase is a communication convention for developers.

## 6. Understanding Python data types

A data type describes the kind of value an object represents. Different types support different operations.

### 6.1 Common built-in types

| Type | Example | Typical ML/data use |
|---|---|---|
| `int` | `25` | Age, count of orders, number of rows |
| `float` | `25.75` | Price, duration, measurement, score |
| `str` | `"Premium"` | Category, name, label, file path |
| `bool` | `True` | Flag, condition, binary status |
| `NoneType` | `None` | Explicitly absent value |

Python also has collection types such as list, tuple, set and dictionary. These are covered in Day 04.

### 6.2 Integer

An integer is a whole-number value.

**Code cell 11 — Integer values**

```python
customer_age = 25
number_of_orders = 14
number_of_features = 8

print(customer_age)
print(number_of_orders)
print(number_of_features)
```

**Expected output**

```text
25
14
8
```

**Explanation**

These values represent whole-number counts or quantities. They are suitable for counting rows, features or orders.

### 6.3 Float

A float represents a numeric value that may include a decimal component.

**Code cell 12 — Float values**

```python
product_price = 1499.50
model_accuracy = 0.91
average_session_time = 12.75

print(product_price)
print(model_accuracy)
print(average_session_time)
```

**Expected output**

```text
1499.5
0.91
12.75
```

**Explanation**

- Price and time can contain decimal values.
- Model scores are often represented as floating-point numbers.
- The printed representation may omit an unnecessary trailing zero.

### 6.4 String

A string represents text and is written inside single or double quotation marks.

**Code cell 13 — String values**

```python
customer_segment = "High Value"
target_column = "churn"
dataset_name = "customer_data.csv"

print(customer_segment)
print(target_column)
print(dataset_name)
```

**Expected output**

```text
High Value
churn
customer_data.csv
```

**Explanation**

A column name, category or file path is text. Even if text contains digits, it remains a string when written inside quotation marks.

### 6.5 Boolean

A Boolean represents either `True` or `False`.

**Code cell 14 — Boolean values**

```python
is_customer_active = True
has_missing_values = False

print(is_customer_active)
print(has_missing_values)
```

**Expected output**

```text
True
False
```

**Explanation**

Booleans are useful for flags and conditions, such as whether a record passed validation or whether a customer is active.

### 6.6 None

`None` is a special Python value used to represent the absence of a value.

**Code cell 15 — A value that is not available**

```python
customer_phone = None

print(customer_phone)
print(type(customer_phone))
```

**Expected output**

```text
None
<class 'NoneType'>
```

**Explanation**

`None` is not the same as zero or an empty string. It can represent an unavailable value in an application. Pandas has its own missing-value handling, which is studied in later data-preparation lessons.

### 6.7 Inspect the type of each value

The built-in `type()` function returns the type of an object.

**Code cell 16 — Check several data types**

```python
print(type(25))
print(type(25.5))
print(type("Machine Learning"))
print(type(True))
print(type(None))
```

**Expected output**

```text
<class 'int'>
<class 'float'>
<class 'str'>
<class 'bool'>
<class 'NoneType'>
```

**Explanation**

The output shows Python's type names. This is useful when a value does not behave as expected.

### 6.8 Validate a type using isinstance()

`isinstance(value, expected_type)` returns a Boolean result indicating whether the value is an instance of the specified type.

**Code cell 17 — Check whether a feature value is numeric**

```python
age_value = 32

print(isinstance(age_value, int))
print(isinstance(age_value, str))
```

**Expected output**

```text
True
False
```

**Explanation**

The first check succeeds because the value is an integer. The second check is false because the value is not a string.

**Code cell 18 — Validate a numeric measurement**

```python
temperature_value = 36.8

if isinstance(temperature_value, (int, float)):
    print("Numeric measurement")
else:
    print("Check the input type")
```

**Expected output**

```text
Numeric measurement
```

**Explanation**

- The second argument can be a tuple of accepted types.
- The check accepts either an integer or a float.
- This is a simple validation example; real data pipelines also need to handle missing values and invalid text.

## 7. Dynamic typing

Python is dynamically typed. You do not have to declare the type of a variable before assigning a value. The type is associated with the object currently referenced by the name.

**Code cell 19 — One name, different values**

```python
result = 100
print(result, type(result))

result = 0.95
print(result, type(result))

result = "Passed"
print(result, type(result))
```

**Expected output**

```text
100 <class 'int'>
0.95 <class 'float'>
Passed <class 'str'>
```

**Explanation**

The name `result` refers to an integer first, then a float, then a string. Python allows this, but mixing types without a clear reason can make ML code harder to maintain.

**Good practice:** Use a variable consistently for one logical purpose. Do not reuse a model score variable later to store a file name.

## 8. Type conversion

Type conversion changes a value from one type to another. It is common when data comes from user input, CSV files, APIs or external systems.

### 8.1 Common conversion functions

| Function | Purpose |
|---|---|
| `int(value)` | Convert a compatible value to an integer |
| `float(value)` | Convert a compatible value to a float |
| `str(value)` | Convert a value to text |
| `bool(value)` | Convert a value to its Boolean interpretation |

### 8.2 Convert text containing an integer

**Code cell 20 — Convert an age received as text**

```python
age_text = "27"

age_number = int(age_text)

print(age_number)
print(type(age_number))
```

**Expected output**

```text
27
<class 'int'>
```

**Explanation**

The original value is text. `int()` converts it to an integer because the text contains a valid whole-number representation.

### 8.3 Convert text containing a decimal number

**Code cell 21 — Convert a price**

```python
price_text = "1250.75"

price_value = float(price_text)

print(price_value)
print(type(price_value))
```

**Expected output**

```text
1250.75
<class 'float'>
```

**Explanation**

Use `float()` when the text represents a decimal number. Do not use `int()` if the decimal portion is meaningful.

### 8.4 Convert a number into text

**Code cell 22 — Prepare a value for a message**

```python
prediction_count = 125

message = "Predictions generated: " + str(prediction_count)

print(message)
```

**Expected output**

```text
Predictions generated: 125
```

**Explanation**

The `+` operator joins strings. Converting the integer with `str()` makes the values compatible.

### 8.5 Convert a float to an integer

**Code cell 23 — Observe truncation**

```python
confidence_score = 0.91

score_as_integer = int(confidence_score)

print(score_as_integer)
```

**Expected output**

```text
0
```

**Explanation**

`int()` removes the fractional part. It does not round a decimal to the nearest whole number. Do not convert model scores to integers unless that is genuinely required.

### 8.6 A conversion that cannot succeed

**Code cell 24 — Invalid numeric text**

```python
age_text = "twenty seven"
age_number = int(age_text)
```

**Expected result**

Python raises a `ValueError` because the text is not a valid integer representation.

**Explanation**

A real dataset may contain values such as `"unknown"`, `"not available"` or `"N/A"` in a numeric column. Do not blindly convert every value. First inspect the values and decide how invalid or missing entries should be handled. Error handling and data cleaning are covered later.

### 8.7 Implicit and explicit conversion

- **Implicit conversion:** Python automatically handles compatible types during an operation.
- **Explicit conversion:** The programmer requests conversion using a function such as `int()` or `float()`.

**Code cell 25 — Compatible numeric operation**

```python
row_count = 10
average_value = 2.5

combined_value = row_count + average_value

print(combined_value)
print(type(combined_value))
```

**Expected output**

```text
12.5
<class 'float'>
```

**Explanation**

Python produces a float result for this numeric operation.

**Code cell 26 — Explicit conversion before addition**

```python
row_count_text = "10"
additional_rows = 5

combined_rows = int(row_count_text) + additional_rows

print(combined_rows)
```

**Expected output**

```text
15
```

**Explanation**

The text is explicitly converted before arithmetic. Without conversion, adding a string and an integer would raise a `TypeError`.

## 9. Arithmetic operators

Arithmetic operators perform calculations. They are used in data preparation, feature creation, summaries and business calculations.

| Operator | Meaning | Example |
|---|---|---|
| `+` | Addition | `10 + 5` |
| `-` | Subtraction | `10 - 5` |
| `*` | Multiplication | `10 * 5` |
| `/` | Division | `10 / 5` |
| `//` | Floor division | `11 // 5` |
| `%` | Remainder | `11 % 5` |
| `**` | Exponentiation | `2 ** 3` |

### 9.1 Addition

**Code cell 27 — Add two feature values**

```python
online_orders = 120
store_orders = 80

total_orders = online_orders + store_orders

print(total_orders)
```

**Expected output**

```text
200
```

**Explanation:** The `+` operator adds the two integer values.

### 9.2 Subtraction

**Code cell 28 — Find remaining records**

```python
total_records = 1000
processed_records = 850

remaining_records = total_records - processed_records

print(remaining_records)
```

**Expected output**

```text
150
```

**Explanation:** The `-` operator subtracts the processed count from the total.

### 9.3 Multiplication

**Code cell 29 — Calculate total revenue**

```python
unit_price = 250.0
units_sold = 12

revenue = unit_price * units_sold

print(revenue)
```

**Expected output**

```text
3000.0
```

**Explanation:** The `*` operator multiplies the price by the number of units.

### 9.4 Division

**Code cell 30 — Calculate an average**

```python
total_sales = 9000
number_of_days = 30

average_daily_sales = total_sales / number_of_days

print(average_daily_sales)
```

**Expected output**

```text
300.0
```

**Explanation:** The `/` operator returns a division result as a float in Python 3.

### 9.5 Floor division

**Code cell 31 — Divide records into complete batches**

```python
total_records = 23
batch_size = 5

complete_batches = total_records // batch_size

print(complete_batches)
```

**Expected output**

```text
4
```

**Explanation:** Floor division returns the whole-number quotient rounded down. Four complete batches of five records can be formed.

### 9.6 Modulus

**Code cell 32 — Find leftover records**

```python
total_records = 23
batch_size = 5

leftover_records = total_records % batch_size

print(leftover_records)
```

**Expected output**

```text
3
```

**Explanation:** The modulus operator returns the remainder. Three records remain after creating four complete batches.

### 9.7 Exponentiation

**Code cell 33 — Raise a value to a power**

```python
value = 3
result = value ** 2

print(result)
```

**Expected output**

```text
9
```

**Explanation:** The `**` operator performs exponentiation. This is a Python operator example, not a machine-learning formula.

### 9.8 Division by zero

**Code cell 34 — Observe an invalid division**

```python
total_records = 100
number_of_groups = 0

average_per_group = total_records / number_of_groups
```

**Expected result**

Python raises `ZeroDivisionError`.

**Explanation**

A program should validate a denominator before dividing. A dataset may contain a zero count, so do not assume every value is safe for arithmetic.

**Code cell 35 — Validate before division**

```python
total_records = 100
number_of_groups = 0

if number_of_groups != 0:
    average_per_group = total_records / number_of_groups
    print(average_per_group)
else:
    print("Cannot divide by zero groups")
```

**Expected output**

```text
Cannot divide by zero groups
```

**Explanation**

The condition checks the denominator first. This connects today's arithmetic operators to the conditional statements taught on Day 03.

## 10. Comparison operators

Comparison operators compare values and return `True` or `False`.

| Operator | Meaning |
|---|---|
| `==` | Equal to |
| `!=` | Not equal to |
| `>` | Greater than |
| `<` | Less than |
| `>=` | Greater than or equal to |
| `<=` | Less than or equal to |

### 10.1 Equal to

**Code cell 36**

```python
actual_label = "churn"
expected_label = "churn"

print(actual_label == expected_label)
```

**Expected output**

```text
True
```

**Explanation:** `==` compares values. It does not assign a value.

### 10.2 Not equal to

**Code cell 37**

```python
training_version = "v1"
production_version = "v2"

print(training_version != production_version)
```

**Expected output**

```text
True
```

**Explanation:** The two strings are different, so the comparison returns `True`.

### 10.3 Greater than

**Code cell 38**

```python
validation_score = 0.91

print(validation_score > 0.90)
```

**Expected output**

```text
True
```

**Explanation:** The score is greater than the comparison value. This only demonstrates a comparison; it does not establish whether a model is good enough for production.

### 10.4 Less than

**Code cell 39**

```python
missing_value_count = 4

print(missing_value_count < 10)
```

**Expected output**

```text
True
```

**Explanation:** The count is smaller than ten.

### 10.5 Greater than or equal to

**Code cell 40**

```python
minimum_rows_required = 100
available_rows = 100

print(available_rows >= minimum_rows_required)
```

**Expected output**

```text
True
```

**Explanation:** The values are equal, so the greater-than-or-equal comparison is true.

### 10.6 Less than or equal to

**Code cell 41**

```python
maximum_allowed_errors = 5
current_errors = 3

print(current_errors <= maximum_allowed_errors)
```

**Expected output**

```text
True
```

**Explanation:** Three errors are not more than the allowed count of five.

### 10.7 Store a comparison result

**Code cell 42**

```python
validation_score = 0.87
minimum_score = 0.80

meets_minimum = validation_score >= minimum_score

print(meets_minimum)
print(type(meets_minimum))
```

**Expected output**

```text
True
<class 'bool'>
```

**Explanation:** A comparison result is a Boolean value. Boolean values are frequently used in validation and filtering.

## 11. Logical operators

Logical operators combine or reverse conditions.

| Operator | Meaning |
|---|---|
| `and` | Both conditions must be true |
| `or` | At least one condition must be true |
| `not` | Reverses a Boolean result |

### 11.1 Using and

**Code cell 43 — Check two data-quality requirements**

```python
has_required_columns = True
has_target_column = True

is_dataset_ready = has_required_columns and has_target_column

print(is_dataset_ready)
```

**Expected output**

```text
True
```

**Explanation:** Both conditions are true, so the combined result is true.

### 11.2 Using and when one condition is false

**Code cell 44**

```python
has_required_columns = True
has_target_column = False

is_dataset_ready = has_required_columns and has_target_column

print(is_dataset_ready)
```

**Expected output**

```text
False
```

**Explanation:** The dataset is not considered ready because the target column is missing.

### 11.3 Using or

**Code cell 45 — Check whether either input source is available**

```python
csv_available = False
database_available = True

has_input_source = csv_available or database_available

print(has_input_source)
```

**Expected output**

```text
True
```

**Explanation:** At least one of the conditions is true, so the combined result is true.

### 11.4 Using not

**Code cell 46 — Reverse a flag**

```python
is_training_complete = False

print(not is_training_complete)
```

**Expected output**

```text
True
```

**Explanation:** `not` reverses the Boolean value.

### 11.5 Combine conditions with parentheses

**Code cell 47**

```python
has_dataset = True
has_model_file = False
has_api_service = True

can_run_prediction = has_dataset and (has_model_file or has_api_service)

print(can_run_prediction)
```

**Expected output**

```text
True
```

**Explanation**

- The expression inside parentheses checks whether a model file or API service is available.
- That result is combined with `has_dataset`.
- Parentheses make the intended grouping clear.

## 12. Assignment operators

Assignment operators store or update values.

| Operator | Example | Meaning |
|---|---|---|
| `=` | `count = 10` | Assign a value |
| `+=` | `count += 2` | Add and assign |
| `-=` | `count -= 2` | Subtract and assign |
| `*=` | `count *= 2` | Multiply and assign |
| `/=` | `count /= 2` | Divide and assign |

### 12.1 Basic assignment

**Code cell 48**

```python
number_of_features = 12
print(number_of_features)
```

**Expected output**

```text
12
```

### 12.2 Add and assign

**Code cell 49**

```python
processed_rows = 100
processed_rows += 25

print(processed_rows)
```

**Expected output**

```text
125
```

**Explanation:** `+=` updates the variable using its current value plus the new amount.

### 12.3 Subtract and assign

**Code cell 50**

```python
pending_rows = 100
pending_rows -= 30

print(pending_rows)
```

**Expected output**

```text
70
```

### 12.4 Multiply and assign

**Code cell 51**

```python
batch_count = 4
batch_count *= 3

print(batch_count)
```

**Expected output**

```text
12
```

### 12.5 Divide and assign

**Code cell 52**

```python
total_duration = 60
total_duration /= 4

print(total_duration)
```

**Expected output**

```text
15.0
```

**Explanation:** The variable is updated with the division result.

## 13. Operator precedence

Operator precedence determines which operation Python performs first when an expression contains multiple operators.

Use parentheses when they make an expression easier to read.

**Code cell 53 — Multiplication before addition**

```python
result = 10 + 5 * 2
print(result)
```

**Expected output**

```text
20
```

**Explanation:** Multiplication is performed before addition.

**Code cell 54 — Parentheses change the grouping**

```python
result = (10 + 5) * 2
print(result)
```

**Expected output**

```text
30
```

**Explanation:** The expression inside parentheses is evaluated first.

**Code cell 55 — Use named intermediate values for clarity**

```python
base_records = 100
additional_records = 25
number_of_sources = 5

total_records = base_records + additional_records
records_per_source = total_records / number_of_sources

print(records_per_source)
```

**Expected output**

```text
25.0
```

**Explanation:** Named intermediate values make the sequence easier to understand than a long expression.

## 14. Input and output basics

The `input()` function reads text entered by a user. The returned value is a string, even when the user types digits.

### 14.1 Read a text value

**Code cell 56**

```python
model_name = input("Enter model name: ")
print("Selected model:", model_name)
```

**Example input**

```text
Random Forest
```

**Expected output**

```text
Enter model name: Random Forest
Selected model: Random Forest
```

**Explanation:** The entered text is stored in `model_name`.

### 14.2 Read a numeric value and inspect it

**Code cell 57**

```python
row_count_text = input("Enter row count: ")

print(row_count_text)
print(type(row_count_text))
```

**Example input**

```text
500
```

**Expected output**

```text
Enter row count: 500
500
<class 'str'>
```

**Explanation:** The input is text. It must be converted before numeric calculations.

### 14.3 Convert user input before calculation

**Code cell 58**

```python
row_count_text = input("Enter row count: ")
row_count = int(row_count_text)

print("Rows received:", row_count)
print("Type:", type(row_count))
```

**Example input**

```text
500
```

**Expected output**

```text
Enter row count: 500
Rows received: 500
Type: <class 'int'>
```

**Explanation:** `int()` converts the input string to an integer. This assumes the user enters valid integer text; robust validation will be covered with exception handling.

## 15. Machine-learning context: configuration values

ML scripts often keep settings in variables. This makes experiments easier to read and update.

**Code cell 59 — Store experiment settings**

```python
dataset_path = "data/customer_churn.csv"
target_column = "churn"
test_size = 0.20
random_state = 42

print(dataset_path)
print(target_column)
print(test_size)
print(random_state)
```

**Expected output**

```text
data/customer_churn.csv
churn
0.2
42
```

**Explanation**

- `dataset_path` stores a file location as text.
- `target_column` stores the name of the value to predict.
- `test_size` stores a decimal setting.
- `random_state` stores an integer setting.
- These values are examples of configuration, not a complete training script.

**Important:** The exact meaning of a setting depends on the library or function that receives it. Do not assume every parameter has the same meaning across tools.

### 15.1 Change one configuration value

**Code cell 60**

```python
test_size = 0.20
print("Initial test size:", test_size)

test_size = 0.25
print("Updated test size:", test_size)
```

**Expected output**

```text
Initial test size: 0.2
Updated test size: 0.25
```

**Explanation:** Reassigning a variable changes the value used by later code cells.

### 15.2 Validate a configuration value

**Code cell 61**

```python
test_size = 0.20

if isinstance(test_size, (int, float)):
    print("Test size is numeric")
else:
    print("Test size must be numeric")
```

**Expected output**

```text
Test size is numeric
```

**Explanation:** This checks the type only. A production-quality configuration check should also verify that the value is within the range expected by the function.

### 15.3 Compare a configuration value with an expected range

**Code cell 62**

```python
test_size = 0.20

if 0 < test_size < 1:
    print("Test size is between zero and one")
else:
    print("Review the test size")
```

**Expected output**

```text
Test size is between zero and one
```

**Explanation:** Python supports chained comparisons. This is a basic range check; the exact allowed range should follow the function's documentation.

## 16. Machine-learning context: represent one observation

A row in a dataset represents one observation. Before using pandas, we can represent one example using separate Python variables.

**Code cell 63 — Store one customer observation**

```python
customer_id = 101
age = 34
monthly_charges = 799.50
contract_type = "Monthly"
has_support_plan = True
churned = False

print(customer_id, age, monthly_charges)
print(contract_type, has_support_plan, churned)
```

**Expected output**

```text
101 34 799.5
Monthly True False
```

**Explanation:** Each variable represents one field from a customer record. A later lesson will show how to organize many records into a DataFrame.

### 16.1 Inspect the observation's data types

**Code cell 64**

```python
print(type(customer_id))
print(type(age))
print(type(monthly_charges))
print(type(contract_type))
print(type(has_support_plan))
print(type(churned))
```

**Expected output**

```text
<class 'int'>
<class 'int'>
<class 'float'>
<class 'str'>
<class 'bool'>
<class 'bool'>
```

**Explanation:** Checking types helps identify whether a value is represented in a form suitable for the next processing step.

### 16.2 A value can look numeric but still be text

**Code cell 65**

```python
monthly_charges = "799.50"

print(monthly_charges)
print(type(monthly_charges))
```

**Expected output**

```text
799.50
<class 'str'>
```

**Explanation:** Quotation marks make the value a string. It cannot be used directly in numeric arithmetic.

### 16.3 Convert and inspect the value

**Code cell 66**

```python
monthly_charges = "799.50"

monthly_charges = float(monthly_charges)

print(monthly_charges)
print(type(monthly_charges))
```

**Expected output**

```text
799.5
<class 'float'>
```

**Explanation:** The text has been converted to a float. In a real dataset, inspect invalid entries before applying conversion to an entire column.

## 17. Common beginner mistakes and how to fix them

### 17.1 Using a variable before creating it

**Incorrect code**

```python
print(model_accuracy)
```

**Expected result:** `NameError` if the name has not been defined.

**Correct code**

```python
model_accuracy = 0.91
print(model_accuracy)
```

**Output**

```text
0.91
```

### 17.2 Confusing assignment with comparison

**Incorrect understanding**

```python
score = 0.90
```

This assigns a value; it does not ask whether the score equals 0.90.

**Comparison**

```python
score = 0.90
print(score == 0.90)
```

**Output**

```text
True
```

### 17.3 Adding a string to an integer

**Incorrect code**

```python
rows_text = "100"
additional_rows = 20

print(rows_text + additional_rows)
```

**Expected result:** `TypeError`.

**Correct code**

```python
rows_text = "100"
additional_rows = 20

print(int(rows_text) + additional_rows)
```

**Output**

```text
120
```

### 17.4 Using a lowercase Boolean

**Incorrect code**

```python
is_valid = true
```

**Expected result:** `NameError` because `true` is not Python's Boolean literal.

**Correct code**

```python
is_valid = True
print(is_valid)
```

**Output**

```text
True
```

### 17.5 Using an invalid variable name

**Incorrect code**

```python
2_features = 10
```

**Expected result:** `SyntaxError`.

**Correct code**

```python
number_of_features = 10
print(number_of_features)
```

**Output**

```text
10
```

### 17.6 Dividing by zero

**Incorrect code**

```python
total_rows = 100
batch_size = 0

print(total_rows / batch_size)
```

**Expected result:** `ZeroDivisionError`.

**Correct code**

```python
total_rows = 100
batch_size = 0

if batch_size > 0:
    print(total_rows / batch_size)
else:
    print("Batch size must be greater than zero")
```

**Output**

```text
Batch size must be greater than zero
```

### 17.7 Assuming input() returns a number

**Incorrect code**

```python
age = input("Enter age: ")
print(age + 1)
```

**Expected result:** `TypeError` because the input is a string.

**Correct code**

```python
age_text = input("Enter age: ")
age = int(age_text)

print(age + 1)
```

**Example input**

```text
25
```

**Output**

```text
Enter age: 25
26
```

### 17.8 Comparing values with different meanings

A value of `0`, an empty string and `None` are not interchangeable.

```python
number_of_purchases = 0
customer_note = ""
missing_phone = None

print(type(number_of_purchases))
print(type(customer_note))
print(type(missing_phone))
```

**Output**

```text
<class 'int'>
<class 'str'>
<class 'NoneType'>
```

**Explanation:** These values communicate different things. Preserve their meaning during data cleaning.

## 18. Short practical examples for classroom discussion

### Example 1 — Dataset size

**Code cell 67**

```python
number_of_rows = 5000
number_of_columns = 12

print("Rows:", number_of_rows)
print("Columns:", number_of_columns)
```

**Output**

```text
Rows: 5000
Columns: 12
```

**Explanation:** Counts are stored as integers.

### Example 2 — Feature measurement

**Code cell 68**

```python
average_purchase = 1250.75
print("Average purchase:", average_purchase)
```

**Output**

```text
Average purchase: 1250.75
```

**Explanation:** A measurement with a decimal component can be represented as a float.

### Example 3 — Target column name

**Code cell 69**

```python
target_column = "churn"
print("Target column:", target_column)
```

**Output**

```text
Target column: churn
```

**Explanation:** A column name is text, so it is represented as a string.

### Example 4 — Data validation flag

**Code cell 70**

```python
all_required_fields_present = True
print(all_required_fields_present)
```

**Output**

```text
True
```

**Explanation:** A validation flag can be represented as a Boolean.

### Example 5 — Optional value

**Code cell 71**

```python
optional_description = None
print(optional_description is None)
```

**Output**

```text
True
```

**Explanation:** Use an explicit `None` check when testing for the absence of a value.

## 19. Quick revision

| Concept | Remember |
|---|---|
| Variable | A name referring to a value/object |
| Assignment | `=` associates a name with a value |
| Integer | Whole-number value |
| Float | Numeric value with a decimal component |
| String | Text |
| Boolean | `True` or `False` |
| None | Represents absence of a value |
| `type()` | Inspects an object's type |
| `isinstance()` | Checks whether a value belongs to a type |
| Type conversion | Changes a value to a compatible type |
| Arithmetic operator | Performs a calculation |
| Comparison operator | Produces a Boolean comparison result |
| Logical operator | Combines or reverses conditions |
| Assignment operator | Stores or updates a value |
| `input()` | Reads user input as text |

## 20. Interview-ready understanding

**Question: Why should an ML engineer care about Python data types?**

A machine-learning workflow receives values from files, APIs and databases. A numeric feature stored as text may fail during calculation or model preprocessing. An ML engineer should inspect the type, understand what the value represents, convert only valid values and handle missing or invalid data deliberately.

**Question: What is the difference between assignment and comparison?**

Assignment uses `=` to associate a variable name with a value. Comparison uses `==` to check whether two values are equal and returns a Boolean result.

**Question: Why does input() often need conversion?**

The `input()` function returns text. If the program needs to perform arithmetic, the text must be converted to a suitable numeric type after checking that the input is valid.

**Question: What should you do if a column expected to be numeric contains text?**

Inspect the distinct or unexpected values, identify whether they represent missing values, formatting differences or invalid entries, then choose an appropriate cleaning and conversion strategy. Do not convert blindly and silently discard information.

## 21. Lesson completion checklist

- [ ] I can open a Python notebook and run a cell.
- [ ] I understand that later cells can reuse earlier variables.
- [ ] I can create readable variable names.
- [ ] I can distinguish int, float, str, bool and None.
- [ ] I can inspect a value with `type()`.
- [ ] I can validate a type with `isinstance()`.
- [ ] I can convert compatible values using `int()`, `float()` and `str()`.
- [ ] I can use arithmetic and comparison operators.
- [ ] I can combine conditions with logical operators.
- [ ] I can explain why type validation matters before ML preprocessing.

**Next lesson:** Day 03 — Conditions, Loops and Functions. We will use conditions to make decisions, loops to process repeated records and functions to organize reusable Python logic.
