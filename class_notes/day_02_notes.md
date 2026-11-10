# Day 02: Python Environment, Variables, Data Types and Operators

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate

## Learning Objectives
- Understand what Python is and where Python code runs.
- Create variables using meaningful names.
- Identify common Python data types.
- Convert values between compatible types.
- Use arithmetic, comparison, logical and assignment operators.
- Recognize type errors that affect data processing.

## 1. What Is Python?
Python is a high-level, general-purpose programming language. It is widely used in automation, data analysis, machine learning and application development because its syntax is readable and its library ecosystem is large.

### Where do we write Python?
- VS Code: code editor with Python extension and terminal.
- Jupyter Notebook: interactive environment with cells for code and output.
- Google Colab: hosted notebook environment that runs in a browser.

**Basic syntax**
```python
print("Hello, Machine Learning")
```
Output: `Hello, Machine Learning`

## 2. Variables
**Definition:** A variable is a name that refers to a value in a program. It lets us store and reuse information.

**Syntax:** `variable_name = value`

**Examples**
```python
customer_name = "Asha"
age = 25
monthly_bill = 1250.50
is_active = True

print(customer_name)
print(age)
print(monthly_bill)
print(is_active)
```
Output: Asha, 25, 1250.5, True (each on a separate line).

```python
sales = 50000
sales = 65000
print(sales)
```
Output: `65000`. The variable now refers to the new value.

```python
price = 250
quantity = 4
total = price * quantity
print(total)
```
Output: `1000`. Variables can be used in calculations.

**Naming rules:** Names can contain letters, digits and underscores; cannot start with a digit; are case-sensitive; and should not be Python keywords. Prefer names such as `monthly_revenue` over unclear names such as `x`.

## 3. Data Types
A **data type** identifies the kind of value a variable refers to and influences which operations are valid.

| Type | Meaning | Example |
|---|---|---|
| int | Whole number | 25 |
| float | Decimal number | 25.75 |
| str | Text | "Python" |
| bool | True/False | True |
| NoneType | No value / missing reference | None |
| list | Ordered mutable collection | [10, 20] |
| tuple | Ordered immutable collection | (10, 20) |
| dict | Key-value mapping | {"age": 25} |
| set | Collection of unique values | {10, 20} |

**Examples**
```python
age = 25
price = 99.5
name = "Ravi"
is_trained = False
print(type(age), type(price), type(name), type(is_trained))
```

```python
customer = {"name": "Asha", "age": 28}
print(customer["name"])
print(type(customer))
```
Output: Asha and dict.

```python
missing_value = None
print(type(missing_value))
print(missing_value is None)
```
Output: <class 'NoneType'> and True.

## 4. Type Conversion
**Definition:** Type conversion changes a value from one data type to another, when the conversion is valid.

**Syntax:** `int(value)`, `float(value)`, `str(value)`, `bool(value)`.

**Examples**
```python
price_text = "250"
price = int(price_text)
print(price + 50)
```
Output: 300.

```python
quantity = 4
quantity_text = str(quantity)
print("Quantity: " + quantity_text)
```
Output: Quantity: 4.

```python
rating_text = "4.5"
rating = float(rating_text)
print(rating * 2)
```
Output: 9.0.

Invalid conversion such as `int("hello")` raises ValueError. Validate incoming data and handle expected conversion errors.

## 5. Operators
An **operator** is a symbol or keyword that performs an operation on values.

### 5.1 Arithmetic operators
| Operator | Meaning | Example result |
|---|---|---|
| + | Addition | 10 + 3 = 13 |
| - | Subtraction | 10 - 3 = 7 |
| * | Multiplication | 10 * 3 = 30 |
| / | Division | 10 / 3 = 3.333... |
| // | Floor division | 10 // 3 = 3 |
| % | Remainder | 10 % 3 = 1 |
| ** | Power | 2 ** 3 = 8 |

Examples:
```python
price = 200
quantity = 3
print(price * quantity)
```
Output: 600.

```python
total = 1000
customers = 4
print(total / customers)
```
Output: 250.0.

```python
print(17 // 5)
print(17 % 5)
print(2 ** 4)
```
Output: 3, 2, 16.

### 5.2 Comparison operators
Comparison operators compare values and return True or False: `==`, `!=`, `>`, `<`, `>=`, `<=`.

Examples:
```python
age = 20
print(age >= 18)
print(age == 20)
```
Output: True, True.

```python
stock = 0
print(stock > 0)
print(stock != 0)
```
Output: False, False.

```python
score = 75
print(60 <= score <= 100)
```
Output: True.

### 5.3 Logical operators
- `and`: both conditions must be True.
- `or`: at least one condition must be True.
- `not`: reverses a Boolean result.

Examples:
```python
age = 25
salary = 40000
print(age >= 18 and salary >= 30000)
```
Output: True.

```python
is_premium = False
purchase = 6000
print(is_premium or purchase > 5000)
```
Output: True.

```python
is_blocked = False
print(not is_blocked)
```
Output: True.

### 5.4 Assignment operators
Assignment operators store or update values: `=`, `+=`, `-=`, `*=`, `/=`.

Examples:
```python
count = 5
count += 2
print(count)
```
Output: 7.

```python
balance = 1000
balance -= 250
print(balance)
```
Output: 750.

```python
price = 100
price *= 1.10
print(price)
```
Output: 110.00000000000001 (floating-point representation).

## 6. Why Data Types Matter in ML
A column containing numeric-looking text is still text until it is converted. This can break arithmetic, sorting, statistics and model input preparation.

**Scenario questions**
1. A CSV stores age as "25" rather than 25. What should you check before calculating average age? Check the type and convert valid values safely.
2. A price column contains "unknown". Should you directly call float() on every value? No. Identify invalid values and decide how to handle them.
3. Why is a Boolean field such as `is_active` useful? It represents a yes/no state clearly.

## 7. Common Mistakes
- Using a variable before assigning a value.
- Confusing `=` with `==`.
- Mixing strings and numbers in arithmetic.
- Assuming a conversion will always succeed.
- Using unclear variable names.
- Treating None as numeric zero without a business rule.
