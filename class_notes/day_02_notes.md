# Day 02: Python Environment, Variables, Data Types and Operators

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate  
**Learning focus:** Understand the concepts, apply them correctly, and explain decisions in an interview.

---

## 1. Learning Objectives

- Understand how Python code is executed in a development environment.
- Create and use variables with meaningful names.
- Identify Python's main built-in data types.
- Convert between compatible types safely.
- Use arithmetic, comparison, logical and assignment operators.
- Understand why correct data types matter in data and ML workflows.

## 2. What Is Python?

Python is a general-purpose programming language used in automation, data analysis, machine learning, web development and scientific computing.

ML engineers commonly use Python because its ecosystem includes NumPy, Pandas, Matplotlib, Seaborn and scikit-learn, along with deep-learning libraries.

## 3. Setting Up a Python Environment

A **Python environment** is the setup in which Python code runs, including the Python interpreter and installed packages.

| Tool | Purpose |
|---|---|
| Python interpreter | Executes Python code |
| VS Code | Code editor with extensions and terminal |
| Jupyter Notebook | Runs code in separate cells and displays outputs |
| Google Colab | Hosted notebook environment accessible through a browser |
| pip | Installs Python packages |
| Virtual environment | Keeps dependencies isolated for a project |

### Check Python installation

Run in a terminal:

```bash
python --version
```

If your system uses the Python launcher on Windows, you can also try:

```bash
py --version
```

### Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Explanation:**

- `python -m venv .venv` creates an isolated environment in `.venv`.
- The activation command makes that environment's Python and pip active in the terminal.
- Upgrading pip updates the package installer inside the active environment.

If PowerShell blocks activation due to execution policy, follow your organization's approved policy; do not change machine-wide security settings blindly. You can also invoke the environment's interpreter directly.

### Install common data science libraries

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

**Important:** A package installed in one Python environment may not be available in another. In a notebook, verify which Python kernel is selected.

## 4. What Is a Variable?

A variable is a name that refers to a value stored by a Python program.

```python
customer_age = 28
monthly_bill = 799.50
customer_name = "Anita"
is_active = True

print(customer_age)
print(monthly_bill)
print(customer_name)
print(is_active)
```

**Output:**

```text
28
799.5
Anita
True
```

Explanation:

- `customer_age` refers to an integer value.
- `monthly_bill` refers to a decimal value.
- `customer_name` refers to text.
- `is_active` refers to a Boolean value.
- `print()` displays values so we can inspect them.

### Variable naming rules

- A name can contain letters, digits and underscores.
- A name cannot begin with a digit.
- Names are case-sensitive: `age` and `Age` are different.
- Python keywords such as `if`, `for` and `class` cannot be used as variable names.
- Use descriptive names such as `monthly_revenue`, not vague names such as `x`, when practical.

## 5. Python Data Types

A data type describes the kind of value a variable refers to.

| Type | Example | Use in ML/data work |
|---|---|---|
| `int` | `25` | Number of orders, age in whole years |
| `float` | `25.75` | Price, probability, measurement |
| `str` | `"Hyderabad"` | City, product category |
| `bool` | `True` | Is customer active? |
| `NoneType` | `None` | Missing/not-yet-assigned value in Python |

### Inspect a value's type

```python
age = 25
price = 499.99
city = "Hyderabad"
active = True
result = None

print(type(age))
print(type(price))
print(type(city))
print(type(active))
print(type(result))
```

**Output:**

```text
<class 'int'>
<class 'float'>
<class 'str'>
<class 'bool'>
<class 'NoneType'>
```

### Why data types matter

Suppose a CSV file stores a price as text (`"1200"`) rather than a number. A calculation may fail or behave differently from what you expect. Data must be checked and converted appropriately before mathematical operations or model training.

## 6. Type Conversion

Type conversion changes a value from one type to another when the conversion is valid.

```python
age_text = "25"
age_number = int(age_text)

price_text = "1499.50"
price_number = float(price_text)

quantity = 3
quantity_text = str(quantity)

print(age_number + 5)
print(price_number * 2)
print(quantity_text)
```

**Output:**

```text
30
2999.0
3
```

### Invalid conversion

```python
value = "twenty"
number = int(value)
```

This raises `ValueError` because the text `"twenty"` is not a valid integer representation.

**Safer input handling:**

```python
raw_value = "27"

try:
    age = int(raw_value)
    print("Age:", age)
except ValueError:
    print("Please provide age as a whole number.")
```

In production, validate input at the boundary and report useful errors rather than silently replacing invalid values.

## 7. Operators in Python

### 7.1 Arithmetic operators

| Operator | Meaning | Example | Result |
|---|---|---|---:|
| `+` | Addition | `10 + 3` | 13 |
| `-` | Subtraction | `10 - 3` | 7 |
| `*` | Multiplication | `10 * 3` | 30 |
| `/` | Division | `10 / 3` | 3.333... |
| `//` | Floor division | `10 // 3` | 3 |
| `%` | Remainder | `10 % 3` | 1 |
| `**` | Power | `2 ** 3` | 8 |

**Business example: calculate revenue**

```python
units_sold = 12
unit_price = 250
revenue = units_sold * unit_price

print("Revenue:", revenue)
```

Output: `Revenue: 3000`

### 7.2 Comparison operators

Comparison operators return `True` or `False`.

| Operator | Meaning | Example |
|---|---|---|
| `==` | Equal to | `score == 80` |
| `!=` | Not equal to | `status != "active"` |
| `>` | Greater than | `age > 18` |
| `<` | Less than | `stock < 10` |
| `>=` | Greater than or equal | `marks >= 40` |
| `<=` | Less than or equal | `bill <= 1000` |

```python
model_score = 0.84
minimum_score = 0.80

print(model_score >= minimum_score)
```

Output: `True`

The comparison says the score is at least the configured threshold. It does not prove the prediction is correct.

### 7.3 Logical operators

- `and`: both conditions must be true.
- `or`: at least one condition must be true.
- `not`: reverses a Boolean condition.

```python
age = 24
has_account = True

eligible_for_offer = age >= 18 and has_account
print(eligible_for_offer)
```

Output: `True`

### 7.4 Assignment operators

```python
total = 100
total += 25   # same as total = total + 25
total -= 10   # same as total = total - 10
print(total)
```

Output: `115`

## 8. Operator Precedence

Python evaluates some operators before others. Parentheses make the intended order explicit.

```python
result_1 = 10 + 2 * 3
result_2 = (10 + 2) * 3

print(result_1)
print(result_2)
```

Output:

```text
16
36
```

Multiplication happens before addition in the first expression. Parentheses change the order in the second.

## 9. Industry Scenario — Monthly Customer Value

An e-commerce company wants to calculate a simple order value and check whether an order qualifies for free delivery.

```python
item_price = 850.0
quantity = 2
delivery_fee = 60.0
free_delivery_limit = 1500.0

subtotal = item_price * quantity

if subtotal >= free_delivery_limit:
    final_amount = subtotal
else:
    final_amount = subtotal + delivery_fee

print("Subtotal:", subtotal)
print("Final amount:", final_amount)
```

Output:

```text
Subtotal: 1700.0
Final amount: 1700.0
```

The example shows how variables, numeric types, multiplication and comparisons combine in a business rule. This is not itself a trained ML model; it is a deterministic rule.

## 10. Common Errors and Best Practices

| Error / weak practice | Why it is a problem | Better approach |
|---|---|---|
| `"10" + 5` | String and integer cannot be added directly | Convert after validating |
| Naming a variable `list` | Hides Python's built-in `list` name | Use `customer_records` |
| Using `=` instead of `==` in a comparison | `=` assigns a value | Use `==` for equality checks |
| Assuming CSV columns have correct types | Files may contain mixed or invalid values | Inspect and validate types |
| Hard-coding unexplained constants | Business rules become difficult to maintain | Use clearly named variables/configuration |
| Converting every value without checking | Invalid data may raise errors or be corrupted | Validate input and handle conversion errors |

## 11. Quick Revision

- A Python environment contains an interpreter and installed dependencies.
- Variables are names referring to values.
- Common types include integer, float, string, Boolean and None.
- `type()` inspects a value's type; `int()`, `float()` and `str()` perform conversions.
- Arithmetic operators calculate; comparison operators return Booleans; logical operators combine conditions.
- Good data types and input validation are essential before ML computations.

---

## Expanded Beginner-to-Developer Teaching Layer

# Day 02: Python environment, variables, data types and operators

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

**Python environment, variables, data types and operators**

Python is a general-purpose programming language. A variable is a name referring to a value. A data type describes what kind of value is stored and what operations are valid.

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

Use variables to give meaningful names to values; select numeric, string, Boolean or collection types according to the information being represented.

**Without this concept or step:** teams may produce incorrect outputs, waste compute, make poor business decisions or trust a result that does not work on new data. The exact risk depends on the application; investigate rather than assume.

## 4. Syntax, Structure and How to Read It

Do not confuse = (assignment) with == (comparison), or convert invalid text to numbers without handling errors.

### Read the syntax carefully

1. Identify the library or Python feature being used.
2. Identify the input variable and its expected type/shape.
3. Identify the operation or method being called.
4. Check optional parameters and their defaults.
5. Inspect the output and verify that it matches the intended result.

### Practical example

```python
# A small, readable example related to the lesson
monthly_fee = 1200.0\nmonths = 6\ntotal = monthly_fee * months\nprint(total)
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

**Scenario:** A churn analysis reads monthly charges as numbers. If charges are stored as text such as '1200', calculations and comparisons may fail or behave incorrectly.

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

Use type(), int(), float(), str(), bool(); arithmetic + - * / // % **; comparisons == != > < >= <=; logical and, or, not. Example: monthly_fee = 1200.0; months = 6; total = monthly_fee * months; print(total) gives 7200.0.

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
| What is it? | Python is a general-purpose programming language. A variable is a name referring to a value. A data type describes what kind of value is stored and what operations are valid. |
| Why use it? | Use variables to give meaningful names to values; select numeric, string, Boolean or collection types according to the information being represented. |
| Main practical use | A churn analysis reads monthly charges as numbers. If charges are stored as text such as '1200', calculations and comparisons may fail or behave incorrectly. |
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

Use the matching `day_02_interview.md` file for detailed interview questions, model answers, real-world troubleshooting scenarios and follow-up questions.

---

**Learning standard:** Understand the meaning first, then syntax, then practical use, then engineering trade-offs. A student should be able to explain the idea without reading code from the screen.


---

## Beginner-to-Developer Expansion: More Examples and Real-World Reasoning

### A. Start with the simplest meaning
Before memorizing a function, explain the idea in your own words. Identify the input, output and the reason the operation exists. A correct syntax call on the wrong data is still a wrong solution.

### B. Additional practical scenario
A data pipeline reads '1200' as text. Adding a numeric fee to that value can fail or cause concatenation in some contexts. Convert deliberately and validate the result before calculation.

### C. Syntax and use: how to choose
Use int for whole-number counts, float for decimal measurements, str for labels and bool for true/false flags. Check type(value) and handle conversion failures with try/except where inputs are untrusted.

### D. What happens if this is skipped?
If types are ignored, sorting can become alphabetical, arithmetic can fail and model features may be interpreted incorrectly.

### E. Senior developer review checklist
- Is the business definition clear?
- Are input types, units, shape and missing values checked?
- Can the result be verified on a small hand-checkable example?
- Have normal, boundary and invalid cases been considered?
- Is the approach simple enough to explain and maintain?
- Are limitations and possible failure costs documented?

### F. Explain it in an interview
Use this order: definition → why it is needed → syntax or method → small example → real business scenario → what can go wrong → how you validate the result. Avoid answering with only a library function name.
