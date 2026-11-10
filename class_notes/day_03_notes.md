# Day 03: Conditional Statements, Loops and Functions

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, Mathematics and Statistics Foundations  
**Level:** Beginner to Intermediate

## Learning Objectives

- Explain what conditions, loops and functions mean in Python.
- Write decision-making logic using `if`, `elif` and `else`.
- Combine conditions using `and`, `or` and `not`.
- Repeat operations using `for` and `while` loops.
- Control loop execution using `break` and `continue`.
- Define reusable functions with parameters and return values.
- Apply these fundamentals to business and beginner ML data workflows.

---

## 1. Introduction: Why Do We Need These Concepts?

A Python program normally executes statements from top to bottom. Real applications need more than that:

- They must make decisions.
- They must repeat the same operation for many records.
- They must organize repeated logic so it can be reused.

| Python concept | What it helps us do | Simple real-world example |
|---|---|---|
| Condition | Make a decision | Check whether a customer qualifies for a discount |
| Loop | Repeat an operation | Calculate a result for every transaction |
| Function | Reuse a task | Calculate the total price whenever needed |

If we do not use these concepts, developers may write repeated code, manually handle each record and struggle to maintain business rules.

# 2. Conditional Statements — Making Decisions

## 2.1 What Is a Condition?

A **condition** is an expression that Python evaluates to either `True` or `False`.

A condition helps a program decide which statements should execute.

**Real-life example:**

- If it rains, take an umbrella.
- If a customer's order is above ₹1,000, offer free delivery.
- If a model's recall is below the required target, review its predictions.

Python uses comparison operators to create conditions.

## 2.2 What Is an `if` Statement?

The **`if` statement** executes a block of code only when its condition is `True`.

### Syntax

```python
if condition:
    statement
```

- `if` is the Python keyword that starts the decision.
- `condition` is an expression that evaluates to True or False.
- The colon `:` marks the beginning of the block.
- The indented statement runs only when the condition is True.

### Example 1 — Temperature

```python
temperature = 35

if temperature > 30:
    print("It is a hot day")
```

**Output**

```text
It is a hot day
```

**Explanation:** Python checks whether 35 is greater than 30. The result is True, so the indented print statement executes.

### Example 2 — Sales target

```python
monthly_sales = 80000

if monthly_sales >= 50000:
    print("Sales target achieved")
```

**Output**

```text
Sales target achieved
```

**Explanation:** The business target is ₹50,000. Sales are ₹80,000, so the condition is True.

### Example 3 — ML model accuracy

```python
model_accuracy = 0.92

if model_accuracy >= 0.90:
    print("Accuracy meets the minimum target")
```

**Output**

```text
Accuracy meets the minimum target
```

**Explanation:** Accuracy is 0.92 (92%). It is greater than or equal to the 0.90 target.

**Important:** This condition only checks accuracy. A model can have high accuracy and still perform poorly on an important minority class. We must also inspect relevant metrics such as precision and recall.

### What happens when the condition is False?

Python skips the indented block and continues with the next statement outside that block.

```python
sales = 30000

if sales >= 50000:
    print("Target achieved")

print("Sales report completed")
```

**Output**

```text
Sales report completed
```

The first print is skipped because the condition is False. The final print is outside the if block, so it executes.

## 2.3 What Is an `if-else` Statement?

An **`if-else` statement** selects between two alternatives:

- If the condition is True, Python executes the `if` block.
- If the condition is False, Python executes the `else` block.

The `else` keyword does not have its own condition. It is the fallback branch.

### Syntax

```python
if condition:
    statement_if_true
else:
    statement_if_false
```

### Example 1 — Stock availability

```python
stock = 4

if stock > 0:
    print("Product is available")
else:
    print("Product is out of stock")
```

**Output**

```text
Product is available
```

### Example 2 — Free delivery

```python
order_amount = 800

if order_amount >= 1000:
    print("Free delivery available")
else:
    print("Delivery charges applicable")
```

**Output**

```text
Delivery charges applicable
```

### Example 3 — Model evaluation

```python
recall = 0.72

if recall >= 0.80:
    print("Recall target achieved")
else:
    print("Recall needs improvement")
```

**Output**

```text
Recall needs improvement
```

**Explanation:** Recall is 0.72, which is below the required 0.80. Therefore, Python executes the else block.

## 2.4 What Is an `if-elif-else` Statement?

**`elif` means “else if”.** It allows us to check multiple conditions when there are more than two possible outcomes.

Python checks conditions from top to bottom. The first condition that evaluates to True executes its block. The remaining branches are skipped. If none is True, the `else` block runs.

### Syntax

```python
if condition_1:
    statement_1
elif condition_2:
    statement_2
elif condition_3:
    statement_3
else:
    default_statement
```

You can use one or more `elif` branches. The final `else` is optional.

### Example 1 — Student grades

```python
marks = 72

if marks >= 90:
    print("Grade A")
elif marks >= 75:
    print("Grade B")
elif marks >= 60:
    print("Grade C")
elif marks >= 40:
    print("Grade D")
else:
    print("Fail")
```

**Output**

```text
Grade C
```

**Explanation:** 72 is not greater than or equal to 90, 75 or 60? It is greater than or equal to 60, so Grade C is printed. (The first two conditions are False.)

### Example 2 — Sales performance

```python
sales = 85000

if sales >= 100000:
    print("Excellent")
elif sales >= 75000:
    print("Good")
elif sales >= 50000:
    print("Average")
else:
    print("Poor")
```

**Output**

```text
Good
```

### Example 3 — Customer support priority

```python
waiting_minutes = 75

if waiting_minutes >= 60:
    print("High priority")
elif waiting_minutes >= 30:
    print("Medium priority")
else:
    print("Normal priority")
```

**Output**

```text
High priority
```

### Why does condition order matter?

A condition that matches too many values must not be placed before a more specific condition.

Incorrect order:

```python
sales = 90000

if sales >= 50000:
    print("Average")
elif sales >= 75000:
    print("Good")
else:
    print("Poor")
```

Output: `Average`

The first condition is already True, so Python never checks the second condition.

Correct order:

```python
sales = 90000

if sales >= 75000:
    print("Good")
elif sales >= 50000:
    print("Average")
else:
    print("Poor")
```

Output: `Good`

## 2.5 Logical Operators — Combining Conditions

Logical operators combine or reverse Boolean conditions.

| Operator | Meaning | Result |
|---|---|---|
| `and` | Both conditions must be True | True only when both are True |
| `or` | At least one condition must be True | True when either or both are True |
| `not` | Reverses the Boolean result | True becomes False and False becomes True |

### Example 1 — Using `and`

```python
age = 25
salary = 40000

if age >= 18 and salary >= 30000:
    print("Eligible for initial screening")
else:
    print("Not eligible")
```

**Output:** `Eligible for initial screening`

Both conditions are True, so the complete expression is True.

### Example 2 — Using `or`

```python
membership = "Regular"
purchase = 6000

if membership == "Premium" or purchase > 5000:
    print("Special offer available")
else:
    print("No special offer")
```

**Output:** `Special offer available`

The membership condition is False, but the purchase condition is True. One True condition is enough for `or`.

### Example 3 — Using `not`

```python
is_available = False

if not is_available:
    print("Product is currently unavailable")
```

**Output:** `Product is currently unavailable`

`not` reverses False to True, so the block executes.

## 2.6 Nested `if` Statements

A **nested if** means placing one if statement inside another. It is useful when a second decision should be checked only after the first decision is satisfied.

### Syntax

```python
if first_condition:
    if second_condition:
        statement
    else:
        alternative_statement
else:
    outer_alternative
```

### Example 1 — Driving eligibility

```python
age = 25
has_license = True

if age >= 18:
    if has_license:
        print("Eligible to drive")
    else:
        print("Driving license required")
else:
    print("Age requirement not satisfied")
```

**Output:** `Eligible to drive`

### Example 2 — Account and payment status

```python
account_active = True
payment_due = False

if account_active:
    if not payment_due:
        print("Account is in good standing")
    else:
        print("Payment is pending")
else:
    print("Account is inactive")
```

**Output:** `Account is in good standing`

### Example 3 — Model deployment check

```python
accuracy = 0.94
recall = 0.82

if accuracy >= 0.90:
    if recall >= 0.80:
        print("Model meets both evaluation targets")
    else:
        print("Review recall before deployment")
else:
    print("Review overall model performance")
```

**Output:** `Model meets both evaluation targets`

A real deployment decision needs more checks than these two metrics. This example only demonstrates nested conditions.

# 3. Loops — Repeating Operations

## 3.1 What Is a Loop?

A **loop** is a programming structure that repeatedly executes a block of code. We use loops when the same operation must be performed for multiple values or until a condition changes.

**Without loops:** If a file contains 1,000 transactions, we would have to write the same calculation many times.

**With loops:** We write the operation once and let Python repeat it for each transaction.

Python provides two main loop types:

- `for` loop: repeats over items in an iterable, such as a list, string, tuple or range.
- `while` loop: repeats as long as a condition remains True.

## 3.2 What Is a `for` Loop?

A **`for` loop** takes items from an iterable one at a time and executes the indented block for each item.

### Syntax

```python
for variable in iterable:
    statements
```

- `for` starts the loop.
- `variable` receives the current item on each iteration.
- `in` connects the variable to the iterable.
- `iterable` is the collection or sequence being processed.
- The indented block executes once for each item.

### Example 1 — Loop through a list

```python
sales = [1000, 1500, 2000]

for amount in sales:
    print(amount)
```

**Output**

```text
1000
1500
2000
```

Python takes 1000 first, then 1500, then 2000. The block runs three times.

### Example 2 — Use `range()`

```python
for number in range(1, 6):
    print(number)
```

**Output**

```text
1
2
3
4
5
```

`range(1, 6)` generates integers starting at 1 and stops before 6.

### Example 3 — Calculate total sales

```python
sales = [1000, 1500, 2000]
total = 0

for amount in sales:
    total = total + amount

print("Total sales:", total)
```

**Output:** `Total sales: 4500`

On each iteration, the current amount is added to total.

## 3.3 What Is a `while` Loop?

A **`while` loop** repeats a block of code as long as its condition is True. It is useful when the number of repetitions is not known in advance.

### Syntax

```python
while condition:
    statements
    update_condition_variable
```

The condition must eventually become False when the loop is intended to stop. Otherwise, the loop may run forever.

### Example 1 — Print numbers

```python
number = 1

while number <= 5:
    print(number)
    number = number + 1
```

**Output:** 1, 2, 3, 4, 5 (each on a separate line).

The counter increases after each iteration. When it becomes 6, the condition is False and the loop stops.

### Example 2 — Retry counter

```python
attempt = 1

while attempt <= 3:
    print("Attempt:", attempt)
    attempt += 1
```

**Output**

```text
Attempt: 1
Attempt: 2
Attempt: 3
```

### Example 3 — Process a limited queue

```python
pending_jobs = 3

while pending_jobs > 0:
    print("Processing one job")
    pending_jobs -= 1

print("All jobs processed")
```

**Output**

```text
Processing one job
Processing one job
Processing one job
All jobs processed
```

## 3.4 Difference Between `for` and `while`

| `for` loop | `while` loop |
|---|---|
| Iterates over an iterable | Repeats while a condition is True |
| Often used when processing a collection | Often used when repetitions depend on a condition |
| Gets the next item automatically | The developer usually updates the condition variable |
| Example: process every row in a list | Example: retry until success or attempt limit |

# 4. Loop Control Statements

## 4.1 What Is `break`?

The **`break` statement** immediately terminates the nearest enclosing loop. Python continues with the first statement after that loop.

### Syntax

```python
for item in iterable:
    if stop_condition:
        break
    statements
```

### Example 1 — Stop at a number

```python
for number in range(1, 6):
    if number == 4:
        break
    print(number)
```

**Output:** 1, 2, 3

### Example 2 — Find the first matching record

```python
customer_ids = [101, 102, 103, 104]

for customer_id in customer_ids:
    if customer_id == 103:
        print("Customer found")
        break
```

**Output:** `Customer found`

### Example 3 — Stop after finding a failed model

```python
model_scores = [0.93, 0.91, 0.62, 0.95]

for score in model_scores:
    if score < 0.70:
        print("Model requires investigation")
        break
```

**Output:** `Model requires investigation`

## 4.2 What Is `continue`?

The **`continue` statement** skips the remaining statements in the current iteration and moves to the next iteration of the loop. It does not terminate the entire loop.

### Syntax

```python
for item in iterable:
    if skip_condition:
        continue
    statements
```

### Example 1 — Skip one number

```python
for number in range(1, 6):
    if number == 3:
        continue
    print(number)
```

**Output:** 1, 2, 4, 5

### Example 2 — Skip invalid sales values

```python
sales = [1000, -50, 1500, 2000]

for amount in sales:
    if amount < 0:
        continue
    print(amount)
```

**Output:** 1000, 1500, 2000

### Example 3 — Skip missing values

```python
scores = [85, None, 90, None, 78]

for score in scores:
    if score is None:
        continue
    print(score)
```

**Output:** 85, 90, 78

This is a simple Python example. Pandas datasets usually use dedicated missing-value handling methods.

## 4.3 Difference Between `break` and `continue`

| `break` | `continue` |
|---|---|
| Stops the nearest loop completely | Skips only the current iteration |
| Execution moves after the loop | Execution moves to the next iteration |
| Use when no more processing is needed | Use when one item should be skipped |

# 5. Functions — Reusable Blocks of Code

## 5.1 What Is a Function?

A **function** is a named, reusable block of code designed to perform a particular task. A function runs when it is called.

Functions help us:
- Avoid writing the same logic repeatedly.
- Divide a large program into smaller tasks.
- Test and debug logic independently.
- Make data preparation and model workflows easier to maintain.

**Without functions:** If the same revenue calculation is copied into five places, changing the calculation requires updating five places. This creates maintenance errors.

## 5.2 Function Syntax

```python
def function_name(parameters):
    statements
    return result
```

- `def` is the keyword used to define a function.
- `function_name` is the name used to call it.
- `parameters` are input names listed in the definition.
- The indented block contains the function logic.
- `return` sends a result back to the caller. It is optional if the function is only intended to perform an action.

## 5.3 What Are Parameters and Arguments?

A **parameter** is the variable written inside the function definition. An **argument** is the actual value supplied when the function is called.

```python
def greet(name):       # name is a parameter
    print("Hello", name)

greet("Asha")          # "Asha" is an argument
```

Output: `Hello Asha`

## 5.4 Example 1 — Function Without Parameters

```python
def welcome():
    print("Welcome to Machine Learning")

welcome()
```

**Output:** `Welcome to Machine Learning`

The function does not require an input value.

## 5.5 Example 2 — Function With Parameters

```python
def calculate_total(price, quantity):
    total = price * quantity
    print("Total:", total)

calculate_total(250, 4)
```

**Output:** `Total: 1000`

The function receives price and quantity, multiplies them and displays the result.

## 5.6 Example 3 — Function With a Return Value

```python
def calculate_average(scores):
    average = sum(scores) / len(scores)
    return average

result = calculate_average([80, 90, 70])
print("Average:", result)
```

**Output:** `Average: 80.0`

The function returns the calculated value. The caller stores it in result and can reuse it.

## 5.7 Difference Between `print()` and `return`

| `print()` | `return` |
|---|---|
| Displays information on the screen | Sends a value back to the caller |
| Mainly used to show output | Used when the result must be reused |
| Does not automatically provide a result to the calling expression | Ends the function and returns a value |

Example:

```python
def add_and_print(a, b):
    print(a + b)

def add_and_return(a, b):
    return a + b

visible_result = add_and_print(10, 20)
reusable_result = add_and_return(10, 20)

print("Print function returned:", visible_result)
print("Return function returned:", reusable_result)
```

**Output**

```text
30
30
Print function returned: None
Return function returned: 30
```

The first function displays 30 but does not explicitly return it. Python therefore returns `None`.

## 5.8 Default and Keyword Arguments

### Example 1 — Default argument

```python
def greet(name="Student"):
    print("Hello", name)

greet()
greet("Asha")
```

Output: `Hello Student` and `Hello Asha`.

### Example 2 — Keyword arguments

```python
def calculate_bill(price, quantity):
    return price * quantity

print(calculate_bill(quantity=3, price=200))
```

Output: `600`.

### Example 3 — Reusable ML metric function

```python
def check_accuracy(accuracy, target=0.90):
    if accuracy >= target:
        return "Target achieved"
    return "Needs improvement"

print(check_accuracy(0.94))
print(check_accuracy(0.82, 0.85))
```

Output:

```text
Target achieved
Needs improvement
```

This is a small illustration of wrapping a repeated decision in a function. Real model evaluation should include more than accuracy.

# 6. Putting Conditions, Loops and Functions Together

A real Python program often uses all three concepts:

- A function contains the reusable logic.
- A loop processes multiple records.
- A condition decides what to do with each record.

### Example — Categorize multiple sales records

```python
def classify_sales(amount):
    if amount >= 100000:
        return "Excellent"
    elif amount >= 50000:
        return "Good"
    else:
        return "Needs improvement"

monthly_sales = [120000, 75000, 30000]

for amount in monthly_sales:
    category = classify_sales(amount)
    print(amount, category)
```

**Output**

```text
120000 Excellent
75000 Good
30000 Needs improvement
```

### Example — Find models below a recall target

```python
def review_recall(model_name, recall):
    if recall < 0.80:
        return model_name + " needs review"
    return model_name + " meets the recall target"

models = [("Model A", 0.91), ("Model B", 0.72), ("Model C", 0.84)]

for name, score in models:
    print(review_recall(name, score))
```

**Output**

```text
Model A meets the recall target
Model B needs review
Model C meets the recall target
```

### Example — Sum valid transaction amounts

```python
def total_valid_transactions(amounts):
    total = 0
    for amount in amounts:
        if amount is None or amount < 0:
            continue
        total += amount
    return total

transactions = [100, 250, None, -20, 150]
print(total_valid_transactions(transactions))
```

**Output:** `500`

This example skips missing and negative values. A real financial pipeline must investigate why values are missing or negative rather than silently discarding them.

# 7. Common Beginner Mistakes

| Mistake | Why it causes a problem | Correct approach |
|---|---|---|
| Using `=` instead of `==` in a comparison | Assignment is not a comparison | Use `==` to compare values |
| Forgetting the colon | Python syntax error | End compound statements with `:` |
| Incorrect indentation | Code belongs to the wrong block or raises an error | Use consistent indentation |
| Putting broad `elif` conditions first | Specific branches may never execute | Order conditions carefully |
| Not updating a while-loop variable | Can create an infinite loop | Ensure the condition can become False |
| Using `break` when only one record should be skipped | Stops the entire loop | Use `continue` to skip one iteration |
| Printing instead of returning a value | The caller cannot reuse the printed result | Use `return` when a result is needed |
| Dividing by the length of an empty list | Can raise ZeroDivisionError | Check that the list is not empty |

# 8. Short Scenario Questions

Try to answer each in two or three sentences. Focus on the decision, not on writing a large program.

1. A store has 12 products, but only 3 are in stock. Which Python concept helps process every product, and which concept checks stock availability?
2. A model evaluation script checks five models. It should report every model with recall below 0.80. Would you use `break` or `continue`, and why?
3. A customer gets free delivery if the order is at least ₹1,000. Which condition structure would you use?
4. A data-cleaning operation is required in several parts of a program. Why is a function better than copying the same code?
5. A while loop never stops. What should you inspect first?
6. A model has 95% accuracy but only 40% recall. Should the program simply print “Model is good”? What additional evaluation is needed?

# 9. Connection to Machine Learning Development

These Python fundamentals are used throughout an ML workflow:

| ML task | Python concept |
|---|---|
| Check whether a dataset has enough records | Condition |
| Apply a rule to each record | Loop |
| Reuse a missing-value validation rule | Function |
| Skip a malformed record during a controlled processing task | `continue` |
| Stop a search after a required item is found | `break` |
| Choose a model review status based on metrics | `if-elif-else` |

Later, libraries such as NumPy and Pandas provide vectorized operations that can process entire arrays or columns without manually writing Python loops. Learn loops first because they explain the basic logic behind repeated processing.

---

**Day 03 takeaway:** Conditions make decisions, loops repeat work, and functions organize reusable logic. A good developer understands what each construct does, when to use it and how to prevent common errors.
