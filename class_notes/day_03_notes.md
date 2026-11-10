# Day 03: Conditions, Loops and Functions

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate  
**Learning focus:** Understand the concepts, apply them correctly, and explain decisions in an interview.

---

## 1. Learning Objectives

- Write conditional logic using `if`, `elif` and `else`.
- Combine conditions using logical operators.
- Repeat operations using `for` and `while` loops.
- Control loops using `break` and `continue`.
- Create reusable functions with parameters and return values.
- Apply these fundamentals to simple data and ML workflows.

## 2. Conditions — Making Decisions

A condition allows a program to choose which block of code should run.

### 2.1 `if`

```python
model_accuracy = 0.91

if model_accuracy >= 0.90:
    print("Accuracy meets the target")
```

Output: `Accuracy meets the target`

Explanation:

- Python evaluates `model_accuracy >= 0.90`.
- The condition is `True`, so the indented block runs.
- Indentation defines the block in Python.

### 2.2 `if` and `else`

```python
stock = 4

if stock > 0:
    print("Product is available")
else:
    print("Product is out of stock")
```

Output: `Product is available`

`else` runs when the `if` condition is false.

### 2.3 `if`, `elif` and `else`

Use `elif` to test additional conditions in order.

```python
score = 72

if score >= 85:
    grade = "A"
elif score >= 70:
    grade = "B"
elif score >= 50:
    grade = "C"
else:
    grade = "Needs improvement"

print(grade)
```

Output: `B`

Python checks from top to bottom and executes the first matching branch. The order matters. For example, checking `score >= 50` first would capture a score of 90 and prevent the A branch from running.

### 2.4 Nested conditions

```python
customer_active = True
payment_due = False

if customer_active:
    if not payment_due:
        print("Customer account is in good standing")
    else:
        print("Review pending payment")
else:
    print("Account is inactive")
```

Use nested conditions only when they make the logic clearer. Complex nested logic can often be simplified with named Boolean variables.

## 3. Loops — Repeating Work

Loops execute a block repeatedly. They are useful for processing records, calculating summaries and running repeated experiments.

### 3.1 `for` loop

A `for` loop iterates over items in a sequence or iterable.

```python
sales = [1200, 1500, 900]

for amount in sales:
    print(amount)
```

Output:

```text
1200
1500
900
```

The variable `amount` receives each item one at a time.

### 3.2 `range()`

`range(stop)` produces numbers starting at 0 and ending before `stop`.

```python
for day in range(1, 4):
    print("Day", day)
```

Output:

```text
Day 1
Day 2
Day 3
```

`range(1, 4)` includes 1, 2 and 3, but not 4.

### 3.3 Calculate a total using a loop

```python
monthly_sales = [10000, 12000, 9000]
total = 0

for sale in monthly_sales:
    total = total + sale

print("Total sales:", total)
```

Output: `Total sales: 31000`

At each iteration, the current sale is added to `total`.

### 3.4 `while` loop

A `while` loop repeats as long as its condition remains true.

```python
attempt = 1

while attempt <= 3:
    print("Attempt", attempt)
    attempt += 1
```

Output:

```text
Attempt 1
Attempt 2
Attempt 3
```

The variable changes on each iteration. Without `attempt += 1`, the condition would remain true and the loop could run forever.

## 4. Loop Control Statements

### `break`

Stops the nearest loop immediately.

```python
scores = [0.72, 0.81, 0.95, 0.78]

for score in scores:
    if score >= 0.90:
        print("Target reached:", score)
        break
```

Output: `Target reached: 0.95`

### `continue`

Skips the rest of the current iteration and moves to the next one.

```python
values = [10, None, 20, None, 30]

for value in values:
    if value is None:
        continue
    print(value)
```

Output:

```text
10
20
30
```

This example skips missing values for display. In a real data pipeline, missing values should be measured and handled according to the data definition; silently skipping them can hide a data-quality problem.

## 5. Functions — Reusable Blocks of Code

A function is a named block of code that performs a specific task. Functions reduce repetition and make programs easier to test and maintain.

### 5.1 Define and call a function

```python
def greet_student():
    print("Welcome to Machine Learning")

greet_student()
```

Output: `Welcome to Machine Learning`

- `def` begins a function definition.
- `greet_student` is the function name.
- The indented line is the function body.
- Calling `greet_student()` executes it.

### 5.2 Function parameters

Parameters allow a function to receive input values.

```python
def calculate_revenue(quantity, unit_price):
    revenue = quantity * unit_price
    print(revenue)

calculate_revenue(5, 200)
```

Output: `1000`

`quantity` and `unit_price` are parameters. The values `5` and `200` are arguments passed during the call.

### 5.3 Return values

`return` sends a result back to the code that called the function.

```python
def calculate_revenue(quantity, unit_price):
    return quantity * unit_price

result = calculate_revenue(5, 200)
print("Revenue:", result)
```

Output: `Revenue: 1000`

**`print` vs `return`:** `print()` displays information. `return` gives a value back so another part of the program can store it, compare it or calculate with it.

### 5.4 Default parameters

```python
def calculate_discount(price, discount_percent=10):
    discount = price * discount_percent / 100
    return price - discount

print(calculate_discount(1000))
print(calculate_discount(1000, 20))
```

Output:

```text
900.0
800.0
```

When the second argument is omitted, the function uses the default 10% discount.

### 5.5 Keyword arguments

```python
def predict_delivery(distance_km, traffic_factor):
    return distance_km * traffic_factor

minutes = predict_delivery(traffic_factor=2.5, distance_km=8)
print(minutes)
```

Output: `20.0`

Keyword arguments make calls clearer and allow arguments to be supplied by name.

### 5.6 Validate function inputs

```python
def average_score(scores):
    if not scores:
        raise ValueError("At least one score is required")
    return sum(scores) / len(scores)

print(average_score([0.8, 0.9, 1.0]))
```

Output: `0.9`

The function rejects an empty list because an average is undefined when there are no values.

## 6. Combining Conditions, Loops and Functions

```python
def classify_risk(score):
    if score >= 0.80:
        return "High risk"
    elif score >= 0.50:
        return "Medium risk"
    else:
        return "Low risk"

customer_scores = [0.91, 0.64, 0.32]

for score in customer_scores:
    label = classify_risk(score)
    print(score, "->", label)
```

Output:

```text
0.91 -> High risk
0.64 -> Medium risk
0.32 -> Low risk
```

This illustrates rule-based categorization, not a trained classifier. In a real ML system, the score may be produced by a model and thresholds must be selected using validation evidence and business costs.

## 7. Industry Scenario — Processing Customer Orders

A retailer receives many orders and wants to calculate total revenue while flagging orders that need review.

```python
orders = [
    {"order_id": "A101", "amount": 1200},
    {"order_id": "A102", "amount": 850},
    {"order_id": "A103", "amount": 5200},
]

def needs_review(amount, review_limit=5000):
    return amount >= review_limit

total_revenue = 0

for order in orders:
    total_revenue += order["amount"]
    if needs_review(order["amount"]):
        print("Review order:", order["order_id"])

print("Total revenue:", total_revenue)
```

Output:

```text
Review order: A103
Total revenue: 7250
```

The loop processes every order, the function centralizes the review rule, and the condition determines which orders need attention.

## 8. Common Mistakes

| Mistake | Result | Correction |
|---|---|---|
| Forgetting the colon after `if`, `for`, `while` or `def` | SyntaxError | Add `:` |
| Incorrect indentation | SyntaxError or wrong block | Align statements consistently |
| Infinite `while` loop | Program does not finish | Update the condition variable or use a safe exit |
| Using `print` when a result is needed | Caller receives `None` | Use `return` |
| Writing same logic repeatedly | Hard to maintain | Place reusable logic in a function |
| Checking a broad condition before a narrow one | Wrong branch may run | Order conditions from specific to general |
| Skipping invalid records silently | Data-quality problems are hidden | Count, log and explain rejected records |

## 9. Quick Revision

- Conditions select which code runs.
- `for` iterates over items; `while` repeats while a condition is true.
- `break` exits a loop; `continue` skips the current iteration.
- Functions package reusable logic; parameters accept inputs and `return` supplies a result.
- Validate inputs and make business rules explicit.

---

## Expanded Beginner-to-Developer Teaching Layer

# Day 03: Conditions, loops and functions

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

**Conditions, loops and functions**

A condition selects a path based on True or False. A loop repeats work. A function is a named reusable block that accepts inputs and may return a result.

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

Use if/elif/else for decisions, for/while for repetition, and functions to organize repeated logic and make it testable.

**Without this concept or step:** teams may produce incorrect outputs, waste compute, make poor business decisions or trust a result that does not work on new data. The exact risk depends on the application; investigate rather than assume.

## 4. Syntax, Structure and How to Read It

Watch indentation, infinite while loops, off-by-one ranges, missing return statements and functions that modify shared data unexpectedly.

### Read the syntax carefully

1. Identify the library or Python feature being used.
2. Identify the input variable and its expected type/shape.
3. Identify the operation or method being called.
4. Check optional parameters and their defaults.
5. Inspect the output and verify that it matches the intended result.

### Practical example

```python
# A small, readable example related to the lesson
def review_payment(amount):\n    if amount >= 50000:\n        return 'manual review'\n    return 'normal review'\n\nprint(review_payment(72000))
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

**Scenario:** A payment service checks whether a transaction exceeds a review threshold. It processes thousands of transactions, so a loop applies the same validation consistently; a function keeps the rule in one place.

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

Syntax: if condition: ... elif condition: ... else: ...; for item in items: ...; def function_name(parameter): ... return result. Example: def classify_amount(x): return 'review' if x >= 50000 else 'normal'.

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
| What is it? | A condition selects a path based on True or False. A loop repeats work. A function is a named reusable block that accepts inputs and may return a result. |
| Why use it? | Use if/elif/else for decisions, for/while for repetition, and functions to organize repeated logic and make it testable. |
| Main practical use | A payment service checks whether a transaction exceeds a review threshold. It processes thousands of transactions, so a loop applies the same validation consistently; a function keeps the rule in one place. |
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

Use the matching `day_03_interview.md` file for detailed interview questions, model answers, real-world troubleshooting scenarios and follow-up questions.

---

**Learning standard:** Understand the meaning first, then syntax, then practical use, then engineering trade-offs. A student should be able to explain the idea without reading code from the screen.


---

## Beginner-to-Developer Expansion: More Examples and Real-World Reasoning

### A. Start with the simplest meaning
Before memorizing a function, explain the idea in your own words. Identify the input, output and the reason the operation exists. A correct syntax call on the wrong data is still a wrong solution.

### B. Additional practical scenario
A risk review process checks every incoming payment. The loop processes records, a condition decides whether an alert is needed and a function centralizes the decision rule so it can be tested once.

### C. Syntax and use: how to choose
A function signature declares the name and parameters; return sends a result back to the caller. A loop repeats over a sequence; if/elif/else selects one branch.

### D. What happens if this is skipped?
Without a reusable function, the same business rule may be copied into several places and drift over time.

### E. Senior developer review checklist
- Is the business definition clear?
- Are input types, units, shape and missing values checked?
- Can the result be verified on a small hand-checkable example?
- Have normal, boundary and invalid cases been considered?
- Is the approach simple enough to explain and maintain?
- Are limitations and possible failure costs documented?

### F. Explain it in an interview
Use this order: definition → why it is needed → syntax or method → small example → real business scenario → what can go wrong → how you validate the result. Avoid answering with only a library function name.
