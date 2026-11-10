# Day 04: Lists, Dictionaries, Tuples and Sets

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate  
**Learning focus:** Understand the concepts, apply them correctly, and explain decisions in an interview.

---

## 1. Learning Objectives

- Understand Python's four common collection types.
- Store, retrieve, update and remove records.
- Choose a collection based on ordering, mutability, uniqueness and lookup needs.
- Use nested collections to represent structured records.
- Understand why collection structure matters when preparing ML input data.

## 2. Why Do We Need Collections?

A variable stores one reference to a value. Real applications often need to store many related values: customer records, model metrics, feature names or transaction details.

Python provides built-in collection types for these needs.

## 3. Lists

A **list** is an ordered, mutable collection. It allows duplicate values and can contain different data types, although using a consistent type is often clearer.

### Create and access a list

```python
sales = [1200, 1500, 900, 1200]

print(sales[0])
print(sales[-1])
print(len(sales))
```

Output:

```text
1200
1200
4
```

- Indexing begins at 0.
- `-1` means the last item.
- `len()` returns the number of items.

### Common list methods

```python
sales = [1200, 1500, 900]

sales.append(1800)       # Add one item at the end
sales.extend([2000, 2200])  # Add multiple items
sales.insert(1, 1300)    # Insert at index 1
sales.remove(900)        # Remove first matching value
last_value = sales.pop() # Remove and return last item

print(sales)
print(last_value)
```

Output:

```text
[1200, 1300, 1500, 1800, 2000]
2200
```

Other useful operations:

| Operation | Meaning |
|---|---|
| `items[index]` | Read item |
| `items[index] = value` | Update item |
| `items[1:4]` | Slice from index 1 up to but not including 4 |
| `items.sort()` | Sort list in place |
| `sorted(items)` | Return a new sorted list |
| `items.clear()` | Remove all items |

### List comprehension

A concise way to create a list from an iterable.

```python
prices = [100, 250, 400]
with_tax = [price * 1.18 for price in prices]
print(with_tax)
```

Output: `[118.0, 295.0, 472.0]`

## 4. Tuples

A **tuple** is an ordered, immutable collection. After creation, its elements cannot be reassigned or removed.

```python
image_size = (224, 224, 3)

print(image_size[0])
print(len(image_size))
```

Output:

```text
224
3
```

Use a tuple for a fixed group of values, such as image dimensions or a coordinate pair.

```python
width, height = (640, 480)
print(width)
print(height)
```

A tuple with one item needs a trailing comma: `(5,)`. Without the comma, `(5)` is simply the integer `5`.

**Why immutability helps:** A fixed configuration is less likely to be accidentally modified during program execution.

## 5. Dictionaries

A **dictionary** stores key-value pairs. Keys are unique and are used to retrieve corresponding values. Dictionaries are mutable.

### Create and read a dictionary

```python
customer = {
    "customer_id": "C101",
    "age": 29,
    "city": "Hyderabad",
    "active": True
}

print(customer["customer_id"])
print(customer.get("city"))
print(customer.get("membership", "Not provided"))
```

Output:

```text
C101
Hyderabad
Not provided
```

- `customer[key]` raises `KeyError` if the key is absent.
- `get()` returns `None` or a supplied default when the key is absent.

### Add, update and delete

```python
customer["age"] = 30
customer["membership"] = "Gold"
del customer["active"]

print(customer)
```

Output:

```text
{'customer_id': 'C101', 'age': 30, 'city': 'Hyderabad', 'membership': 'Gold'}
```

### Iterate through a dictionary

```python
for key, value in customer.items():
    print(key, "=", value)
```

`items()` provides key-value pairs.

## 6. Sets

A **set** is an unordered collection of unique elements. It is useful for removing duplicates and checking membership.

```python
categories = {"Laptop", "Mobile", "Laptop", "Tablet"}
print(categories)
print("Mobile" in categories)
```

Output conceptually contains each unique item once, and the membership check returns `True`. The display order should not be relied on.

### Set operations

```python
online_customers = {"C1", "C2", "C3"}
premium_customers = {"C2", "C3", "C4"}

print(online_customers | premium_customers)  # Union
print(online_customers & premium_customers)  # Intersection
print(online_customers - premium_customers)  # Difference
```

Results:

- Union: customers in either set.
- Intersection: customers in both sets (`C2`, `C3`).
- Difference: customers in the first set but not the second (`C1`).

Use sets when uniqueness and membership are important, not when row order must be preserved.

## 7. Comparison of Collections

| Feature | List | Tuple | Dictionary | Set |
|---|---|---|---|---|
| Syntax | `[ ]` | `( )` | `{key: value}` | `{value1, value2}` |
| Ordered | Yes | Yes | Insertion order preserved in modern Python | No guaranteed positional order |
| Mutable | Yes | No | Yes | Yes |
| Duplicates | Allowed | Allowed | Keys unique | Elements unique |
| Access | Index | Index | Key | Membership / iteration |
| Common use | Sequence of records/values | Fixed configuration | Structured named fields | Unique values |

## 8. Nested Collections — Representing Business Records

A list of dictionaries is a common way to represent a small collection of records.

```python
customers = [
    {"id": "C101", "age": 24, "monthly_spend": 1200},
    {"id": "C102", "age": 35, "monthly_spend": 2500},
    {"id": "C103", "age": 29, "monthly_spend": 1800},
]

for customer in customers:
    print(customer["id"], customer["monthly_spend"])
```

Output:

```text
C101 1200
C102 2500
C103 1800
```

The outer list holds records; each dictionary stores named attributes for one customer.

### Calculate average spend

```python
customers = [
    {"id": "C101", "spend": 1200},
    {"id": "C102", "spend": 2500},
    {"id": "C103", "spend": 1800},
]

spend_values = [record["spend"] for record in customers]
average_spend = sum(spend_values) / len(spend_values)
print(average_spend)
```

Output: `1833.3333333333333`

For larger analytical datasets, Pandas DataFrames are generally more convenient than manually managing lists of dictionaries.

## 9. Industry Scenario — Building a Customer Feature Record

A subscription company stores information for each customer. A dictionary is useful for named attributes, a list is useful for multiple customer records, and a set can identify unique plan names.

```python
customers = [
    {"id": "U1", "plan": "Basic", "months": 4, "support_calls": 3},
    {"id": "U2", "plan": "Premium", "months": 18, "support_calls": 0},
    {"id": "U3", "plan": "Basic", "months": 2, "support_calls": 5},
]

plans = {customer["plan"] for customer in customers}

for customer in customers:
    print(customer["id"], customer["months"], customer["support_calls"])

print("Unique plans:", plans)
```

The values `months` and `support_calls` could become candidate numerical features after validation and correct target/feature timing checks. The customer ID is usually an identifier, not automatically a meaningful predictive feature.

## 10. Common Mistakes

- Accessing a list index that does not exist causes `IndexError`.
- Accessing a missing dictionary key with square brackets causes `KeyError`.
- Trying to modify a tuple causes `TypeError`.
- Expecting a set to preserve a specific order leads to fragile code.
- Using mutable lists or dictionaries as dictionary keys is not allowed because they are unhashable.
- Removing duplicates without defining what makes two records duplicates can delete valid data.

## 11. Quick Revision

- List: ordered, mutable, duplicates allowed.
- Tuple: ordered, immutable, duplicates allowed.
- Dictionary: key-value mapping with unique keys.
- Set: unique elements, useful for membership and set operations.
- A list of dictionaries can represent structured records; Pandas is generally preferred for larger tabular analysis.

---

## Expanded Beginner-to-Developer Teaching Layer

# Day 04: Lists, dictionaries, tuples and sets

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

**Lists, dictionaries, tuples and sets**

A list is an ordered mutable sequence; a tuple is an ordered immutable sequence; a dictionary maps unique keys to values; a set stores unique elements without positional indexing.

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

Choose the structure based on access pattern: ordered records, fixed grouped values, key-based lookup or uniqueness/membership checks.

**Without this concept or step:** teams may produce incorrect outputs, waste compute, make poor business decisions or trust a result that does not work on new data. The exact risk depends on the application; investigate rather than assume.

## 4. Syntax, Structure and How to Read It

Do not use a list when fast key lookup is needed, assume sets preserve order, or mutate a tuple.

### Read the syntax carefully

1. Identify the library or Python feature being used.
2. Identify the input variable and its expected type/shape.
3. Identify the operation or method being called.
4. Check optional parameters and their defaults.
5. Inspect the output and verify that it matches the intended result.

### Practical example

```python
# A small, readable example related to the lesson
customer = {'name': 'Asha', 'orders': 3}\ncustomer['orders'] += 1\nprint(customer)
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

**Scenario:** A customer service application stores ordered recent tickets in a list, customer profile fields in a dictionary, fixed coordinates in a tuple and unique product categories in a set.

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

Syntax: items = [10,20]; profile = {'name':'Asha','age':25}; point = (17.4,78.4); unique_ids = {101,102}. Use append(), get(), keys(), values(), add(), update(), discard().

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
| What is it? | A list is an ordered mutable sequence; a tuple is an ordered immutable sequence; a dictionary maps unique keys to values; a set stores unique elements without positional indexing. |
| Why use it? | Choose the structure based on access pattern: ordered records, fixed grouped values, key-based lookup or uniqueness/membership checks. |
| Main practical use | A customer service application stores ordered recent tickets in a list, customer profile fields in a dictionary, fixed coordinates in a tuple and unique product categories in a set. |
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

Use the matching `day_04_interview.md` file for detailed interview questions, model answers, real-world troubleshooting scenarios and follow-up questions.

---

**Learning standard:** Understand the meaning first, then syntax, then practical use, then engineering trade-offs. A student should be able to explain the idea without reading code from the screen.
