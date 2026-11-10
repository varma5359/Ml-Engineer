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