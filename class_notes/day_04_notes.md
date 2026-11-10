# Day 04: Lists, Dictionaries, Tuples and Sets

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, Mathematics and Statistics Foundations  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

By the end of this lesson, you should be able to:

- Explain why data structures are needed.
- Create, inspect and update Python lists.
- Use positive and negative indexing and slicing.
- Add, replace, remove and search list elements.
- Iterate through lists and use list comprehensions.
- Create tuples and explain their immutability.
- Create sets to store unique values and compare groups.
- Create dictionaries to store key-value records.
- Access, update, remove and iterate through dictionary items.
- Choose a suitable data structure for simple ML and data-processing tasks.

## 2. Why do we need data structures?

A variable can store one reference, but real applications often need to work with many values together. A dataset may contain thousands of customer records, product categories, feature names and model settings.

Instead of creating a separate variable for every item, Python provides collections called **data structures**.

### 2.1 Without a collection

```python
customer_1 = "C001"
customer_2 = "C002"
customer_3 = "C003"

print(customer_1)
print(customer_2)
print(customer_3)
```

**Expected output**

```text
C001
C002
C003
```

**Explanation:** This works for a few values, but it becomes difficult to maintain when there are hundreds or thousands of records.

### 2.2 Store related values together

```python
customer_ids = ["C001", "C002", "C003"]

print(customer_ids)
```

**Expected output**

```text
['C001', 'C002', 'C003']
```

**Explanation:** One list stores all three IDs. We can loop through it, add values and access a particular item.

## 3. Four important Python data structures

| Structure | Main purpose | Example |
|---|---|---|
| List | Ordered, changeable collection | `[10, 20, 30]` |
| Tuple | Ordered, unchangeable collection | `(10, 20, 30)` |
| Set | Collection of unique values | `{10, 20, 30}` |
| Dictionary | Key-value mapping | `{"name": "Ravi"}` |

A simple way to choose:

- Use a **list** when order matters and values may change.
- Use a **tuple** when you want an ordered collection that should not be changed.
- Use a **set** when uniqueness and set comparisons matter.
- Use a **dictionary** when each value needs a meaningful key.

## 4. Python lists

A list stores multiple elements in one collection. Lists are ordered, mutable and allow duplicate values. They can contain different data types, although in data work it is often clearer to keep similar values together.

### 4.1 Create a list

```python
feature_names = ["age", "monthly_charges", "tenure"]

print(feature_names)
print(type(feature_names))
```

**Expected output**

```text
['age', 'monthly_charges', 'tenure']
<class 'list'>
```

**Explanation**

- Square brackets create a list.
- Commas separate the elements.
- `type()` confirms that the object is a list.

### 4.2 Create a list of numbers

```python
monthly_sales = [12000, 15000, 18000, 22000]

print(monthly_sales)
```

**Expected output**

```text
[12000, 15000, 18000, 22000]
```

**Explanation:** A list can store multiple numeric values. This is useful for small examples before learning NumPy arrays.

### 4.3 Create a mixed-type list

```python
customer_record = ["C101", 29, 4500.75, True]

print(customer_record)
```

**Expected output**

```text
['C101', 29, 4500.75, True]
```

**Explanation:** Python permits different types in one list. In a real tabular dataset, a Pandas DataFrame is usually more appropriate for columns with different data types.

### 4.4 Create an empty list

```python
predictions = []

print(predictions)
print(len(predictions))
```

**Expected output**

```text
[]
0
```

**Explanation:** An empty list is useful when we intend to collect values gradually.

## 5. List indexing

Indexing means retrieving one element by its position. Python starts counting from zero.

Consider:

| Element | `"age"` | `"income"` | `"tenure"` | `"charges"` |
|---|---:|---:|---:|---:|
| Positive index | 0 | 1 | 2 | 3 |
| Negative index | -4 | -3 | -2 | -1 |

### 5.1 Access the first element

```python
feature_names = ["age", "income", "tenure", "charges"]

print(feature_names[0])
```

**Expected output**

```text
age
```

**Explanation:** Index zero refers to the first element.

### 5.2 Access the second element

```python
feature_names = ["age", "income", "tenure", "charges"]

print(feature_names[1])
```

**Expected output**

```text
income
```

**Explanation:** Index one refers to the second element, not the first.

### 5.3 Access the last element with a negative index

```python
feature_names = ["age", "income", "tenure", "charges"]

print(feature_names[-1])
```

**Expected output**

```text
charges
```

**Explanation:** Negative indexing counts from the end. Index `-1` is the last element.

### 5.4 Access the second-last element

```python
feature_names = ["age", "income", "tenure", "charges"]

print(feature_names[-2])
```

**Expected output**

```text
tenure
```

**Explanation:** Index `-2` refers to the second-last element.

### 5.5 What happens with an invalid index?

```python
feature_names = ["age", "income", "tenure"]

print(feature_names[5])
```

**Expected result**

```text
IndexError: list index out of range
```

**Explanation:** The list has only three elements, with valid indexes 0, 1 and 2. Check the list length before using a position that may not exist.

## 6. List slicing

Slicing extracts a portion of a list. The ending position is excluded.

Syntax:

```python
list_name[start:end]
```

### 6.1 Select a range

```python
feature_names = ["age", "income", "tenure", "charges", "contract"]

print(feature_names[1:4])
```

**Expected output**

```text
['income', 'tenure', 'charges']
```

**Explanation:** Start at index 1 and stop before index 4.

### 6.2 Slice from the beginning

```python
feature_names = ["age", "income", "tenure", "charges", "contract"]

print(feature_names[:3])
```

**Expected output**

```text
['age', 'income', 'tenure']
```

**Explanation:** When the start is omitted, Python starts at index zero.

### 6.3 Slice to the end

```python
feature_names = ["age", "income", "tenure", "charges", "contract"]

print(feature_names[2:])
```

**Expected output**

```text
['tenure', 'charges', 'contract']
```

**Explanation:** When the end is omitted, Python continues through the final element.

### 6.4 Use a step

```python
numbers = [10, 20, 30, 40, 50, 60]

print(numbers[::2])
```

**Expected output**

```text
[10, 30, 50]
```

**Explanation:** A step of two selects every second element.

### 6.5 Reverse a list using slicing

```python
monthly_sales = [12000, 15000, 18000, 22000]

print(monthly_sales[::-1])
```

**Expected output**

```text
[22000, 18000, 15000, 12000]
```

**Explanation:** A step of negative one traverses the list backwards. This creates a reversed list; it does not change the original list.

## 7. Updating list elements

Lists are mutable, which means their elements can be changed after creation.

### 7.1 Update one element

```python
feature_names = ["age", "income", "tenure"]

feature_names[1] = "annual_income"

print(feature_names)
```

**Expected output**

```text
['age', 'annual_income', 'tenure']
```

**Explanation:** The value at index one is replaced.

### 7.2 Update a range

```python
status_values = ["new", "old", "inactive", "unknown"]

status_values[1:3] = ["existing", "active"]

print(status_values)
```

**Expected output**

```text
['new', 'existing', 'active', 'unknown']
```

**Explanation:** Slice assignment replaces the selected range.

### 7.3 Replace a range with a different number of elements

```python
columns = ["id", "age", "old_status", "target"]

columns[2:3] = ["contract_type", "payment_method"]

print(columns)
```

**Expected output**

```text
['id', 'age', 'contract_type', 'payment_method', 'target']
```

**Explanation:** Slice assignment can change the length of a list. Here one element is replaced with two elements.

## 8. Adding elements to a list

### 8.1 append()

`append()` adds one object to the end of a list.

```python
feature_names = ["age", "income"]

feature_names.append("tenure")

print(feature_names)
```

**Expected output**

```text
['age', 'income', 'tenure']
```

**Explanation:** The string is added as one new element.

### 8.2 append() with a number

```python
validation_scores = [0.82, 0.85]

validation_scores.append(0.88)

print(validation_scores)
```

**Expected output**

```text
[0.82, 0.85, 0.88]
```

**Explanation:** This is a simple example of collecting experiment results. For a larger experiment-tracking workflow, use a suitable table or tracking tool.

### 8.3 append() with another list

```python
feature_groups = ["customer", "billing"]

feature_groups.append(["usage", "support"])

print(feature_groups)
```

**Expected output**

```text
['customer', 'billing', ['usage', 'support']]
```

**Explanation:** The complete inner list is added as one element. This creates a nested list.

### 8.4 extend()

`extend()` adds each element from another iterable to the end of the list.

```python
feature_names = ["age", "income"]

feature_names.extend(["tenure", "charges"])

print(feature_names)
```

**Expected output**

```text
['age', 'income', 'tenure', 'charges']
```

**Explanation:** The two strings are added separately.

### 8.5 Compare append() and extend()

```python
first_list = ["age", "income"]
first_list.append(["tenure", "charges"])

second_list = ["age", "income"]
second_list.extend(["tenure", "charges"])

print(first_list)
print(second_list)
```

**Expected output**

```text
['age', 'income', ['tenure', 'charges']]
['age', 'income', 'tenure', 'charges']
```

**Explanation:** Use `append()` when adding one object. Use `extend()` when adding each item from another iterable.

### 8.6 insert()

`insert(index, value)` inserts an element at a chosen position.

```python
feature_names = ["age", "tenure"]

feature_names.insert(1, "income")

print(feature_names)
```

**Expected output**

```text
['age', 'income', 'tenure']
```

**Explanation:** The new value is inserted at index one. Existing elements shift to the right.

## 9. Removing elements from a list

### 9.1 remove()

`remove(value)` removes the first matching value.

```python
feature_names = ["age", "income", "old_status"]

feature_names.remove("old_status")

print(feature_names)
```

**Expected output**

```text
['age', 'income']
```

**Explanation:** The method searches for the supplied value and removes its first occurrence.

### 9.2 remove() when a value is absent

```python
feature_names = ["age", "income"]

feature_names.remove("tenure")
```

**Expected result**

```text
ValueError: list.remove(x): x not in list
```

**Explanation:** Python cannot remove a value that is not present. Check membership first when the value may be absent.

### 9.3 pop()

`pop(index)` removes and returns an element. If no index is supplied, it removes the last element.

```python
feature_names = ["age", "income", "tenure"]

removed_feature = feature_names.pop(1)

print(removed_feature)
print(feature_names)
```

**Expected output**

```text
income
['age', 'tenure']
```

**Explanation:** The returned value is stored in `removed_feature`, and the original list becomes shorter.

### 9.4 pop() without an index

```python
validation_scores = [0.81, 0.84, 0.87]

latest_score = validation_scores.pop()

print(latest_score)
print(validation_scores)
```

**Expected output**

```text
0.87
[0.81, 0.84]
```

**Explanation:** The final element is removed and returned.

### 9.5 del

`del` removes an element or a slice by position.

```python
feature_names = ["age", "income", "tenure", "charges"]

del feature_names[1]

print(feature_names)
```

**Expected output**

```text
['age', 'tenure', 'charges']
```

**Explanation:** The element at index one is deleted.

### 9.6 clear()

`clear()` removes every element but keeps the list object.

```python
temporary_results = [0.81, 0.84, 0.87]

temporary_results.clear()

print(temporary_results)
```

**Expected output**

```text
[]
```

**Explanation:** The list is now empty. Use this carefully; it removes all elements.

## 10. Searching and inspecting lists

### 10.1 len()

`len()` returns the number of elements.

```python
feature_names = ["age", "income", "tenure", "charges"]

print(len(feature_names))
```

**Expected output**

```text
4
```

**Explanation:** There are four items in the list.

### 10.2 in

The `in` operator checks whether a value is present.

```python
feature_names = ["age", "income", "tenure"]

print("income" in feature_names)
print("churn" in feature_names)
```

**Expected output**

```text
True
False
```

**Explanation:** The first value exists; the second does not.

### 10.3 not in

```python
feature_names = ["age", "income", "tenure"]

print("target" not in feature_names)
```

**Expected output**

```text
True
```

**Explanation:** The result is true because `"target"` is absent.

### 10.4 count()

`count(value)` returns the number of times a value occurs.

```python
status_values = ["active", "inactive", "active", "active"]

print(status_values.count("active"))
```

**Expected output**

```text
3
```

**Explanation:** The value `"active"` appears three times.

### 10.5 index()

`index(value)` returns the position of the first matching value.

```python
feature_names = ["age", "income", "tenure"]

print(feature_names.index("tenure"))
```

**Expected output**

```text
2
```

**Explanation:** The value `"tenure"` is at index two. If it does not exist, `index()` raises a `ValueError`.

## 11. Sorting and reversing lists

### 11.1 sort() ascending

```python
validation_scores = [0.91, 0.78, 0.86, 0.82]

validation_scores.sort()

print(validation_scores)
```

**Expected output**

```text
[0.78, 0.82, 0.86, 0.91]
```

**Explanation:** `sort()` changes the original list and sorts it in ascending order.

### 11.2 sort() descending

```python
validation_scores = [0.91, 0.78, 0.86, 0.82]

validation_scores.sort(reverse=True)

print(validation_scores)
```

**Expected output**

```text
[0.91, 0.86, 0.82, 0.78]
```

**Explanation:** `reverse=True` asks `sort()` to order values from largest to smallest.

### 11.3 reverse()

```python
experiment_order = ["baseline", "logistic", "tree"]

experiment_order.reverse()

print(experiment_order)
```

**Expected output**

```text
['tree', 'logistic', 'baseline']
```

**Explanation:** `reverse()` reverses the current order. It does not sort values alphabetically or numerically.

### 11.4 sorted()

```python
scores = [0.91, 0.78, 0.86]

sorted_scores = sorted(scores)

print(sorted_scores)
print(scores)
```

**Expected output**

```text
[0.78, 0.86, 0.91]
[0.91, 0.78, 0.86]
```

**Explanation:** `sorted()` creates a new sorted list and leaves the original list unchanged.

## 12. Useful built-in functions with lists

### 12.1 sum()

```python
monthly_sales = [10000, 12000, 15000]

print(sum(monthly_sales))
```

**Expected output**

```text
37000
```

**Explanation:** `sum()` adds the numeric elements.

### 12.2 min() and max()

```python
validation_scores = [0.81, 0.92, 0.87, 0.79]

print(min(validation_scores))
print(max(validation_scores))
```

**Expected output**

```text
0.79
0.92
```

**Explanation:** `min()` returns the smallest value and `max()` returns the largest value.

### 12.3 Combine len() and sum()

```python
monthly_sales = [10000, 12000, 15000]

average_sales = sum(monthly_sales) / len(monthly_sales)

print(average_sales)
```

**Expected output**

```text
12333.333333333334
```

**Explanation:** The list length gives the number of entries. This example demonstrates a basic average calculation; NumPy and Pandas provide more convenient functions for larger data.

### 12.4 Use round() for display

```python
monthly_sales = [10000, 12000, 15000]

average_sales = sum(monthly_sales) / len(monthly_sales)

print(round(average_sales, 2))
```

**Expected output**

```text
12333.33
```

**Explanation:** `round(value, 2)` rounds the displayed numeric result to two decimal places. Do not round data automatically without considering the task's precision requirements.

## 13. Iterating through a list

Iteration means visiting collection elements one at a time.

### 13.1 for loop over strings

```python
feature_names = ["age", "income", "tenure"]

for feature in feature_names:
    print(feature)
```

**Expected output**

```text
age
income
tenure
```

**Explanation:** The loop assigns each element to `feature` and executes the indented statement.

### 13.2 for loop over numeric values

```python
validation_scores = [0.81, 0.84, 0.89]

for score in validation_scores:
    print(score)
```

**Expected output**

```text
0.81
0.84
0.89
```

**Explanation:** Each score is processed in sequence.

### 13.3 Accumulate values

```python
batch_sizes = [32, 32, 16]

total = 0

for size in batch_sizes:
    total = total + size

print(total)
```

**Expected output**

```text
80
```

**Explanation:** The variable `total` starts at zero. Each iteration adds the current list value.

### 13.4 enumerate()

`enumerate()` provides an index and value together.

```python
feature_names = ["age", "income", "tenure"]

for index, feature in enumerate(feature_names):
    print(index, feature)
```

**Expected output**

```text
0 age
1 income
2 tenure
```

**Explanation:** The loop receives two values on every iteration: the position and the item.

### 13.5 Start enumeration from one

```python
feature_names = ["age", "income", "tenure"]

for index, feature in enumerate(feature_names, start=1):
    print(index, feature)
```

**Expected output**

```text
1 age
2 income
3 tenure
```

**Explanation:** The `start=1` argument changes the displayed index sequence. The list's actual indexes still begin at zero.

## 14. List comprehension

List comprehension creates a new list using a compact expression.

### 14.1 Traditional approach

```python
numbers = [1, 2, 3, 4]

squares = []

for number in numbers:
    squares.append(number ** 2)

print(squares)
```

**Expected output**

```text
[1, 4, 9, 16]
```

**Explanation:** A new list is created, and each squared value is appended.

### 14.2 List comprehension version

```python
numbers = [1, 2, 3, 4]

squares = [number ** 2 for number in numbers]

print(squares)
```

**Expected output**

```text
[1, 4, 9, 16]
```

**Explanation:** The expression before `for` describes the value to place into the new list.

### 14.3 Filter with a condition

```python
scores = [45, 67, 82, 39, 91]

passed_scores = [score for score in scores if score >= 50]

print(passed_scores)
```

**Expected output**

```text
[67, 82, 91]
```

**Explanation:** Only values that satisfy the condition are included.

### 14.4 Transform values

```python
prices = [100, 200, 300]

updated_prices = [round(price * 1.10, 2) for price in prices]

print(updated_prices)
```

**Expected output**

```text
[110.0, 220.0, 330.0]
```

**Explanation:** Each price is transformed and placed in a new list.

### 14.5 Use if-else inside comprehension

```python
scores = [45, 67, 82, 39]

labels = ["Pass" if score >= 50 else "Review" for score in scores]

print(labels)
```

**Expected output**

```text
['Review', 'Pass', 'Pass', 'Review']
```

**Explanation:** Every input produces an output label. This is a transformation, not a filter, because the output list keeps the same number of elements.

### 14.6 When not to use list comprehension

If the logic has many branches, side effects or debugging checkpoints, use a normal loop. Readability is more important than writing the shortest possible code.

## 15. List copying and references

### 15.1 Assignment does not create an independent copy

```python
original_features = ["age", "income"]

second_reference = original_features

second_reference.append("tenure")

print(original_features)
print(second_reference)
```

**Expected output**

```text
['age', 'income', 'tenure']
['age', 'income', 'tenure']
```

**Explanation:** Both names refer to the same list object. Updating through one name is visible through the other.

### 15.2 Use copy()

```python
original_features = ["age", "income"]

independent_features = original_features.copy()

independent_features.append("tenure")

print(original_features)
print(independent_features)
```

**Expected output**

```text
['age', 'income']
['age', 'income', 'tenure']
```

**Explanation:** The shallow copy creates a separate outer list. For nested mutable objects, a shallow copy does not recursively copy every inner object.

## 16. Python tuples

A tuple is an ordered collection that cannot be changed after it is created. Tuples are useful when a group of values should remain fixed.

### 16.1 Create a tuple

```python
dataset_metadata = ("customer_churn.csv", 5000, 12)

print(dataset_metadata)
print(type(dataset_metadata))
```

**Expected output**

```text
('customer_churn.csv', 5000, 12)
<class 'tuple'>
```

**Explanation:** Parentheses and comma-separated values create a tuple.

### 16.2 Access tuple elements

```python
dataset_metadata = ("customer_churn.csv", 5000, 12)

print(dataset_metadata[0])
print(dataset_metadata[1])
print(dataset_metadata[2])
```

**Expected output**

```text
customer_churn.csv
5000
12
```

**Explanation:** Tuples support indexing just like lists.

### 16.3 Negative indexing in tuples

```python
dataset_metadata = ("customer_churn.csv", 5000, 12)

print(dataset_metadata[-1])
```

**Expected output**

```text
12
```

**Explanation:** Negative indexes count from the end.

### 16.4 Slice a tuple

```python
model_metrics = ("accuracy", "precision", "recall", "f1")

print(model_metrics[1:3])
```

**Expected output**

```text
('precision', 'recall')
```

**Explanation:** Slicing returns a tuple containing the selected values.

### 16.5 A one-element tuple needs a comma

```python
single_value = ("churn",)

print(single_value)
print(type(single_value))
```

**Expected output**

```text
('churn',)
<class 'tuple'>
```

**Explanation:** The comma makes this a tuple. Parentheses alone do not always create a tuple.

### 16.6 Parentheses without a comma

```python
single_value = ("churn")

print(type(single_value))
```

**Expected output**

```text
<class 'str'>
```

**Explanation:** This is just a string inside parentheses.

### 16.7 Tuples cannot be updated

```python
model_metrics = ("accuracy", "precision", "recall")

model_metrics[0] = "auc"
```

**Expected result**

```text
TypeError: 'tuple' object does not support item assignment
```

**Explanation:** Tuples are immutable. Create a new tuple if a different collection is needed.

### 16.8 Tuple methods: count() and index()

```python
experiment_statuses = ("passed", "failed", "passed", "running")

print(experiment_statuses.count("passed"))
print(experiment_statuses.index("failed"))
```

**Expected output**

```text
2
1
```

**Explanation:** `count()` counts occurrences. `index()` returns the first matching position.

### 16.9 Tuple unpacking

```python
dataset_metadata = ("customer_churn.csv", 5000, 12)

file_name, row_count, column_count = dataset_metadata

print(file_name)
print(row_count)
print(column_count)
```

**Expected output**

```text
customer_churn.csv
5000
12
```

**Explanation:** Python assigns tuple values to names in the same order.

### 16.10 Unpacking count mismatch

```python
dataset_metadata = ("customer_churn.csv", 5000, 12)

file_name, row_count = dataset_metadata
```

**Expected result**

```text
ValueError: too many values to unpack
```

**Explanation:** Ordinary unpacking expects the number of target variables to match the number of values.

## 17. Python sets

A set stores unique elements. It is mutable, but it does not support positional indexing. Sets are useful for removing duplicates and comparing groups.

### 17.1 Create a set

```python
customer_ids = {"C001", "C002", "C003"}

print(customer_ids)
print(type(customer_ids))
```

**Expected output**

```text
{'C001', 'C002', 'C003'}
<class 'set'>
```

**Explanation:** Curly braces with comma-separated values create a set. The display order may vary.

### 17.2 Duplicate values are removed

```python
customer_ids = {"C001", "C002", "C001", "C003", "C002"}

print(customer_ids)
print(len(customer_ids))
```

**Expected output (order may vary)**

```text
{'C001', 'C002', 'C003'}
3
```

**Explanation:** A set keeps one occurrence of each unique value.

### 17.3 Create an empty set

```python
unique_values = set()

print(type(unique_values))
```

**Expected output**

```text
<class 'set'>
```

**Explanation:** Use `set()` to create an empty set. Empty curly braces `{}` create a dictionary instead.

### 17.4 Add an element

```python
feature_categories = {"customer", "billing"}

feature_categories.add("usage")

print(feature_categories)
```

**Expected output (order may vary)**

```text
{'customer', 'billing', 'usage'}
```

**Explanation:** `add()` inserts one element. Adding an element that already exists does not create a duplicate.

### 17.5 Update a set with multiple elements

```python
feature_categories = {"customer", "billing"}

feature_categories.update(["usage", "support"])

print(feature_categories)
```

**Expected output (order may vary)**

```text
{'customer', 'billing', 'usage', 'support'}
```

**Explanation:** `update()` adds each element from the supplied iterable.

### 17.6 Remove a known element

```python
feature_categories = {"customer", "billing", "usage"}

feature_categories.remove("billing")

print(feature_categories)
```

**Expected output (order may vary)**

```text
{'customer', 'usage'}
```

**Explanation:** `remove()` removes the requested element. It raises `KeyError` if the element does not exist.

### 17.7 Discard a possibly missing element

```python
feature_categories = {"customer", "usage"}

feature_categories.discard("billing")

print(feature_categories)
```

**Expected output (order may vary)**

```text
{'customer', 'usage'}
```

**Explanation:** `discard()` does not raise an error when the element is absent.

### 17.8 Pop an arbitrary set element

```python
feature_categories = {"customer", "billing", "usage"}

removed_value = feature_categories.pop()

print("Removed:", removed_value)
print("Remaining:", feature_categories)
```

**Expected output**

The removed value can vary because sets do not have a defined positional order.

**Explanation:** `pop()` removes and returns an arbitrary element. Do not use it when you need a specific item.

### 17.9 Clear a set

```python
temporary_ids = {"C001", "C002"}

temporary_ids.clear()

print(temporary_ids)
```

**Expected output**

```text
set()
```

**Explanation:** `clear()` removes all elements.

## 18. Set operations

Set operations compare groups of unique values.

Use these two sample groups:

```python
website_customers = {"C001", "C002", "C003", "C004"}
mobile_customers = {"C002", "C003", "C005"}
```

### 18.1 Union

Union returns all unique elements from both sets.

```python
website_customers = {"C001", "C002", "C003", "C004"}
mobile_customers = {"C002", "C003", "C005"}

all_customers = website_customers.union(mobile_customers)

print(all_customers)
```

**Expected output (order may vary)**

```text
{'C001', 'C002', 'C003', 'C004', 'C005'}
```

**Explanation:** Customers appearing in both sets are included only once.

### 18.2 Intersection

Intersection returns only the common elements.

```python
website_customers = {"C001", "C002", "C003", "C004"}
mobile_customers = {"C002", "C003", "C005"}

common_customers = website_customers.intersection(mobile_customers)

print(common_customers)
```

**Expected output (order may vary)**

```text
{'C002', 'C003'}
```

**Explanation:** These customer IDs occur in both groups.

### 18.3 Difference

Difference returns elements in the first set that are not in the second set.

```python
website_customers = {"C001", "C002", "C003", "C004"}
mobile_customers = {"C002", "C003", "C005"}

website_only = website_customers.difference(mobile_customers)

print(website_only)
```

**Expected output (order may vary)**

```text
{'C001', 'C004'}
```

**Explanation:** Difference is directional. Reversing the sets gives a different result.

### 18.4 Symmetric difference

Symmetric difference returns values that occur in one set or the other, but not both.

```python
website_customers = {"C001", "C002", "C003", "C004"}
mobile_customers = {"C002", "C003", "C005"}

one_platform_only = website_customers.symmetric_difference(mobile_customers)

print(one_platform_only)
```

**Expected output (order may vary)**

```text
{'C001', 'C004', 'C005'}
```

**Explanation:** The common IDs are excluded; only the IDs unique to one group remain.

### 18.5 Membership in a set

```python
known_customer_ids = {"C001", "C002", "C003"}

print("C002" in known_customer_ids)
print("C010" in known_customer_ids)
```

**Expected output**

```text
True
False
```

**Explanation:** Membership checks whether an element exists in the set.

### 18.6 Set subset

```python
all_features = {"age", "income", "tenure", "charges"}
required_features = {"age", "income"}

print(required_features.issubset(all_features))
```

**Expected output**

```text
True
```

**Explanation:** Every required feature is present in the larger feature collection.

### 18.7 Set superset

```python
all_features = {"age", "income", "tenure", "charges"}
required_features = {"age", "income"}

print(all_features.issuperset(required_features))
```

**Expected output**

```text
True
```

**Explanation:** The first set contains all elements of the second set.

### 18.8 Disjoint sets

```python
training_ids = {"C001", "C002"}
testing_ids = {"C003", "C004"}

print(training_ids.isdisjoint(testing_ids))
```

**Expected output**

```text
True
```

**Explanation:** The two sets have no common elements. This demonstrates a simple overlap check; a correct train/test split needs additional care and is covered on Day 13.

## 19. Python dictionaries

A dictionary stores information as key-value pairs. A key is used to look up its associated value.

### 19.1 Create a dictionary

```python
customer = {
    "customer_id": "C101",
    "age": 29,
    "membership": "Premium"
}

print(customer)
print(type(customer))
```

**Expected output**

```text
{'customer_id': 'C101', 'age': 29, 'membership': 'Premium'}
<class 'dict'>
```

**Explanation:** Each key maps to a value. A dictionary is useful for representing one structured record.

### 19.2 Read a value using its key

```python
customer = {
    "customer_id": "C101",
    "age": 29,
    "membership": "Premium"
}

print(customer["age"])
```

**Expected output**

```text
29
```

**Explanation:** The key `"age"` retrieves the associated value.

### 19.3 Access a missing key

```python
customer = {"customer_id": "C101", "age": 29}

print(customer["monthly_charges"])
```

**Expected result**

```text
KeyError: 'monthly_charges'
```

**Explanation:** Direct key access raises an error if the key is missing.

### 19.4 Use get() for safer lookup

```python
customer = {"customer_id": "C101", "age": 29}

print(customer.get("age"))
print(customer.get("monthly_charges"))
```

**Expected output**

```text
29
None
```

**Explanation:** `get()` returns `None` by default for a missing key rather than raising `KeyError`.

### 19.5 Supply a default value to get()

```python
customer = {"customer_id": "C101", "age": 29}

print(customer.get("monthly_charges", 0))
```

**Expected output**

```text
0
```

**Explanation:** The second argument is the default returned when the key is absent. Choose defaults carefully; zero may not mean the same thing as missing data.

### 19.6 Add a new key-value pair

```python
customer = {"customer_id": "C101", "age": 29}

customer["membership"] = "Premium"

print(customer)
```

**Expected output**

```text
{'customer_id': 'C101', 'age': 29, 'membership': 'Premium'}
```

**Explanation:** Assigning a value to a new key adds it to the dictionary.

### 19.7 Update an existing value

```python
customer = {"customer_id": "C101", "age": 29}

customer["age"] = 30

print(customer)
```

**Expected output**

```text
{'customer_id': 'C101', 'age': 30}
```

**Explanation:** Assigning to an existing key replaces its value.

### 19.8 Update multiple keys with update()

```python
customer = {"customer_id": "C101", "age": 29}

customer.update({
    "age": 30,
    "membership": "Premium"
})

print(customer)
```

**Expected output**

```text
{'customer_id': 'C101', 'age': 30, 'membership': 'Premium'}
```

**Explanation:** `update()` adds new keys and replaces values for keys that already exist.

### 19.9 Check whether a key exists

```python
customer = {"customer_id": "C101", "age": 29}

print("age" in customer)
print("salary" in customer)
```

**Expected output**

```text
True
False
```

**Explanation:** By default, `in` checks dictionary keys, not values.

### 19.10 Check dictionary values

```python
customer = {"customer_id": "C101", "membership": "Premium"}

print("Premium" in customer.values())
```

**Expected output**

```text
True
```

**Explanation:** Use `values()` when the question is whether a value exists.

## 20. Removing dictionary items

### 20.1 pop()

```python
customer = {
    "customer_id": "C101",
    "age": 29,
    "membership": "Premium"
}

removed_age = customer.pop("age")

print(removed_age)
print(customer)
```

**Expected output**

```text
29
{'customer_id': 'C101', 'membership': 'Premium'}
```

**Explanation:** `pop(key)` removes the key and returns its value.

### 20.2 pop() with a default

```python
customer = {"customer_id": "C101"}

removed_value = customer.pop("age", "Not provided")

print(removed_value)
print(customer)
```

**Expected output**

```text
Not provided
{'customer_id': 'C101'}
```

**Explanation:** A default prevents an error if the key does not exist.

### 20.3 popitem()

```python
customer = {
    "customer_id": "C101",
    "age": 29,
    "membership": "Premium"
}

last_item = customer.popitem()

print(last_item)
print(customer)
```

**Expected output**

```text
('membership', 'Premium')
{'customer_id': 'C101', 'age': 29}
```

**Explanation:** `popitem()` removes and returns the last inserted key-value pair in modern Python dictionaries.

### 20.4 del

```python
customer = {
    "customer_id": "C101",
    "age": 29,
    "membership": "Premium"
}

del customer["age"]

print(customer)
```

**Expected output**

```text
{'customer_id': 'C101', 'membership': 'Premium'}
```

**Explanation:** `del` removes the selected key and its value.

### 20.5 clear()

```python
temporary_record = {"id": "C101", "age": 29}

temporary_record.clear()

print(temporary_record)
```

**Expected output**

```text
{}
```

**Explanation:** The dictionary becomes empty.

## 21. Dictionary methods

### 21.1 keys()

```python
model_config = {
    "model": "RandomForest",
    "n_estimators": 100,
    "random_state": 42
}

print(list(model_config.keys()))
```

**Expected output**

```text
['model', 'n_estimators', 'random_state']
```

**Explanation:** `keys()` returns a view of dictionary keys. `list()` converts that view into a list for display.

### 21.2 values()

```python
model_config = {
    "model": "RandomForest",
    "n_estimators": 100,
    "random_state": 42
}

print(list(model_config.values()))
```

**Expected output**

```text
['RandomForest', 100, 42]
```

**Explanation:** `values()` returns the stored values.

### 21.3 items()

```python
model_config = {
    "model": "RandomForest",
    "n_estimators": 100,
    "random_state": 42
}

print(list(model_config.items()))
```

**Expected output**

```text
[('model', 'RandomForest'), ('n_estimators', 100), ('random_state', 42)]
```

**Explanation:** `items()` provides key-value pairs as tuples.

### 21.4 setdefault()

```python
model_config = {"model": "RandomForest"}

model_config.setdefault("random_state", 42)

print(model_config)
```

**Expected output**

```text
{'model': 'RandomForest', 'random_state': 42}
```

**Explanation:** If the key is absent, `setdefault()` inserts the supplied default. If it already exists, its current value is retained.

### 21.5 copy()

```python
original_config = {"model": "RandomForest", "random_state": 42}

new_config = original_config.copy()
new_config["model"] = "DecisionTree"

print(original_config)
print(new_config)
```

**Expected output**

```text
{'model': 'RandomForest', 'random_state': 42}
{'model': 'DecisionTree', 'random_state': 42}
```

**Explanation:** A shallow copy creates a separate outer dictionary. Nested mutable values may still be shared.

## 22. Iterating through dictionaries

### 22.1 Loop through keys

```python
model_config = {
    "model": "RandomForest",
    "n_estimators": 100,
    "random_state": 42
}

for key in model_config:
    print(key)
```

**Expected output**

```text
model
n_estimators
random_state
```

**Explanation:** A basic dictionary loop iterates over its keys.

### 22.2 Loop through values

```python
model_config = {
    "model": "RandomForest",
    "n_estimators": 100,
    "random_state": 42
}

for value in model_config.values():
    print(value)
```

**Expected output**

```text
RandomForest
100
42
```

**Explanation:** `values()` supplies each value.

### 22.3 Loop through key-value pairs

```python
model_config = {
    "model": "RandomForest",
    "n_estimators": 100,
    "random_state": 42
}

for key, value in model_config.items():
    print(key, ":", value)
```

**Expected output**

```text
model : RandomForest
n_estimators : 100
random_state : 42
```

**Explanation:** Each dictionary item is unpacked into a key and value.

## 23. Nested data structures

A nested structure contains one collection inside another. It can represent records, experiment settings or grouped information.

### 23.1 List of dictionaries

```python
customers = [
    {"customer_id": "C001", "age": 25, "segment": "Regular"},
    {"customer_id": "C002", "age": 34, "segment": "Premium"},
    {"customer_id": "C003", "age": 29, "segment": "Regular"}
]

print(customers[0])
```

**Expected output**

```text
{'customer_id': 'C001', 'age': 25, 'segment': 'Regular'}
```

**Explanation:** The outer list stores multiple records. Each record is a dictionary with named fields.

### 23.2 Access a field in a nested record

```python
customers = [
    {"customer_id": "C001", "age": 25, "segment": "Regular"},
    {"customer_id": "C002", "age": 34, "segment": "Premium"}
]

print(customers[1]["segment"])
```

**Expected output**

```text
Premium
```

**Explanation:** First select the second dictionary using index one, then retrieve its `segment` key.

### 23.3 Update a nested record

```python
customers = [
    {"customer_id": "C001", "age": 25, "segment": "Regular"},
    {"customer_id": "C002", "age": 34, "segment": "Premium"}
]

customers[0]["segment"] = "New"

print(customers[0])
```

**Expected output**

```text
{'customer_id': 'C001', 'age': 25, 'segment': 'New'}
```

**Explanation:** The first dictionary is updated through its key.

### 23.4 Loop through a list of dictionaries

```python
customers = [
    {"customer_id": "C001", "age": 25, "segment": "Regular"},
    {"customer_id": "C002", "age": 34, "segment": "Premium"}
]

for customer in customers:
    print(customer["customer_id"], customer["segment"])
```

**Expected output**

```text
C001 Regular
C002 Premium
```

**Explanation:** The loop visits one dictionary at a time.

### 23.5 Dictionary containing lists

```python
experiment_results = {
    "model_names": ["Logistic Regression", "Decision Tree"],
    "scores": [0.84, 0.88]
}

print(experiment_results["model_names"])
print(experiment_results["scores"])
```

**Expected output**

```text
['Logistic Regression', 'Decision Tree']
[0.84, 0.88]
```

**Explanation:** A dictionary can organize related lists under meaningful keys.

## 24. Comprehensions for collections

### 24.1 List comprehension recap

```python
feature_names = ["age", "income", "tenure"]

uppercase_features = [feature.upper() for feature in feature_names]

print(uppercase_features)
```

**Expected output**

```text
['AGE', 'INCOME', 'TENURE']
```

**Explanation:** Each string is transformed and collected in a new list.

### 24.2 Set comprehension

```python
raw_categories = ["A", "B", "A", "C", "B"]

unique_categories = {category for category in raw_categories}

print(unique_categories)
```

**Expected output (order may vary)**

```text
{'A', 'B', 'C'}
```

**Explanation:** A set comprehension creates a set, so duplicate values are removed.

### 24.3 Dictionary comprehension

```python
feature_names = ["age", "income", "tenure"]

feature_lengths = {
    feature: len(feature)
    for feature in feature_names
}

print(feature_lengths)
```

**Expected output**

```text
{'age': 3, 'income': 6, 'tenure': 6}
```

**Explanation:** Each feature name becomes a dictionary key, and its string length becomes the value.

### 24.4 Dictionary comprehension with a condition

```python
scores = {"model_a": 0.82, "model_b": 0.91, "model_c": 0.78}

high_scores = {
    name: score
    for name, score in scores.items()
    if score >= 0.80
}

print(high_scores)
```

**Expected output**

```text
{'model_a': 0.82, 'model_b': 0.91}
```

**Explanation:** Only key-value pairs with a score of at least 0.80 are included.

## 25. Choosing the right structure

| Requirement | Suitable structure | Reason |
|---|---|---|
| Keep an ordered list of feature names | List | Order and modification are useful |
| Store fixed dataset metadata fields together | Tuple | Ordered values that should remain unchanged |
| Remove duplicate customer IDs | Set | Stores unique values |
| Look up a model setting by name | Dictionary | Key-value lookup |
| Store many simple customer records | List of dictionaries | Each record has named fields |
| Store experiment settings | Dictionary | Setting names map to values |

Python lists, tuples, sets and dictionaries are general-purpose containers. For large numerical arrays, use NumPy; for labelled tabular data, use Pandas. Those tools are covered in the next lessons.

## 26. Common mistakes

| Mistake | What happens | Better approach |
|---|---|---|
| Accessing a list with an invalid index | `IndexError` | Check list length and valid positions |
| Confusing append and extend | Nested list instead of separate elements | Choose based on whether you add one object or each item |
| Removing a missing value | `ValueError` | Check membership or handle the exception |
| Assuming reverse sorts values | Only reverses current order | Use `sort()` or `sorted()` for sorting |
| Updating a tuple | `TypeError` | Create a new tuple |
| Expecting a set to keep insertion order | Order is not guaranteed | Do not depend on set display order |
| Indexing a set | `TypeError` | Use membership or iteration |
| Using `{}` for an empty set | Creates a dictionary | Use `set()` |
| Accessing a missing dictionary key | `KeyError` | Use `get()` or check key membership |
| Assuming assignment copies a collection | Two names may refer to the same object | Use `copy()` when an independent shallow copy is needed |

## 27. Practical ML-oriented examples

### 27.1 Maintain a feature-name list

```python
features = ["age", "monthly_charges"]

features.append("tenure")
features.append("contract_type")

print(features)
```

**Expected output**

```text
['age', 'monthly_charges', 'tenure', 'contract_type']
```

**Explanation:** A list is useful for maintaining an ordered collection of feature names.

### 27.2 Keep a fixed set of evaluation labels

```python
metric_names = ("accuracy", "precision", "recall", "f1")

print(metric_names)
```

**Expected output**

```text
('accuracy', 'precision', 'recall', 'f1')
```

**Explanation:** A tuple can represent a fixed ordered group of labels.

### 27.3 Find unique categories

```python
contract_types = ["Monthly", "Annual", "Monthly", "Two Year", "Annual"]

unique_contracts = set(contract_types)

print(unique_contracts)
```

**Expected output (order may vary)**

```text
{'Monthly', 'Annual', 'Two Year'}
```

**Explanation:** A set gives the unique category values. Pandas will provide convenient unique-value operations on real columns later.

### 27.4 Store model configuration

```python
model_config = {
    "model_name": "RandomForestClassifier",
    "n_estimators": 100,
    "random_state": 42
}

print(model_config["model_name"])
print(model_config["n_estimators"])
```

**Expected output**

```text
RandomForestClassifier
100
```

**Explanation:** A dictionary makes each configuration value accessible by a meaningful key.

### 27.5 Store experiment records

```python
experiments = [
    {"name": "baseline", "score": 0.81},
    {"name": "decision_tree", "score": 0.86},
    {"name": "random_forest", "score": 0.90}
]

for experiment in experiments:
    print(experiment["name"], experiment["score"])
```

**Expected output**

```text
baseline 0.81
decision_tree 0.86
random_forest 0.9
```

**Explanation:** This is a small in-memory representation of experiment results. It is for learning data structures, not a replacement for a proper experiment-tracking system.

## 28. Revision questions

1. Why would you choose a list instead of separate variables?
2. What is the difference between positive and negative indexing?
3. Why is the end index excluded in a slice?
4. What is the difference between `append()` and `extend()`?
5. How are `remove()` and `pop()` different?
6. What is the difference between `sort()` and `reverse()`?
7. Why does a one-element tuple need a comma?
8. Why can a tuple not be updated?
9. What problem does a set solve when data contains duplicates?
10. Why is a set not suitable when positional order is required?
11. What is a dictionary key?
12. How does `get()` differ from direct dictionary indexing?
13. What is the difference between `keys()`, `values()` and `items()`?
14. What does it mean when two variables refer to the same list?
15. When would a list of dictionaries be useful?

## 29. Interview-ready explanations

**What is the difference between a list and a tuple?**  
Both are ordered collections and support indexing. A list is mutable, so elements can be changed. A tuple is immutable, so its elements cannot be reassigned after creation. Use a list for a collection that changes and a tuple for a fixed group of values.

**When would you use a set in data work?**  
Use a set when unique values or comparisons between groups are required. For example, it can identify unique customer IDs or compare IDs present in two systems. Do not use it when the original order or duplicates must be preserved.

**Why use a dictionary?**  
A dictionary maps keys to values. It is useful for named settings, structured records and quick lookup by an identifier. For example, a model configuration can map `"n_estimators"` to `100`.

**What is the difference between append() and extend()?**  
`append()` adds its argument as one element. `extend()` iterates over the supplied iterable and adds its elements individually.

**Why can two variables show the same list after one is modified?**  
Assignment such as `second = first` makes both names refer to the same list object. Use `copy()` to create a separate shallow copy when that is the intended behaviour.

## 30. Lesson completion checklist

- [ ] I can create a list and access elements using indexes.
- [ ] I can extract values using slicing.
- [ ] I can add, update, remove and search list elements.
- [ ] I understand the difference between append and extend.
- [ ] I can loop through a list and use enumerate.
- [ ] I can write a simple list comprehension.
- [ ] I can create and unpack a tuple.
- [ ] I understand tuple immutability.
- [ ] I can create a set and perform union/intersection/difference.
- [ ] I can create a dictionary and safely retrieve values.
- [ ] I can update, remove and iterate through dictionary items.
- [ ] I can select an appropriate data structure for a small ML task.

**Next lesson:** Day 05 — NumPy Arrays, Indexing and Slicing. We will move from general Python collections to efficient numerical arrays used in data and machine learning workflows.
