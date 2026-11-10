# Day 04: Interview Questions — Lists, Dictionaries, Tuples and Sets

**Pathway:** Machine Learning Engineer  
**Focus:** Python collections, data organization, safe lookup and practical ML scenarios

## 1. Core concept questions

### Q1. Why do we need data structures?

**Answer:** Data structures organize multiple related values so a program can access and process them efficiently. Instead of creating a separate variable for every customer or feature, we can use a list, tuple, set or dictionary.

**Follow-up:** Which structure would you use for a collection of feature names that may change during preprocessing?

**Answer:** A list is suitable because it preserves order and can be updated.

### Q2. What are the main differences between a list, tuple, set and dictionary?

**Answer:**

- A list is ordered, mutable and allows duplicates.
- A tuple is ordered, immutable and allows duplicates.
- A set stores unique values and does not support positional indexing.
- A dictionary stores key-value pairs and uses keys for lookup.

**Follow-up:** Which one is best for every task?

**Answer:** None. Choose based on whether order, mutation, uniqueness or key-based lookup is required.

### Q3. What is the difference between mutable and immutable?

**Answer:** A mutable object can be changed after creation. A list is mutable. An immutable object cannot have its elements reassigned after creation; a tuple is immutable.

**Follow-up:** Are strings mutable?

**Answer:** No. Strings are immutable. Operations that appear to change a string create a new string value.

## 2. List questions

### Q4. What is list indexing?

**Answer:** Indexing accesses an element by its position. Python uses zero-based indexing, so the first element has index zero. Negative indexes count from the end.

**Follow-up:** What error occurs when the requested index does not exist?

**Answer:** Python raises `IndexError`.

### Q5. What is list slicing?

**Answer:** Slicing extracts a portion of a list using start, end and optionally step values. The start position is included and the end position is excluded.

**Follow-up:** What does `values[::-1]` do?

**Answer:** It creates a reversed sequence using a negative step.

### Q6. What is the difference between append() and extend()?

**Answer:** `append()` adds its argument as one element. `extend()` adds the individual elements from an iterable.

**Follow-up:** What happens if you append a list to another list?

**Answer:** The complete inner list becomes one nested element.

### Q7. What is the difference between remove() and pop()?

**Answer:** `remove(value)` searches for and removes the first matching value. `pop(index)` removes and returns an element by position; without an index, it removes the last element.

**Follow-up:** Which method is more useful when you need the removed value?

**Answer:** `pop()`, because it returns the removed element.

### Q8. What is the difference between sort() and sorted()?

**Answer:** `list.sort()` sorts the original list in place. `sorted(iterable)` returns a new sorted list and leaves the original iterable unchanged.

**Follow-up:** Does reverse() sort a list?

**Answer:** No. It reverses the current order without comparing or sorting the values.

### Q9. What is list comprehension?

**Answer:** List comprehension is a concise way to create a new list from an iterable, optionally transforming or filtering values.

**Follow-up:** When should you prefer a normal loop?

**Answer:** When the logic is complex, has multiple branches, needs side effects or benefits from intermediate debugging steps.

## 3. Tuple questions

### Q10. Why would you use a tuple instead of a list?

**Answer:** Use a tuple when the values form an ordered group that should not be modified. For example, a fixed group of dataset metadata fields can be represented as a tuple.

**Follow-up:** Can a tuple contain a list?

**Answer:** Yes. The tuple's references cannot be reassigned, but a mutable object stored inside it may itself be modified. Tuple immutability does not recursively freeze nested objects.

### Q11. Why does a one-element tuple need a comma?

**Answer:** The comma identifies the tuple. Parentheses alone are also used for grouping expressions. For example, `(10,)` is a tuple, while `(10)` is an integer.

**Follow-up:** Can tuples be unpacked into variables?

**Answer:** Yes, provided the number of values matches the target variables, unless extended unpacking syntax is used.

## 4. Set questions

### Q12. Why are sets useful in data processing?

**Answer:** Sets store unique values and support membership checks and group comparisons. They can help identify unique IDs or compare customer IDs received from two sources.

**Follow-up:** What information is lost when converting a list with duplicates to a set?

**Answer:** Duplicate occurrences and the original positional order are not preserved.

### Q13. What is the difference between union and intersection?

**Answer:** Union returns all unique values from both sets. Intersection returns only the values common to both sets.

**Follow-up:** Which operation identifies values present in one source but not another?

**Answer:** Difference, with attention to which set is placed first.

### Q14. What is the difference between remove() and discard() for sets?

**Answer:** Both remove an element. `remove()` raises `KeyError` if the element is absent, while `discard()` does not raise an error for a missing element.

**Follow-up:** Can you access a set element using index zero?

**Answer:** No. Sets do not support positional indexing.

## 5. Dictionary questions

### Q15. What is a dictionary?

**Answer:** A dictionary stores key-value pairs. The key provides a meaningful way to retrieve the associated value. It is useful for model configuration, structured records and lookup tables.

**Follow-up:** Can dictionary keys repeat?

**Answer:** Keys are unique. If the same key is assigned again, the later value replaces the earlier value.

### Q16. What is the difference between dictionary indexing and get()?

**Answer:** Direct indexing such as `record["age"]` raises `KeyError` if the key is absent. `record.get("age")` returns `None` by default or a supplied default value.

**Follow-up:** Is a default of zero always appropriate for a missing numeric field?

**Answer:** No. Zero may be a meaningful measurement. Choose a default only when the business and data definitions justify it.

### Q17. Explain keys(), values() and items().

**Answer:** `keys()` provides the dictionary keys, `values()` provides its values, and `items()` provides key-value pairs.

**Follow-up:** Which method would you use to loop through both a model setting name and its value?

**Answer:** `items()`.

### Q18. What is a nested dictionary?

**Answer:** A nested dictionary is a dictionary that contains other dictionaries or collections. It can represent multiple structured records or grouped configuration.

**Follow-up:** What is the risk of deeply nested structures?

**Answer:** Access and validation become harder to read and maintain. For tabular datasets, use a DataFrame or another suitable schema-based structure.

## 6. Machine-learning and data scenarios

### Q19. You receive a list of feature names and need to add a new feature while preserving order. Which structure would you choose?

**Answer:** A list. It preserves order and supports adding new values with methods such as `append()` or `extend()`.

**Follow-up:** How would you check whether a feature name is already present?

**Answer:** Use the `in` operator before adding it, if duplicates are not desired.

### Q20. Two data sources provide customer IDs. You need to identify IDs present in both sources. How would you approach this?

**Answer:** If the task is to compare unique IDs, convert the collections to sets and use intersection. Before doing so, normalize the ID format consistently and decide how to handle missing or malformed IDs.

**Follow-up:** Why might converting to a set be inappropriate?

**Answer:** If duplicate occurrences, ordering or counts matter, a set removes information that the task may require.

### Q21. You need to store a model name, number of estimators and random state. Which structure is suitable?

**Answer:** A dictionary is suitable because each setting has a meaningful key. For example, `{"model": "RandomForest", "n_estimators": 100, "random_state": 42}`.

**Follow-up:** How can you safely read an optional setting?

**Answer:** Use `get()` with a default that is documented and valid for the model or configuration.

### Q22. A list of customer records contains dictionaries. How would you read each customer's age?

**Answer:** Iterate over the outer list. Each item is a dictionary, so access the age by its key. Validate that the key exists and that the value is of an expected type before using it.

**Follow-up:** What should happen if one record does not contain age?

**Answer:** Follow the data-quality rule: record or report the missing field, and apply the agreed missing-data treatment. Do not silently substitute a value without justification.

### Q23. A developer writes `second = first` and then modifies second. Why does first also appear to change?

**Answer:** Both names refer to the same list object. The assignment did not create an independent copy. Use `first.copy()` for a shallow copy when an independent outer list is required.

**Follow-up:** Is a shallow copy enough for a list containing nested lists?

**Answer:** Not if independent nested mutable objects are also required. A shallow copy copies only the outer collection; a deep copy may be needed depending on the intended behaviour.

### Q24. You need to preserve the order of repeated event labels and count how many times each label occurs. Would a set be enough?

**Answer:** No. A set removes duplicates, so it cannot preserve occurrence counts. Use a list if order and every event matter, and use a dictionary or a counting tool to store frequencies.

**Follow-up:** Which dictionary pattern can update a count?

**Answer:** Check whether the key exists, then increment its value, or use a suitable standard-library counting utility.

### Q25. Your team stores a whole tabular dataset as a nested list. Is this the best choice for ML data preparation?

**Answer:** It may be acceptable for a small teaching example, but it is usually not the best tool for real tabular work. Pandas DataFrames provide labelled columns, data types, filtering, missing-value handling and integration with ML libraries. NumPy arrays are appropriate for numerical array operations.

**Follow-up:** Why is the structure choice important?

**Answer:** It affects clarity, validation, supported operations and the effort needed to process data safely.

## 7. Short code interpretation questions

### Q26. What is the output?

```python
values = [10, 20, 30, 40]
print(values[1:3])
```

**Answer:** `[20, 30]`. The slice starts at index one and stops before index three.

**Follow-up:** What does `values[-1]` return?

**Answer:** 40, the last element.

### Q27. What is the output?

```python
values = [1, 2]
values.append([3, 4])
print(values)
```

**Answer:** `[1, 2, [3, 4]]`. The nested list is appended as one element.

**Follow-up:** What would extend() produce?

**Answer:** `[1, 2, 3, 4]`.

### Q28. What is the output?

```python
ids = ["A", "B", "A", "C"]
print(set(ids))
```

**Answer:** A set containing `A`, `B` and `C`. The display order may vary.

**Follow-up:** Does this preserve how many times A occurred?

**Answer:** No. The duplicate occurrence is removed.

### Q29. What is the output?

```python
record = {"age": 30}
print(record.get("salary", 0))
```

**Answer:** 0, because the salary key is absent and zero was supplied as the default.

**Follow-up:** Why should a developer be careful with this default?

**Answer:** A missing salary and a true salary of zero are different meanings. The default must be appropriate for the data definition.

### Q30. What is the output?

```python
left = {"A", "B", "C"}
right = {"B", "C", "D"}
print(left.intersection(right))
```

**Answer:** A set containing B and C.

**Follow-up:** Which operation returns all unique values from both sets?

**Answer:** Union.

## 8. Interview answer framework

When asked to choose or debug a Python collection:

1. **Identify the requirement:** Is order important? Are duplicates meaningful? Will values change? Is key-based lookup required?
2. **Choose the structure:** List, tuple, set or dictionary.
3. **Explain the trade-off:** Mention what information the structure preserves or discards.
4. **Validate the data:** Check missing keys, invalid positions, duplicates and value types.
5. **Test edge cases:** Empty collections, repeated values, missing keys and unexpected types.
6. **Scale appropriately:** Move to NumPy or Pandas for numerical arrays and tabular datasets.

**Senior developer principle:** Do not choose a collection just because its syntax is familiar. Choose the structure that preserves the information the task needs and makes the next operation safe and clear.
