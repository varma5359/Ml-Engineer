# Day 04: Interview Questions — Lists, Dictionaries, Tuples and Sets

**Pathway:** Machine Learning Engineer

---

## Question 1: What is the difference between a list and a tuple?

**Answer:** Both are ordered and allow duplicates. A list is mutable; a tuple is immutable.

**Example:** Use a list for customer transactions that may be updated and a tuple for a fixed image shape such as `(224, 224, 3)`.

**Follow-up:** Can a tuple contain a list?

**Answer:** Yes, but the tuple's reference to that list cannot be reassigned. The list object itself can still be mutated.

## Question 2: When would you use a dictionary?

**Answer:** Use a dictionary when values should be accessed by meaningful keys rather than numeric positions.

**Example:** A customer record can store `{"customer_id": "C10", "age": 28}`.

**Follow-up:** What is the difference between `data[key]` and `data.get(key)`?

**Answer:** Square-bracket access raises `KeyError` when the key is absent. `get()` can return a default value instead.

## Question 3: Why would you use a set in data preparation?

**Answer:** A set stores unique elements and supports fast membership checks and union/intersection/difference operations.

**Example:** Find unique product categories or compare IDs between two data extracts.

**Follow-up:** Should you use a set to preserve transaction order?

**Answer:** No. Use a list or an ordered tabular structure when order matters.

## Question 4: How can you represent many customer records using Python collections?

**Answer:** A common simple structure is a list of dictionaries. Each dictionary represents one customer and each key identifies a field.

**Follow-up:** Is this always ideal for a million-row dataset?

**Answer:** No. Pandas or a database is usually better for large tabular workloads, filtering, grouping and vectorized operations.

## Question 5: Your code accesses `customer["age"]` and raises `KeyError`. What does it indicate?

**Answer:** The dictionary does not contain the key `"age"` at that moment. Check the schema, spelling, source record and whether the field is optional. Use `get()` only when a missing key is an expected condition and handle it explicitly.

**Follow-up:** Would `customer.get("age", 0)` always be a good fix?

**Answer:** No. Zero may be a valid age and can hide a schema defect. A suitable missing value or explicit validation error is often better.

## Question 6: How do you remove duplicates from a list?

**Answer:** A set can remove duplicates when order is not important: `list(set(values))`. To preserve first-seen order, use `list(dict.fromkeys(values))` for hashable values.

**Follow-up:** Should duplicate rows always be removed from a business dataset?

**Answer:** No. First determine the business key and whether the rows represent true duplicate records or repeated legitimate events.

## Question 7: Which collection would you choose for model feature names that should not change?

**Answer:** A tuple can be appropriate because it is ordered and immutable. A list may be used when the names need to be updated during preprocessing.

**Follow-up:** Why does feature order matter?

**Answer:** A model expects input columns in the same order and meaning used during training. Incorrect order can send values into the wrong model inputs.

## Question 8: What collection design would you use for counting orders by status?

**Answer:** A dictionary can map each status to its count. For example, `{"delivered": 20, "pending": 5}`. `collections.Counter` is also designed for counting hashable values.

**Follow-up:** What happens when a new status appears?

**Answer:** The counting logic should support new keys safely, while the application should validate whether the status is allowed by the business schema.