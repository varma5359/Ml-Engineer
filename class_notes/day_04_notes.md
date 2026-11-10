# Day 04: Lists, Dictionaries, Tuples and Sets

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define Python lists, dictionaries, tuples and sets.
- Create, access, update and remove collection elements.
- Understand indexing, slicing, keys and membership checks.
- Compare ordering, mutability and uniqueness.
- Select a suitable collection for structured data.

## 1. Why Do We Need Collections?
A variable can refer to one value. Applications often need to store many values such as customer names, transaction amounts, feature values or model metrics. Collections store multiple related values.

## 2. Lists
**Definition:** A list is an ordered, mutable collection. It allows duplicate values and supports indexing.

**Syntax:** `list_name = [value1, value2, value3]`

**Example 1 — Create and access**
```python
sales = [1200, 1500, 900, 1200]
print(sales[0])
print(sales[-1])
print(len(sales))
```
Output: 1200, 1200, 4.

**Example 2 — Add and remove**
```python
sales = [1000, 1500]
sales.append(2000)
sales.insert(1, 1200)
sales.remove(1500)
print(sales)
```
Output: [1000, 1200, 2000].

**Example 3 — Slice and update**
```python
scores = [70, 80, 90, 95]
print(scores[1:3])
scores[0] = 75
print(scores)
```
Output: [80, 90] and [75, 80, 90, 95].

Useful methods: append(), extend(), insert(), remove(), pop(), sort(), reverse(), clear(). `sort()` changes the list in place; `sorted()` returns a new sorted list.

## 3. Tuples
**Definition:** A tuple is an ordered collection that cannot be changed after creation (immutable). It allows duplicates and supports indexing.

**Syntax:** `point = (x, y)`

**Example 1 — Create and access**
```python
location = (17.4, 78.4)
print(location[0])
print(location[1])
```
Output: 17.4 and 78.4.

**Example 2 — Unpacking**
```python
customer = ("Asha", 28, "Hyderabad")
name, age, city = customer
print(name, age, city)
```
Output: Asha 28 Hyderabad.

**Example 3 — Single-item tuple**
```python
status = ("active",)
print(type(status))
```
Output: <class 'tuple'>. The comma makes it a tuple.

Use tuples for fixed grouped values, such as coordinates or fixed configuration. Immutability applies to the tuple's references; a mutable object stored inside a tuple can still be modified.

## 4. Dictionaries
**Definition:** A dictionary stores data as key-value pairs. Keys are unique and are used to retrieve values.

**Syntax:** `record = {"key": value}`

**Example 1 — Create and access**
```python
customer = {"name": "Ravi", "age": 30, "city": "Hyderabad"}
print(customer["name"])
print(customer.get("age"))
```
Output: Ravi and 30.

**Example 2 — Add and update**
```python
customer = {"name": "Asha", "orders": 3}
customer["orders"] = 4
customer["active"] = True
print(customer)
```
Output: {'name': 'Asha', 'orders': 4, 'active': True}.

**Example 3 — Safe lookup and iteration**
```python
customer = {"name": "Asha", "orders": 4}
print(customer.get("email", "Not provided"))
for key, value in customer.items():
    print(key, value)
```
Output begins with Not provided, followed by each key and value.

Useful methods: get(), keys(), values(), items(), update(), pop(), setdefault().

## 5. Sets
**Definition:** A set is a collection of unique elements. It removes duplicates and supports fast membership checks. Sets are not indexed by position.

**Syntax:** `unique_values = {value1, value2}`; an empty set is `set()`, not `{}`.

**Example 1 — Remove duplicates**
```python
cities = {"Hyd", "Vizag", "Hyd"}
print(cities)
```
Output contains only Hyd and Vizag; display order is not guaranteed.

**Example 2 — Add and discard**
```python
skills = {"Python", "NumPy"}
skills.add("Pandas")
skills.discard("NumPy")
print(skills)
```
Output contains Python and Pandas.

**Example 3 — Set operations**
```python
a = {"Python", "SQL", "ML"}
b = {"Python", "AWS"}
print(a.intersection(b))
print(a.union(b))
print(a.difference(b))
```
Intersection: Python; union contains Python, SQL, ML, AWS; difference: SQL and ML. Set display order is not guaranteed.

## 6. Comparison
| Feature | List | Tuple | Dictionary | Set |
|---|---|---|---|---|
| Ordered | Yes | Yes | Insertion order | No positional order |
| Mutable | Yes | No | Yes | Yes |
| Duplicates | Yes | Yes | Keys unique | No duplicate elements |
| Access | Index | Index | Key | Membership/iteration |

## 7. Real-World Choice
- Recent transactions in order: list.
- Fixed coordinate pair: tuple.
- Customer record with named fields: dictionary.
- Unique product IDs: set.

**Scenario questions**
1. A customer buys the same product twice. Which collection preserves both purchases? A list.
2. You need to retrieve a customer's email using the field name. Which collection? A dictionary.
3. You want only distinct customer IDs. Which collection? A set.
