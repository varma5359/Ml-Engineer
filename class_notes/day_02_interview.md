# Day 02: Interview Questions — Python Environment, Variables, Data Types and Operators

**Pathway:** Machine Learning Engineer  
**Focus:** Practical Python fundamentals, data validation and ML workflow scenarios

## 1. Core concept questions

### Q1. What is a Python environment?

**Answer:** A Python environment is the setup used to execute Python code. It includes the Python interpreter and the installed packages required by a project. A virtual environment isolates project dependencies so one project's package versions do not unintentionally affect another.

**Example:** A churn-prediction project and an image-classification project may require different package versions. Separate environments help keep their dependencies independent.

**Follow-up:** Does creating a virtual environment install every package the project needs?

**Answer:** No. It creates an isolated environment. Required packages must still be installed in that environment.

### Q2. What is a variable?

**Answer:** A variable is a name that refers to a value or object. Variables let us store and reuse information without repeating literal values throughout a program.

**Example:** `target_column = "churn"` stores the name of the target column used by a project.

**Follow-up:** Is a variable a permanent box in memory?

**Answer:** That is a useful beginner analogy, but more precisely a Python name refers to an object. Reassignment makes the name refer to another object.

### Q3. What is the difference between int, float, str and bool?

**Answer:**
- `int` represents whole numbers, such as a row count.
- `float` represents numeric values with decimal components, such as a price.
- `str` represents text, such as a column name.
- `bool` represents `True` or `False`, such as a validation flag.

**Follow-up:** If a value looks like 500 but is stored as `"500"`, what is its type?

**Answer:** It is a string because it is enclosed in quotation marks.

### Q4. What does None represent?

**Answer:** `None` is a special Python value that represents the absence of a value. It is different from zero and from an empty string.

**Follow-up:** How should you check whether a variable is None?

**Answer:** Use `value is None` or `value is not None`. Do not use equality as a substitute for the conventional identity check.

### Q5. What is the difference between = and ==?

**Answer:** `=` assigns a value to a name. `==` compares two values and returns a Boolean result.

**Example:** `score = 0.90` assigns a score. `score == 0.90` checks whether it equals that value.

**Follow-up:** What does != do?

**Answer:** It checks whether two values are not equal.

### Q6. What is dynamic typing?

**Answer:** Python is dynamically typed. We do not declare a variable's type before assigning a value. A name can later refer to an object of a different type.

**Example:** A name can first refer to an integer and later to a string. Although Python permits this, reusing a variable for unrelated purposes can make code confusing.

**Follow-up:** Does dynamic typing mean Python values have no types?

**Answer:** No. Objects still have types; Python determines and checks them at runtime.

## 2. Practical Python and debugging questions

### Q7. What is the difference between type() and isinstance()?

**Answer:** `type(value)` reports the object's type. `isinstance(value, expected_type)` checks whether an object is an instance of a specified type or an allowed type tuple.

**Example:**
```python
value = 12.5
print(type(value))
print(isinstance(value, (int, float)))
```

Expected result: the type is `float`, and the validation result is `True`.

**Follow-up:** Why might isinstance() be more useful for input validation?

**Answer:** It directly answers whether a value belongs to one of the accepted types.

### Q8. What is type conversion?

**Answer:** Type conversion changes a value from one data type to another. Examples include `int("25")`, `float("25.5")` and `str(25)`.

**Follow-up:** Does int(99.99) round the value?

**Answer:** No. It removes the fractional part and returns 99.

### Q9. What is the difference between implicit and explicit conversion?

**Answer:** Implicit conversion is performed automatically by Python in compatible operations, such as combining an integer and a float. Explicit conversion is requested by the programmer using functions such as `int()` or `float()`.

**Follow-up:** Should you convert every text column to a number?

**Answer:** No. First understand what the column means. A digit-only customer ID may be an identifier, not a quantity.

### Q10. What happens when int("abc") is executed?

**Answer:** Python raises `ValueError` because the text cannot be interpreted as an integer.

**Follow-up:** How should an application handle this?

**Answer:** Validate the input and handle the conversion error, for example with `try/except ValueError`. The program should provide a useful message or follow a documented data-cleaning rule.

### Q11. What is the difference between / and //?

**Answer:** `/` performs true division. `//` performs floor division and rounds the quotient down to the next integer value for positive inputs.

**Example:** `10 / 3` returns approximately 3.3333, while `10 // 3` returns 3.

**Follow-up:** What does % return?

**Answer:** The remainder after division. For example, `10 % 3` returns 1.

### Q12. Explain and, or and not.

**Answer:** `and` requires both conditions to be true; `or` requires at least one condition to be true; `not` reverses a Boolean result.

**Example:** A data file may be considered ready only when `file_exists and required_columns_present` are both true.

**Follow-up:** What is short-circuit evaluation?

**Answer:** Python may stop evaluating a logical expression when the result is already determined. For example, a false first condition in an `and` expression means the complete expression is false.

### Q13. What is operator precedence?

**Answer:** Operator precedence determines which operation is evaluated first in an expression. Parentheses can make the intended grouping explicit and improve readability.

**Follow-up:** Why is a long expression sometimes harder to maintain?

**Answer:** The reader must mentally track precedence and grouping. Named intermediate variables or parentheses can make the logic clearer.

## 3. Machine-learning data scenarios

### Q14. A customer dataset stores age values as strings. Can you train a model directly?

**Answer:** Not safely without inspecting the column. If age is genuinely numeric, validate the text values and convert valid values to a numeric type. Handle invalid and missing entries according to the data definition. If the column is an identifier or category represented by digits, numeric conversion may be inappropriate.

**Follow-up:** What if some rows contain "unknown"?

**Answer:** Investigate how often it occurs and why. Treat it as invalid or missing according to the agreed data rules. Do not automatically convert it to zero because zero has a different meaning.

### Q15. A CSV file contains the values "100", "200" and "unknown" in a price column. What would you do?

**Answer:** I would inspect the column's distinct values and confirm that it is meant to represent price. I would identify valid numeric strings, identify the special text value, decide whether it represents missing or invalid data, and then convert the valid values. I would report how many rows were affected and validate the result.

**Follow-up:** Why not simply use errors="coerce" and move on?

**Answer:** Coercion can turn invalid text into missing values, but it does not explain why the values were invalid or whether information was lost. The conversion should be followed by an audit of affected rows.

### Q16. An API returns a prediction threshold as the string "0.75". What should your code do?

**Answer:** Validate that the input is present and represents a valid numeric value, convert it to a float, and check that it lies in the range expected by the application. Return a clear validation error if it does not.

**Follow-up:** Is a value of 0.75 automatically the correct threshold?

**Answer:** No. The value is only a configuration choice. A suitable threshold must be selected and validated for the business use case.

### Q17. Your training script works in a notebook but fails in a scheduled job with a package import error. How would you investigate?

**Answer:** I would compare the interpreter and environment used by the notebook with the scheduled job. I would verify which environment is activated, inspect installed package versions, check the dependency file and reproduce the job in a clean environment. I would avoid randomly installing packages into the system Python.

**Follow-up:** How can the team reduce this problem?

**Answer:** Maintain an explicit dependency specification, document environment setup and test the project in a clean environment before deployment.

### Q18. A variable named model_score sometimes contains a float and sometimes contains the text "failed". Is this a good design?

**Answer:** It is risky because later code may expect a number and fail when it receives text. Use separate variables or a structured result with explicit fields such as status and score. The status should indicate whether a valid score is available.

**Follow-up:** What should happen when the model run fails?

**Answer:** Record the failure status and useful error details. Do not invent a numeric score or silently treat failure as zero.

### Q19. A batch-processing script divides the number of rows by a batch size. The batch size comes from a configuration file. What should you validate?

**Answer:** Check that the batch size is numeric, is an integer if the processing API requires an integer, and is greater than zero. Then handle the case where the total number of rows is not an exact multiple of the batch size.

**Follow-up:** Which Python operator gives the remainder?

**Answer:** The modulus operator `%`.

### Q20. A model configuration contains a random_state value. Why might the team store it in a variable?

**Answer:** A named configuration variable makes the setting visible and reusable. It helps keep experiment configuration consistent and makes it easier to compare runs. The exact behaviour of the setting depends on the library using it.

**Follow-up:** Does using a fixed random_state guarantee every part of a production ML system is reproducible?

**Answer:** No. Reproducibility also depends on library versions, data versions, hardware, algorithms, execution settings and other sources of randomness.

## 4. Practical debugging exercise

### Q21. Review the following code. What is wrong?

```python
row_count = input("Enter row count: ")
print(row_count + 10)
```

**Answer:** `input()` returns a string. Adding an integer to that string raises a `TypeError`. Convert and validate the input before performing arithmetic.

**Follow-up:** What error might occur if the user enters "ten"?

**Answer:** Converting that value with `int()` raises `ValueError`.

### Q22. Review this condition. What does it check?

```python
has_file = True
has_target = False
ready = has_file and has_target
print(ready)
```

**Answer:** It checks whether both the file and target column are available. The output is `False` because `has_target` is false.

**Follow-up:** What operator would be appropriate if either of two independent data sources is sufficient?

**Answer:** `or`, provided that either source really is sufficient for the workflow.

### Q23. The following expression gives a result different from what a beginner expects. Explain why.

```python
result = 10 + 5 * 2
print(result)
```

**Answer:** The output is 20 because multiplication is evaluated before addition. Use `(10 + 5) * 2` if addition must happen first.

**Follow-up:** What is a maintainable alternative for a complex calculation?

**Answer:** Use parentheses and descriptive intermediate variables, then test the calculation with known inputs.

## 5. Interview response framework

For a Python or data-quality scenario, structure the answer in this order:

1. **Clarify the expected value:** What should the variable or column represent?
2. **Inspect the actual value:** Check examples, type and missing/invalid cases.
3. **Validate:** Define accepted types, ranges and formats.
4. **Convert carefully:** Convert only values that satisfy the data rules.
5. **Handle exceptions:** Return a clear message or apply an approved cleaning rule.
6. **Test:** Include normal, boundary and invalid inputs.
7. **Document:** Record assumptions and the number of changed or rejected records.

**Senior developer principle:** A value that looks correct when printed may still have the wrong type or meaning. Validate both representation and business semantics before using it in a model.
