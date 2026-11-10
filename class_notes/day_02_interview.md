# Day 02: Interview Questions — Python Environment, Variables, Data Types and Operators

**Pathway:** Machine Learning Engineer

---

## Question 1: What is a Python environment and why do we use a virtual environment?

**Answer:** A Python environment is the interpreter and package setup used to run code. A virtual environment isolates a project's dependencies so that one project's package versions do not unintentionally affect another project.

**Example:** Project A may require one scikit-learn version while Project B requires another. Separate environments help manage these requirements.

**Follow-up:** Does activating `.venv` install Python itself?

**Answer:** No. It creates an isolated environment using an installed Python interpreter and provides an isolated package location.

## Question 2: What is the difference between `int`, `float`, `str` and `bool`?

**Answer:** `int` represents whole numbers, `float` represents decimal numbers, `str` represents text, and `bool` represents `True` or `False`.

**Example:** Number of purchases can be an integer, product price a float, product name a string and whether a customer is active a Boolean.

**Follow-up:** Why should a model's numeric feature not accidentally remain text?

**Answer:** Numeric calculations, scaling and most ML estimators expect numeric values. Text representations may cause conversion errors or be interpreted as categories rather than quantities.

## Question 3: What happens when you run `int("abc")`?

**Answer:** Python raises `ValueError` because `"abc"` cannot be interpreted as an integer.

**Follow-up:** How would you handle user input safely?

**Answer:** Use validation and a `try/except ValueError` block, then show a useful message or request corrected input.

## Question 4: What is the difference between `=` and `==`?

**Answer:** `=` assigns a value to a variable. `==` compares two values and returns a Boolean result.

**Example:** `age = 20` stores 20. `age == 20` evaluates to `True`.

**Follow-up:** What is the purpose of `!=`?

**Answer:** It checks whether two values are not equal.

## Question 5: A customer dataset stores age values as strings. Can you train a model directly?

**Answer:** Not safely without understanding the column. If age is genuinely numeric, validate the text values and convert them to numeric values. Handle invalid entries and missing values explicitly. If the column is an identifier or category encoded as digits, numeric conversion may be inappropriate.

**Follow-up:** What if some rows contain `"unknown"`?

**Answer:** Treat them as invalid or missing according to the data definition, investigate their frequency and decide on a suitable missing-data strategy. Do not convert them to zero automatically because zero has a different meaning.

## Question 6: What is the difference between `/` and `//`?

**Answer:** `/` performs true division and returns a floating-point result. `//` performs floor division, rounding the quotient down to the next whole-number value for positive numbers.

**Example:** `10 / 3` gives approximately `3.3333`; `10 // 3` gives `3`.

**Follow-up:** What does `%` return?

**Answer:** It returns the remainder after division. For example, `10 % 3` returns `1`.

## Question 7: Explain `and`, `or` and `not`.

**Answer:** `and` is true only when both conditions are true; `or` is true when at least one condition is true; `not` reverses a Boolean condition.

**Example:** A discount may require `is_member and cart_total >= 1000`.

**Follow-up:** What happens when one condition in an `and` expression is false?

**Answer:** The complete expression is false. Python may short-circuit and avoid evaluating the second condition when the first already determines the result.

## Question 8: A prediction score is 0.92. Does `score >= 0.90` prove that the prediction is correct?

**Answer:** No. It only confirms the score is greater than or equal to the selected threshold. The score may be a probability-like estimate, but correctness must be checked against actual outcomes and an appropriate evaluation process.

**Follow-up:** Why is threshold selection important?

**Answer:** A threshold determines which predictions are treated as positive. Changing it affects false positives and false negatives, so it should reflect the use case and error costs.