# Day 03: Interview Questions — Conditions, Loops and Functions

**Pathway:** Machine Learning Engineer

---

## Question 1: What is the difference between `if`, `elif` and `else`?

**Answer:** `if` checks the first condition, `elif` checks another condition when earlier conditions are false, and `else` runs when none of the previous conditions match.

**Example:** Assigning grades based on marks.

**Follow-up:** Why should conditions be ordered carefully?

**Answer:** Python executes the first matching branch. A broad condition placed before a more specific condition may prevent the specific branch from running.

## Question 2: What is the difference between `for` and `while` loops?

**Answer:** A `for` loop iterates over an iterable such as a list or range. A `while` loop repeats as long as its condition is true.

**Example:** Use `for` to process each row in a list; use `while` to repeat a process until a stopping condition is met.

**Follow-up:** What can cause an infinite `while` loop?

**Answer:** If the condition never becomes false—for example, the counter is not updated—the loop can continue indefinitely.

## Question 3: Explain `break` and `continue`.

**Answer:** `break` exits the nearest loop. `continue` skips the remaining code in the current iteration and proceeds to the next iteration.

**Follow-up:** Which would you use to stop searching after finding the first matching record?

**Answer:** `break`.

## Question 4: What is a function and why do we use it?

**Answer:** A function is a reusable named block of code designed to perform a task. It reduces duplication, improves readability and makes logic easier to test.

**Example:** A function that calculates revenue can be used for many orders.

**Follow-up:** What is the difference between a parameter and an argument?

**Answer:** A parameter is the name in the function definition; an argument is the actual value passed when calling the function.

## Question 5: What is the difference between `print()` and `return`?

**Answer:** `print()` displays information. `return` sends a value back to the caller so it can be stored or used in another calculation.

**Scenario:** A function prints a calculated average, but another function receives `None` when trying to use it. Why?

**Answer:** The first function printed the average but did not return it. A Python function without an explicit return value returns `None`.

## Question 6: How would you handle an empty list while calculating an average?

**Answer:** Check whether the list is empty before dividing by its length. Raise a meaningful error or return a documented missing result depending on the application's requirements.

**Follow-up:** Why is returning zero potentially misleading?

**Answer:** Zero is a valid average for some datasets. It is not the same as having no observations, so replacing an undefined average with zero can distort analysis.

## Question 7: You need to process 100,000 transaction records. Would you write 100,000 separate statements?

**Answer:** No. I would store or read the records in a suitable collection or dataframe and process them using reusable logic, usually vectorized dataframe/NumPy operations for numerical workloads rather than slow Python loops where appropriate.

**Follow-up:** Are Python loops always wrong for data processing?

**Answer:** No. They are useful for control flow, custom logic and small workloads. For large numerical arrays, vectorized library operations are often faster and clearer.

## Question 8: A model returns a probability for each customer. How could a function convert the score into a business action?

**Answer:** A function can apply a validated threshold and return a category such as high-priority review or normal monitoring. The threshold should be chosen using validation data and the cost of false positives and false negatives. The rule should not be described as the model itself.

**Follow-up:** Should the threshold always be 0.5?

**Answer:** No. The best threshold depends on class balance, probability calibration and business costs. It should be evaluated for the intended application.