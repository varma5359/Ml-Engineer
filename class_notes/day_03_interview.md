# Day 03: Interview Questions — Conditions, Loops and Functions

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is the difference between if, elif and else?
**Answer:** if checks the first condition, elif checks another condition when earlier conditions are false, and else runs when none of the previous conditions match.
**Follow-up:** Why should conditions be ordered carefully?  
**Answer:** Python executes the first matching branch. A broad condition can prevent a more specific branch from running.

### Q2. What is the difference between for and while loops?
**Answer:** A for loop iterates over an iterable such as a list or range. A while loop repeats as long as its condition is true.
**Follow-up:** What can cause an infinite while loop?  
**Answer:** The condition never becomes false, for example because the loop variable is not updated.

### Q3. Explain break and continue.
**Answer:** break exits the nearest loop. continue skips the rest of the current iteration and moves to the next one.
**Follow-up:** Which would you use to stop searching after finding the first matching record?  
**Answer:** break.

### Q4. What is a function and why do we use it?
**Answer:** A function is a reusable named block of code designed to perform a task. It reduces duplication and makes logic easier to test.
**Follow-up:** What is the difference between a parameter and an argument?  
**Answer:** A parameter is named in the function definition; an argument is the value passed when the function is called.

### Q5. What is the difference between print() and return?
**Answer:** print() displays information. return sends a value back to the caller so it can be stored or used elsewhere.
**Follow-up:** What happens if a function has no return statement?  
**Answer:** It returns None by default.

## Practical and ML scenarios

### Q6. How would you handle an empty list while calculating an average?
**Answer:** Check whether the list is empty before dividing by its length. Raise a meaningful error or return a documented missing result, depending on the application.
**Follow-up:** Why is returning zero potentially misleading?  
**Answer:** Zero can be a valid average. It is not the same as having no observations.

### Q7. You need to process 100,000 transaction records. Would you write a separate statement for every row?
**Answer:** No. Use a suitable data structure or DataFrame and reusable logic. For numerical workloads, vectorized Pandas/NumPy operations are often more efficient than Python loops.
**Follow-up:** Are Python loops always wrong for data processing?  
**Answer:** No. They are useful for control flow and custom logic. Choose the clearest correct approach and consider performance.

### Q8. A model returns a probability for each customer. How could a function convert the score into a business action?
**Answer:** A function can apply a validated threshold and return a category such as high-priority review or normal monitoring. The threshold should be selected using validation data and business costs.
**Follow-up:** Should the threshold always be 0.5?  
**Answer:** No. It depends on class balance, calibration and the cost of false positives and false negatives.

### Q9. A loop skips some records unexpectedly. How do you debug it?
**Answer:** Reproduce the issue with a small input, inspect the loop condition and update statements, check break/continue placement and print or log the current item during debugging.
**Follow-up:** What should you add after fixing the bug?  
**Answer:** A test that covers the case that was skipped.

### Q10. A function prints a calculated score, but another function receives None. Why?
**Answer:** The first function displayed the score but did not return it. Use return when another part of the program needs the result.
**Follow-up:** How do you make the function reusable?  
**Answer:** Pass inputs as parameters, return the result, validate inputs and avoid hidden global dependencies.

## Interview answer framework

1. Identify the input and expected output.
2. Explain the decision or repetition needed.
3. Choose the appropriate control structure.
4. Handle empty, invalid and boundary cases.
5. Put reusable logic into functions.
6. Test normal and edge cases.

**Senior developer principle:** Clear control flow and small reusable functions make data-processing code easier to debug and maintain.
