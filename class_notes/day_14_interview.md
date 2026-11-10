# Day 14: Interview Questions — EDA, Distributions and Feature Relationships

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is Exploratory Data Analysis?
**Answer:** EDA is a structured process for understanding dataset quality, structure, distributions, patterns and relationships before modelling.
**Follow-up:** Is EDA only chart creation?  
**Answer:** No. It includes schema inspection, summaries, missingness, group comparisons and visual analysis.

### Q2. What is univariate, bivariate and multivariate analysis?
**Answer:** Univariate analysis studies one variable; bivariate analysis studies two variables together; multivariate analysis examines several variables together.
**Follow-up:** Give a multivariate example.  
**Answer:** Comparing churn rate by city and contract type while also inspecting average spend.

### Q3. Why inspect distributions?
**Answer:** Distributions reveal typical values, spread, skewness, gaps and possible unusual observations.
**Follow-up:** Does a histogram prove a feature is normally distributed?  
**Answer:** No. It is an exploratory visualization, and formal assessment may be needed for specific assumptions.

### Q4. Why use groupby()?
**Answer:** It summarizes metrics within meaningful categories, such as average spend or churn rate by contract type.
**Follow-up:** Why show group counts too?  
**Answer:** Rates from very small groups can be unstable or misleading.

## Scenario questions

### Q5. Annual customers show lower churn, but there are only five annual customers. What do you say?
**Answer:** It is a small-sample descriptive pattern. Show the count and avoid a strong conclusion until more representative data is available.
**Follow-up:** Does the pattern still deserve investigation?  
**Answer:** Yes, but treat it as a hypothesis to validate.

### Q6. A scatter plot shows a relationship between spend and churn. Does spend cause churn?
**Answer:** No. A chart shows association. Other variables may explain the pattern.
**Follow-up:** What can you do next?  
**Answer:** Inspect relevant segments, timing and possible confounding variables, then validate on suitable data.

### Q7. A column almost perfectly predicts the target. What do you inspect?
**Answer:** Its definition and timestamp. It may be recorded after the outcome or directly encode the target.
**Follow-up:** Why is prediction time important?  
**Answer:** The deployed model can only use information available when it makes the prediction.

### Q8. One region has more support calls than another. Does it mean service quality is worse?
**Answer:** Not necessarily. Compare calls per customer or per order, group size and customer mix.
**Follow-up:** Why use a rate instead of a count?  
**Answer:** Groups of different sizes cannot be compared fairly using raw counts alone.

### Q9. Spend has a long right tail. What does that suggest?
**Answer:** The feature is positively skewed. Inspect high values, compare mean and median, and consider robust summaries or a suitable transformation.
**Follow-up:** Should you automatically remove high values?  
**Answer:** No. Verify whether they are valid business observations.

## Practical interpretation

### Q10. Which chart would you use for numeric feature distribution?
**Answer:** A histogram, supported by a box plot when spread and potential outliers need inspection.
**Follow-up:** Which chart for relationship between two numeric features?  
**Answer:** Scatter plot.

### Q11. A city-category pivot table has missing combinations. What might that mean?
**Answer:** There may be no observations for those combinations, or the data may be incomplete. Do not automatically interpret missing as zero without confirming the meaning.

### Q12. How do you communicate an EDA finding?
**Answer:** State the observation, group or time period, supporting counts or summaries, business implication and limitations. Avoid unsupported causal claims.

**Senior developer principle:** EDA findings are evidence for further investigation, not automatic business conclusions.
