# Day 17: Interview Questions — Matplotlib and Seaborn for ML Visualization

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is the difference between Matplotlib and Seaborn?
**Answer:** Matplotlib is a foundational plotting library with detailed figure and axes control. Seaborn provides higher-level statistical plots and convenient integration with Pandas.
**Follow-up:** Can Seaborn work with Matplotlib?  
**Answer:** Yes. Seaborn charts are built on Matplotlib and can be customized with Matplotlib functions.

### Q2. When do you use a histogram?
**Answer:** To inspect the distribution of a numerical variable, including spread, skewness and possible unusual values.
**Follow-up:** What does changing bins do?  
**Answer:** It changes how values are grouped. Too few bins can hide patterns; too many can create noise.

### Q3. When do you use a box plot?
**Answer:** To compare medians, quartiles, spread and potential extreme observations across groups.
**Follow-up:** Are points outside whiskers always errors?  
**Answer:** No. They are potential outliers under the box-plot convention and need investigation.

### Q4. When do you use a scatter plot?
**Answer:** To inspect the relationship between two numerical variables and identify trends, clusters or unusual observations.
**Follow-up:** Does a trend prove causation?  
**Answer:** No.

### Q5. Why label axes and units?
**Answer:** Without labels and units, a chart can be misinterpreted. A chart should be understandable without reading the code.

## Scenario questions

### Q6. You want to compare customer spend distributions. Which charts would you use?
**Answer:** A histogram to inspect the distribution and a box plot to summarize spread and potential extremes.
**Follow-up:** What if spend is strongly skewed?  
**Answer:** Report median and spread, inspect extreme values and consider whether a transformed view is useful.

### Q7. You want to compare customer counts by contract type. Which chart?
**Answer:** A count plot or bar chart.
**Follow-up:** What if the categories have long names?  
**Answer:** Use a horizontal bar chart or rotate labels for readability.

### Q8. You want to show monthly sales trends. Which chart?
**Answer:** A line chart with months in chronological order.
**Follow-up:** Can a line chart prove seasonality from six months?  
**Answer:** No. More repeated time periods are needed to assess seasonality.

### Q9. One region has twice as many returns because it has twice as many orders. What should you compare?
**Answer:** Return rate per order, not raw return count alone.
**Follow-up:** Why?  
**Answer:** Rates account for different group sizes.

### Q10. A chart has a truncated axis that exaggerates a small difference. What is the issue?
**Answer:** The visual scale may mislead the reader. Use an appropriate scale and clearly disclose any non-zero baseline or transformation.

## Code interpretation

### Q11. What does hue do in Seaborn?
**Answer:** It separates observations by a categorical variable using different visual encodings, commonly colors.
**Follow-up:** What is a risk of using too many hue categories?  
**Answer:** The plot becomes crowded and difficult to interpret.

### Q12. Why use plt.show()?
**Answer:** It displays the current figure in many notebook and script workflows. Exact display behaviour depends on the environment.

**Senior developer principle:** Choose the chart to answer a question, not to decorate a notebook.
