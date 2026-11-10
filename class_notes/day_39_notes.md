# Day 39: Association Rules and Market Basket Analysis

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain market basket analysis in plain language.
- Understand transactions and item sets.
- Explain frequent itemsets.
- Understand support, confidence and lift conceptually.
- Prepare transaction data for association analysis.
- Use Apriori to discover frequent itemsets.
- Generate association rules.
- Filter rules using suitable thresholds.
- Interpret rules carefully without claiming causation.
- Identify practical business uses and limitations.

## 2. What is market basket analysis?

**Market basket analysis** studies items that occur together in transactions.

Example:
- Customers who buy bread may also buy butter.
- Customers who buy a phone may also buy a phone case.
- Customers who purchase coffee may also purchase snacks.

A retail company can use these patterns to:
- Design product bundles.
- Improve product placement.
- Recommend related products.
- Plan promotions.
- Understand purchasing combinations.

The analysis identifies associations. It does not prove that buying one item causes another item to be purchased.

## 3. Understand transactions and itemsets

A **transaction** is one shopping event or basket.

An **itemset** is a collection of one or more items that appear together.

Example:

| Transaction | Items |
|---|---|
| T001 | Bread, Milk |
| T002 | Bread, Butter, Milk |
| T003 | Coffee, Sugar |
| T004 | Bread, Butter |

A one-item itemset contains one product. A two-item itemset contains two products purchased together.

## 4. Understand frequent itemsets

A **frequent itemset** is a group of items that appears in enough transactions to meet a minimum support requirement.

Frequent itemsets are used as the starting point for creating association rules.

## 5. Understand support, confidence and lift

| Measure | Beginner meaning |
|---|---|
| Support | How frequently the item combination appears across all transactions |
| Confidence | How often the consequent appears among transactions containing the antecedent |
| Lift | How much more often the items occur together than would be expected from their independent occurrence rates |

Interpretation:
- High support means a pattern occurs frequently.
- High confidence means the consequent often appears when the antecedent is present.
- Lift above one indicates a positive association compared with independent occurrence.
- Lift near one indicates little association beyond the baseline occurrence rates.
- Lift below one indicates a negative association.

No single measure is enough. A rule can have high confidence simply because the consequent is very common.

## 6. Create transaction data

### 6.1 Import libraries

### 6.1. Import Pandas and common tools

```python
import pandas as pd
import numpy as np

print("Ready")
```

**Expected output**

```text
Ready
```

**Explanation**

Pandas and NumPy prepare and inspect the transaction records.



### 6.2 Create a list of baskets

### 6.2. Prepare sample transactions

```python
transactions = [
    ["Bread", "Milk", "Butter"],
    ["Bread", "Milk"],
    ["Bread", "Butter"],
    ["Milk", "Butter"],
    ["Bread", "Milk", "Butter", "Eggs"],
    ["Coffee", "Sugar"],
    ["Coffee", "Milk", "Sugar"],
    ["Bread", "Eggs"],
    ["Milk", "Eggs"],
    ["Bread", "Milk", "Eggs"],
    ["Coffee", "Sugar", "Biscuits"],
    ["Bread", "Butter", "Eggs"]
]

print(transactions[:5])
```

**Expected output**

```text
The first five shopping baskets
```

**Explanation**

Each inner list represents one transaction. A transaction can contain several different items.



### 6.3 Count transactions

### 6.3. Check number of baskets

```python
print(len(transactions))
```

**Expected output**

```text
12
```

**Explanation**

There are 12 transactions in this small teaching dataset.



### 6.4 Inspect unique products

### 6.4. Find all distinct items

```python
unique_items = sorted({
    item
    for basket in transactions
    for item in basket
})

print(unique_items)
```

**Expected output**

```text
A sorted list of distinct products
```

**Explanation**

The set comprehension collects each product once across all transactions.



## 7. Convert transactions into one-hot format

Apriori expects a transaction matrix where each column represents an item and each row represents a transaction. A value indicates whether the item is present.

### 7.1 Import TransactionEncoder

### 7.1. Import TransactionEncoder

```python
from mlxtend.preprocessing import TransactionEncoder

print("TransactionEncoder imported")
```

**Expected output**

```text
TransactionEncoder imported
```

**Explanation**

TransactionEncoder converts transaction lists into a one-hot representation. The mlxtend package may need to be installed in the active Python environment.



### 7.2 Initialize the encoder

### 7.2. Create the encoder object

```python
encoder = TransactionEncoder()

print(encoder)
```

**Expected output**

```text
A TransactionEncoder object
```

**Explanation**

The object is initialized and ready to learn the item vocabulary.



### 7.3 Fit and transform transactions

### 7.3. Encode the baskets

```python
encoded_array = encoder.fit(transactions).transform(transactions)

print(encoded_array.shape)
```

**Expected output**

```text
A Boolean matrix shape with 12 rows and one column per unique product
```

**Explanation**

The encoder learns the item vocabulary and creates one row per transaction.



### 7.4 Convert to a DataFrame

### 7.4. Create the transaction matrix

```python
basket_df = pd.DataFrame(
    encoded_array,
    columns=encoder.columns_
)

print(basket_df.head())
```

**Expected output**

```text
A table with one row per transaction and Boolean item columns
```

**Explanation**

True means the product is present in that basket; False means it is absent.



### 7.5 Inspect product counts

### 7.5. Count product occurrences

```python
item_counts = basket_df.sum().sort_values(ascending=False)

print(item_counts)
```

**Expected output**

```text
A count of transactions containing each product
```

**Explanation**

This is a useful first summary before generating itemsets.



## 8. Generate frequent itemsets using Apriori

### 8.1 Import Apriori

### 8.1. Import the Apriori function

```python
from mlxtend.frequent_patterns import apriori

print("Apriori imported")
```

**Expected output**

```text
Apriori imported
```

**Explanation**

Apriori searches for item combinations that meet the minimum support threshold.



### 8.2 Generate itemsets

### 8.2. Find frequent itemsets

```python
frequent_itemsets = apriori(
    basket_df,
    min_support=0.25,
    use_colnames=True
)

print(frequent_itemsets)
```

**Expected output**

```text
A table containing frequent itemsets and support values
```

**Explanation**

min_support=0.25 means an itemset must occur in at least 25% of transactions. The itemsets are returned using product names.



### 8.3 Sort frequent itemsets by support

### 8.3. Rank itemsets by support

```python
print(
    frequent_itemsets.sort_values(
        "support",
        ascending=False
    )
)
```

**Expected output**

```text
Frequent itemsets ordered by support
```

**Explanation**

Higher-support itemsets occur in more transactions.



### 8.4 Inspect itemset sizes

### 8.4. Count items in each itemset

```python
frequent_itemsets["itemset_size"] = frequent_itemsets["itemsets"].apply(len)

print(frequent_itemsets[["itemsets", "support", "itemset_size"]])
```

**Expected output**

```text
A table showing itemsets, support and number of items
```

**Explanation**

This helps distinguish single-product frequencies from multi-product combinations.



## 9. Generate association rules

### 9.1 Import association_rules

### 9.1. Import the rule-generation function

```python
from mlxtend.frequent_patterns import association_rules

print("Association rules imported")
```

**Expected output**

```text
Association rules imported
```

**Explanation**

The function converts frequent itemsets into directional association rules.



### 9.2 Generate rules using confidence

### 9.2. Create association rules

```python
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

print(rules.head())
```

**Expected output**

```text
A table of association rules and their measures
```

**Explanation**

A rule has an antecedent and a consequent. The threshold filters rules using confidence.



### 9.3 Inspect important rule columns

### 9.3. Select readable rule columns

```python
rule_columns = [
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift"
]

print(rules[rule_columns].head())
```

**Expected output**

```text
A table showing antecedent, consequent, support, confidence and lift
```

**Explanation**

These are common measures for reviewing association rules.



### 9.4 Sort by lift

### 9.4. Rank rules by lift

```python
print(
    rules[rule_columns].sort_values(
        "lift",
        ascending=False
    )
)
```

**Expected output**

```text
Rules ordered by descending lift
```

**Explanation**

Higher lift indicates a stronger positive association relative to the independent occurrence rates, but check support and rule stability too.



### 9.5 Filter rules with support and lift

### 9.5. Apply additional rule filters

```python
filtered_rules = rules[
    (rules["support"] >= 0.25) &
    (rules["lift"] > 1)
]

print(filtered_rules[rule_columns])
```

**Expected output**

```text
A subset of rules meeting the chosen support and lift conditions
```

**Explanation**

Thresholds depend on transaction volume and business objectives. Very rare rules may not be actionable even if lift is high.



## 10. Understand rule direction

An association rule such as Bread → Butter is directional in its confidence interpretation:
- Antecedent: Bread.
- Consequent: Butter.
- Confidence asks how often Butter appears among baskets containing Bread.

The reverse rule Butter → Bread can have a different confidence because Bread and Butter may have different overall frequencies.

### 10.1 Inspect the reverse rule

### 10.1. Compare both rule directions

```python
direction_rules = rules[
    (
        rules["antecedents"].apply(lambda items: "Bread" in items) &
        rules["consequents"].apply(lambda items: "Butter" in items)
    ) |
    (
        rules["antecedents"].apply(lambda items: "Butter" in items) &
        rules["consequents"].apply(lambda items: "Bread" in items)
    )
]

print(direction_rules[rule_columns])
```

**Expected output**

```text
Any generated Bread/Butter rules in both directions
```

**Explanation**

Confidence can differ when the direction is reversed. Association does not imply that one product causes the other purchase.



## 11. Explain a rule to a business team

Suppose a rule has:
- Antecedent: Bread.
- Consequent: Milk.
- Support: 0.40.
- Confidence: 0.70.
- Lift: 1.20.

A plain-language explanation:
- Bread and Milk appear together in 40% of all baskets.
- Among baskets containing Bread, 70% also contain Milk.
- The pair occurs more often than expected from their independent occurrence rates.

A retailer could test a bundle or product placement experiment. The rule alone does not prove that a promotion will increase sales.

## 12. Visualize association rules

### 12.1 Plot support against confidence

### 12.1. Create a rule scatter plot

```python
if not rules.empty:
    plt.scatter(
        rules["support"],
        rules["confidence"],
        s=rules["lift"] * 30
    )
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association rules: support and confidence")
    plt.show()
else:
    print("No rules found for the selected threshold")
```

**Expected output**

```text
A scatter plot when rules are available, or a message if no rules meet the threshold
```

**Explanation**

Bubble size is scaled by lift for visual exploration. A chart is not a substitute for reading the actual rule values.



### 12.2 Inspect the strongest rules

### 12.2. Display top rules

```python
if not rules.empty:
    top_rules = rules.sort_values(
        ["lift", "support"],
        ascending=False
    ).head(10)

    print(top_rules[rule_columns])
else:
    print("No rules to display")
```

**Expected output**

```text
Up to ten rules with the highest lift and support
```

**Explanation**

Sorting by both lift and support can help prioritize rules for review.



## 13. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Treat confidence as probability of causation | Explain it as conditional co-occurrence |
| Ignore support | A rare rule may not matter operationally |
| Rank only by lift | Review support, confidence and business value |
| Assume rule direction is symmetric | Compare both directions |
| Use transactions with inconsistent product names | Normalize item labels before encoding |
| Include repeated item names in one basket without thought | Define whether basket data is binary presence or quantity |
| Choose thresholds to create impressive rules | Select thresholds based on business requirements |
| Assume a discovered rule guarantees campaign success | Run a controlled business experiment |

## 14. Interview-ready explanations

**What is market basket analysis?**  
It identifies products or items that frequently occur together in transactions.

**What is support?**  
It measures how often an itemset appears across all transactions.

**What is confidence?**  
It measures how often the consequent appears among transactions containing the antecedent.

**What is lift?**  
It compares the observed co-occurrence with what would be expected if the items occurred independently. Lift above one indicates a positive association.

**Does an association rule prove causation?**  
No. It identifies co-occurrence patterns. A controlled experiment is needed to test whether a promotion or placement causes a change.

## 15. Scenario questions

1. Bread → Milk has high confidence but low support. What do you consider?  
   **Answer:** The rule may apply to a small number of baskets. Check volume, stability and whether the pattern is actionable.

2. A product pair has high lift but appears in only three baskets. Is it useful?  
   **Answer:** Not necessarily. The evidence may be too sparse to support a reliable business decision.

3. The rule A → B has a different confidence from B → A. Is that a bug?  
   **Answer:** No. Confidence is directional and depends on the frequency of the antecedent.

4. A retailer uses the rule to create a bundle. Does the rule prove sales will increase?  
   **Answer:** No. Test the bundle with an experiment and measure incremental impact.

5. Product names differ due to spelling and capitalization. What should happen before Apriori?  
   **Answer:** Normalize and validate product names so the same product is not treated as multiple items.

## 16. Lesson completion checklist

- [ ] I can explain market basket analysis.
- [ ] I understand transactions and itemsets.
- [ ] I can encode baskets into one-hot columns.
- [ ] I can generate frequent itemsets with Apriori.
- [ ] I understand support, confidence and lift.
- [ ] I can generate and filter association rules.
- [ ] I can explain rule direction.
- [ ] I understand why association does not prove causation.
- [ ] I can communicate rules in business language.

**Next lesson:** Day 40 — Anomaly Detection Concepts.
