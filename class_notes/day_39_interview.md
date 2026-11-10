# Day 39: Interview Questions — Association Rules and Market Basket Analysis

**Pathway:** Machine Learning Engineer

### Q1. What is market basket analysis?
**Answer:** It identifies products or items that frequently occur together in transactions.
**Follow-up:** Give a business use.  
**Answer:** Product bundles, recommendations, promotion planning or store placement.

### Q2. What is a transaction?
**Answer:** One shopping event or basket containing the items purchased together.

### Q3. What is a frequent itemset?
**Answer:** A group of items that appears in enough transactions to meet a minimum support threshold.
**Follow-up:** Can it contain one item only?  
**Answer:** Yes. A one-item itemset is a valid itemset.

### Q4. What is support?
**Answer:** It measures how frequently an itemset appears across all transactions.
**Follow-up:** Why is low support a concern?  
**Answer:** A pattern may be based on very few baskets and may not be stable or actionable.

### Q5. What is confidence?
**Answer:** It measures how often the consequent appears among transactions containing the antecedent.
**Follow-up:** Is confidence symmetric?  
**Answer:** No. A → B and B → A can have different confidence.

### Q6. What is lift?
**Answer:** It compares observed co-occurrence with what would be expected if the items occurred independently. Lift above one indicates a positive association.
**Follow-up:** Does high lift alone guarantee usefulness?  
**Answer:** No. Check support, volume and business value.

## Scenario questions

### Q7. Bread → Milk has high confidence but low support. What do you consider?
**Answer:** The rule may apply to very few baskets. Check volume, stability and whether it is actionable.

### Q8. A product pair has high lift but appears in only three baskets. Is it useful?
**Answer:** Not necessarily. The evidence may be too sparse to support a reliable decision.

### Q9. The rule A → B has different confidence from B → A. Is that a bug?
**Answer:** No. Confidence is directional and depends on the frequency of the antecedent.

### Q10. A retailer uses a rule to create a bundle. Does the rule prove sales will increase?
**Answer:** No. It identifies co-occurrence. A controlled experiment is needed to measure incremental impact.

### Q11. Product names differ due to spelling and capitalization. What should happen before Apriori?
**Answer:** Normalize product names so the same item is not treated as multiple distinct products.

**Senior developer principle:** Association rules reveal co-occurrence patterns, not causal effects.
