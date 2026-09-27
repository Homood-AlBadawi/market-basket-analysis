# Market Basket Analysis — Groceries Dataset

Association rule mining on retail transaction data, using the Apriori algorithm.
Coursework for IS446 (Knowledge Discovery and Data Mining), Prince Sultan University,
extended afterwards with a corrected analysis.

## Dataset

[Groceries Dataset for Market Basket Analysis](https://www.kaggle.com/datasets/rashikrahmanpritom/groceries-dataset-for-market-basket-analysismba) (Kaggle)

- 38,765 rows of individual item purchases
- 3,898 customers, 167 distinct grocery items, 2014–2015

The CSV is not committed here. Download it from the link above and place it beside
the script.

## Running it

```bash
pip install pandas mlxtend
python market_basket_analysis.py --csv "Groceries data.csv"
```

## What this repository shows

The original submission defined a transaction as **one customer** — every item that
customer bought across the full two years, merged into a single row. The Apriori run
produced rules with confidence between 53% and 83%, but lift values clustered at 1.00
to 1.06.

Lift of 1.0 means the items are statistically independent. Checking all 45 item pairs
in the submitted data:

| | |
|---|---|
| Highest pair lift | 1.13 (sausage + yogurt) |
| Mean pair lift | 0.96 |
| Lowest pair lift | 0.74 |

So there was no association to find, and almost every generated rule was built from
`=FALSE` conditions.

**The cause was the transaction definition.** A market basket is one shopping trip.
Merging two years of visits into one row puts every frequently-bought item beside
every other frequently-bought item, which averages the co-occurrence signal away.

`market_basket_analysis.py` runs both definitions on the same data so the difference
is visible:

- **per-visit** — one transaction = one customer on one date, all 167 items kept
- **per-customer** — the original grouping, kept for comparison

It also drops the two constraints that were not actually necessary: the analysis was
cut to 1,000 rows and the top 10 items to fit WEKA's memory, but 38,765 rows is small
enough to mine in full with mlxtend.

## Method

1. Load and clean the transaction log
2. Group into baskets by the chosen transaction key
3. One-hot encode with `TransactionEncoder`
4. Generate frequent itemsets with Apriori
5. Derive rules and filter by confidence and lift

Rules are scored on three measures:

- **Support** — how often the itemset appears
- **Confidence** — P(consequent | antecedent)
- **Lift** — observed co-occurrence over what independence would predict; above 1.0
  means a real positive association

## Method files

| File | |
|---|---|
| `market_basket_analysis.py` | Corrected analysis, both transaction definitions |
| `docs/` | Original term project report and presentation |
| `data/op1_member_1000.arff` | The reduced dataset used in the original WEKA run |

## Reference

Çiçekli, U. G., & Kabasakal, İ. (2021). *Market Basket Analysis of Basket Data with
Demographics: A Case Study in E-Retailing.* Alphanumeric Journal, 9(1).

## Team

Homood Saeed AlBadawi · Malik Tariq Mohammed · Abdullah Sameer Babkeer
