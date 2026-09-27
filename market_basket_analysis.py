"""
Market Basket Analysis of the Groceries dataset.

Two ways of defining a "transaction" are compared:

  per-customer  every item a customer bought across 2014-2015, merged into
                one transaction. This is what the original IS446 submission
                did, and it flattens the association signal (lift ~= 1.0).

  per-visit     one transaction = one customer on one date. This is what a
                shopping basket actually is, and it is what association rule
                mining assumes.

Run:
    pip install pandas mlxtend
    python market_basket_analysis.py --csv "Groceries data.csv"

Dataset: https://www.kaggle.com/datasets/rashikrahmanpritom/groceries-dataset-for-market-basket-analysismba
"""

import argparse

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Member_number", "Date", "itemDescription"])
    df["itemDescription"] = df["itemDescription"].str.strip()
    return df


def baskets(df, mode):
    """Return a list of transactions (each a list of item names)."""
    if mode == "per_visit":
        keys = ["Member_number", "Date"]
    elif mode == "per_customer":
        keys = ["Member_number"]
    else:
        raise ValueError(mode)
    grouped = df.groupby(keys)["itemDescription"].apply(lambda s: sorted(set(s)))
    return list(grouped)


def encode(transactions):
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)


def mine(matrix, min_support, min_confidence, min_lift):
    items = apriori(matrix, min_support=min_support, use_colnames=True)
    if items.empty:
        return pd.DataFrame()
    rules = association_rules(items, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift]
    return rules.sort_values("lift", ascending=False)


def readable(rules, limit=15):
    if rules.empty:
        return "no rules met the thresholds"
    out = rules.head(limit).copy()
    out["antecedents"] = out["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    out["consequents"] = out["consequents"].apply(lambda s: ", ".join(sorted(s)))
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    return out[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}")


def report(df, mode, min_support, min_confidence, min_lift):
    txns = baskets(df, mode)
    sizes = [len(t) for t in txns]
    matrix = encode(txns)

    print(f"\n{'=' * 72}")
    print(mode)
    print(f"{'=' * 72}")
    print(f"transactions      : {len(txns):,}")
    print(f"distinct items    : {matrix.shape[1]}")
    print(f"mean basket size  : {sum(sizes) / len(sizes):.2f}")
    print(f"max basket size   : {max(sizes)}")

    rules = mine(matrix, min_support, min_confidence, min_lift)
    print(f"rules found       : {len(rules)}")
    if not rules.empty:
        print(f"max lift          : {rules['lift'].max():.3f}")
    print()
    print(readable(rules))
    return rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to Groceries data.csv")
    ap.add_argument("--min-support", type=float, default=0.001)
    ap.add_argument("--min-confidence", type=float, default=0.05)
    ap.add_argument("--min-lift", type=float, default=1.0)
    args = ap.parse_args()

    df = load(args.csv)
    print(f"loaded {len(df):,} rows, "
          f"{df['Member_number'].nunique():,} customers, "
          f"{df['itemDescription'].nunique()} items")

    # A shopping basket is one visit. This is the correct unit.
    report(df, "per_visit", args.min_support, args.min_confidence, args.min_lift)

    # Shown for comparison: merging a customer's whole history removes the
    # signal, because every frequent item ends up beside every other one.
    report(df, "per_customer", 0.10, 0.50, 1.0)


if __name__ == "__main__":
    main()
