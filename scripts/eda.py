#!/usr/bin/env python3
"""Reproducible EDA for Bangalore house prices."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml_project.preprocessing import (
    EXPECTED_COLUMNS,
    clean_total_sqft,
    load_and_prepare_training_frame,
)


def run_eda(data_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    raw = pd.read_csv(data_path)

    summary = {
        "rows_raw": len(raw),
        "columns": list(raw.columns),
        "nulls": raw.isnull().sum().to_dict(),
        "duplicates": int(raw.duplicated().sum()),
        "unique_locations_raw": int(raw["location"].nunique()),
    }

    parse_fail = raw["total_sqft"].apply(clean_total_sqft).isna().sum()
    summary["sqft_parse_failures"] = int(parse_fail)

    df = load_and_prepare_training_frame(data_path)
    summary["rows_after_preprocessing"] = len(df)
    summary["unique_locations_after"] = int(df["location"].nunique())
    summary["price_mean_lakhs"] = round(float(df["price"].mean()), 2)
    summary["price_median_lakhs"] = round(float(df["price"].median()), 2)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["price"], bins=40, kde=True, ax=ax)
    ax.set_title("Price distribution (lakhs)")
    ax.set_xlabel("Price (lakhs)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "price_histogram.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df.sample(min(2000, len(df))), x="total_sqft", y="price", alpha=0.4, ax=ax)
    ax.set_title("Total sqft vs price")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sqft_vs_price.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="bhk", y="price", ax=ax)
    ax.set_title("Price by BHK")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "price_by_bhk.png"), dpi=120)
    plt.close(fig)

    numeric = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=120)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"EDA artifacts written to {out_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run reproducible EDA")
    parser.add_argument(
        "--data",
        default=os.path.join(ROOT, "data", "bengaluru_house_prices.csv"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "reports", "eda"),
    )
    args = parser.parse_args()
    if not os.path.exists(args.data):
        raise SystemExit(f"Data not found: {args.data}")
    run_eda(args.data, args.out)


if __name__ == "__main__":
    main()
