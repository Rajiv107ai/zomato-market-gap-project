"""
Phase 1: Clean and preprocess the raw Zomato Bangalore dataset.

Input:  data/zomato.csv
Output: data/zomato_clean.csv
"""
import pandas as pd
from utils import parse_rate, parse_cost, split_cuisines

RAW_PATH = "data/zomato.csv"
OUT_PATH = "data/zomato_clean.csv"


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Raw rows: {len(df)}")

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Parse rating ("4.1/5" -> 4.1, "NEW"/"-" -> NaN)
    df["rating"] = df["rate"].apply(parse_rate)

    # Parse cost for two
    df["cost_for_two"] = df["approx_cost(for two people)"].apply(parse_cost)

    # Drop rows with no rating, no cost, or no location
    df = df.dropna(subset=["rating", "cost_for_two", "location"])

    # Cuisines as a list column, plus a primary (first-listed) cuisine
    df["cuisine_list"] = df["cuisines"].apply(split_cuisines)
    df["primary_cuisine"] = df["cuisine_list"].apply(lambda x: x[0] if x else "Unknown")

    # Standardize boolean-ish columns
    df["online_order"] = df["online_order"].map({"Yes": 1, "No": 0})
    df["book_table"] = df["book_table"].map({"Yes": 1, "No": 0})

    # Clean up location strings (strip whitespace, fix casing)
    df["location"] = df["location"].astype(str).str.strip()

    # Keep only columns we need downstream
    keep = [
        "name", "location", "rating", "votes", "cost_for_two",
        "online_order", "book_table", "rest_type",
        "cuisine_list", "primary_cuisine", "cuisines", "dish_liked",
    ]
    df = df[keep].reset_index(drop=True)

    print(f"Clean rows: {len(df)}")
    print(f"Unique localities: {df['location'].nunique()}")
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
