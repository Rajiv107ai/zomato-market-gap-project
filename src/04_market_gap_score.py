"""
Phase 4: Market Gap Scoring

Builds a composite opportunity score per (locality, cuisine) combination:
    - LOW existing restaurant count for that cuisine in that locality   (less competition)
    - HIGH average rating of nearby restaurants                        (signals demand/spending power)
    - HIGH average nearest-competitor distance                          (room to breathe)
    - price-tier gap: locality has restaurants but missing a cost tier

Input:  data/zomato_with_competition.csv
Output: outputs/market_gap_scores.csv (ranked)
        outputs/recommendation_memo.md
"""
import pandas as pd
import numpy as np

IN_PATH = "data/zomato_with_competition.csv"
SCORES_OUT = "outputs/market_gap_scores.csv"
MEMO_OUT = "outputs/recommendation_memo.md"

MIN_LOCALITY_RESTAURANTS = 15  # ignore tiny localities, not enough signal


def top_dishes_for_locality(df, locality, top_n=5):
    """Most-liked dishes for restaurants in a given locality."""
    subset = df[df["location"] == locality]["dish_liked"].dropna()
    all_dishes = subset.str.split(",").explode().str.strip()
    return all_dishes.value_counts().head(top_n)


def online_features_correlation(df):
    """Check if online_order / book_table correlate with rating."""
    corr_online = df["online_order"].corr(df["rating"])
    corr_booking = df["book_table"].corr(df["rating"])
    avg_rating_with_online = df[df["online_order"] == 1]["rating"].mean()
    avg_rating_without_online = df[df["online_order"] == 0]["rating"].mean()
    avg_rating_with_booking = df[df["book_table"] == 1]["rating"].mean()
    avg_rating_without_booking = df[df["book_table"] == 0]["rating"].mean()
    return {
        "corr_online_order_rating": corr_online,
        "corr_book_table_rating": corr_booking,
        "avg_rating_with_online": avg_rating_with_online,
        "avg_rating_without_online": avg_rating_without_online,
        "avg_rating_with_booking": avg_rating_with_booking,
        "avg_rating_without_booking": avg_rating_without_booking,
    }


def price_tier(cost):
    if cost < 300:
        return "budget"
    elif cost < 700:
        return "mid"
    else:
        return "premium"


def main():
    df = pd.read_csv(IN_PATH)
    df["price_tier"] = df["cost_for_two"].apply(price_tier)

    # Only consider localities with enough data to be meaningful
    locality_counts = df.groupby("location").size()
    valid_localities = locality_counts[locality_counts >= MIN_LOCALITY_RESTAURANTS].index
    df = df[df["location"].isin(valid_localities)]

    grouped = (
        df.groupby(["location", "primary_cuisine"])
        .agg(
            restaurant_count=("name", "count"),
            avg_rating=("rating", "mean"),
            avg_competitor_distance=("nearest_competitor_km", "mean"),
            avg_cost=("cost_for_two", "mean"),
            tiers_present=("price_tier", lambda x: set(x)),
        )
        .reset_index()
    )

    # Normalize each component 0-1 so they're comparable
    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    grouped["score_low_competition"] = 1 - norm(grouped["restaurant_count"])
    grouped["score_demand"] = norm(grouped["avg_rating"])
    grouped["score_room_to_breathe"] = norm(grouped["avg_competitor_distance"].fillna(0))
    grouped["missing_price_tiers"] = grouped["tiers_present"].apply(
        lambda present: len({"budget", "mid", "premium"} - present)
    )
    grouped["score_price_gap"] = norm(grouped["missing_price_tiers"])

    # Composite opportunity score
    grouped["opportunity_score"] = (
        0.35 * grouped["score_low_competition"]
        + 0.35 * grouped["score_demand"]
        + 0.15 * grouped["score_room_to_breathe"]
        + 0.15 * grouped["score_price_gap"]
    )

    grouped = grouped.sort_values("opportunity_score", ascending=False)
    grouped.drop(columns=["tiers_present"]).to_csv(SCORES_OUT, index=False)
    print(f"Saved -> {SCORES_OUT}")

    # Write a short business-facing memo with the top 10 opportunities
    top10 = grouped.head(10)
    lines = [
        "# Recommendation Memo: Top Restaurant Opportunities in Bangalore\n",
        "Based on a composite opportunity score combining competition density, ",
        "demand signal (average ratings), spatial room-to-breathe, and price-tier gaps.\n",
        "## Top 10 Locality x Cuisine Opportunities\n",
        "| Rank | Locality | Cuisine | Restaurants | Avg Rating | Opportunity Score |",
        "|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(top10.itertuples(), start=1):
        lines.append(
            f"| {i} | {row.location} | {row.primary_cuisine} | "
            f"{row.restaurant_count} | {row.avg_rating:.2f} | {row.opportunity_score:.3f} |"
        )
    lines.append("\n## How to read this")
    lines.append(
        "A high opportunity score means: relatively few existing competitors "
        "in that cuisine, strong nearby demand signal (ratings), restaurants "
        "aren't packed on top of each other, and there's an open price tier."
    )

    # ---  Most-liked dishes in each top opportunity locality ---
    lines.append("\n## Popular dishes already working in these localities\n")
    lines.append(
        "Dishes people already like nearby — useful as a starting menu idea "
        "for a new restaurant entering that locality.\n"
    )
    seen_localities = set()
    for row in top10.itertuples():
        if row.location in seen_localities:
            continue
        seen_localities.add(row.location)
        dishes = top_dishes_for_locality(df, row.location, top_n=5)
        if len(dishes) == 0:
            continue
        dish_str = ", ".join(dishes.index.tolist())
        lines.append(f"- **{row.location}**: {dish_str}")

    # --- Does online ordering / table booking correlate with rating? ---
    online_stats = online_features_correlation(df)
    lines.append("\n## Does online ordering or table booking matter?\n")
    lines.append(
        f"- Correlation between online_order and rating: {online_stats['corr_online_order_rating']:.3f}\n"
        f"- Correlation between book_table and rating: {online_stats['corr_book_table_rating']:.3f}\n"
        f"- Avg rating with online ordering: {online_stats['avg_rating_with_online']:.2f} "
        f"vs without: {online_stats['avg_rating_without_online']:.2f}\n"
        f"- Avg rating with table booking: {online_stats['avg_rating_with_booking']:.2f} "
        f"vs without: {online_stats['avg_rating_without_booking']:.2f}"
    )

    with open(MEMO_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved -> {MEMO_OUT}")


if __name__ == "__main__":
    main()
