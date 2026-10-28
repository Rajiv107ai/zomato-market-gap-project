"""
Phase 0 & 6: Data-layer fixes, Model Persistence, and Web Data Export.

Handles:
1. Re-computes v1 score and verifies exact reproduction (<1e-9) against outputs/market_gap_scores.csv
2. Computes v2 score replacing degenerate distance with locality-level cuisine saturation
   and applies empirical Bayes shrinkage to ratings to mitigate small-sample inflation.
3. Assigns confidence tiers (High >= 10, Medium 3-9, Low < 3).
4. Trains two models:
   - with_votes (leakage baseline, ~91% accuracy)
   - no_votes (zero-day honest predictor, ~84% accuracy, ~0.90 ROC-AUC)
5. Persists models and label encoders via joblib.
6. Precomputes global SHAP summary and 2,000-sample beeswarm data.
7. Emits and validates versioned typed JSONs to outputs/web_data/ and web/public/data/:
   - localities.json
   - opportunities.json
   - restaurants.min.json
   - cuisine_matrix.json
   - model_meta.json
   - shap_global.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
WEB_DATA_DIR = ROOT_DIR / "web" / "public" / "data"
OUTPUTS_WEB_DIR = OUTPUTS_DIR / "web_data"

IN_ZOMATO = DATA_DIR / "zomato_with_competition.csv"
IN_LOCALITY_COORDS = DATA_DIR / "locality_coords.csv"
IN_SATURATION = OUTPUTS_DIR / "cuisine_saturation.csv"
IN_SCORES_V1 = OUTPUTS_DIR / "market_gap_scores.csv"

MIN_LOCALITY_RESTAURANTS = 15
BAYES_PRIOR_WEIGHT = 5.0  # m in (n * r + m * global_mean) / (n + m)


# ---------------------------------------------------------------------------
# Pydantic Schemas for Data Contracts
# ---------------------------------------------------------------------------
class ScoreBreakdown(BaseModel):
    score_low_competition: float
    score_demand: float
    score_room_to_breathe: float
    score_price_gap: float
    opportunity_score: float


class OpportunityItem(BaseModel):
    id: str
    location: str
    primary_cuisine: str
    restaurant_count: int
    locality_total: int
    share_of_locality: float
    avg_rating: float
    avg_rating_adj: float
    avg_cost: float
    avg_competitor_distance: float
    missing_price_tiers: int
    price_tiers_present: List[str]
    missing_tiers: List[str]
    confidence: str
    scores_v1: ScoreBreakdown
    scores_v2: ScoreBreakdown
    lat: float
    lon: float


class LocalityCuisineShare(BaseModel):
    cuisine: str
    count: int
    share: float


class LocalityItem(BaseModel):
    locality: str
    lat: float
    lon: float
    total_restaurants: int
    avg_rating: float
    avg_cost: float
    top_cuisines: List[LocalityCuisineShare]


class RestaurantsMin(BaseModel):
    count: int
    lat: List[float]
    lon: List[float]
    name: List[str]
    location: List[str]
    cuisine: List[str]
    rating: List[float]
    cost: List[float]
    online_order: List[int]
    book_table: List[int]


class CuisineMatrix(BaseModel):
    cuisines: List[str]
    localities: List[str]
    locality_totals: Dict[str, int]
    # Matrix entries: mapping locality -> dict(cuisine -> count)
    counts: Dict[str, Dict[str, int]]
    shares: Dict[str, Dict[str, float]]


class ModelEvaluation(BaseModel):
    name: str
    description: str
    features: List[str]
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    feature_importance: Dict[str, float]


class ModelMeta(BaseModel):
    target_threshold: float
    models: Dict[str, ModelEvaluation]
    categorical_options: Dict[str, List[str]]
    median_defaults: Dict[str, float]


class ShapSamplePoint(BaseModel):
    feature_values: Dict[str, float]
    shap_values: Dict[str, float]


class ShapGlobal(BaseModel):
    feature_names: List[str]
    feature_display_names: Dict[str, str]
    mean_abs_shap: Dict[str, float]
    sample_points: List[ShapSamplePoint]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def price_tier(cost: float) -> str:
    if cost < 300:
        return "budget"
    elif cost < 700:
        return "mid"
    else:
        return "premium"


def norm(s: pd.Series) -> pd.Series:
    s_min = s.min()
    s_max = s.max()
    return (s - s_min) / (s_max - s_min + 1e-9)


def get_confidence_tier(restaurant_count: int) -> str:
    if restaurant_count >= 10:
        return "high"
    elif restaurant_count >= 3:
        return "medium"
    else:
        return "low"


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_export():
    print("=" * 60)
    print("STARTING PHASE 0: DATA AUDIT, REPRODUCIBILITY & EXPORT PIPELINE")
    print("=" * 60)

    # 1. Load Data
    df_raw = pd.read_csv(IN_ZOMATO)
    coords_df = pd.read_csv(IN_LOCALITY_COORDS).set_index("location")
    sat_df = pd.read_csv(IN_SATURATION)
    
    global_mean_rating = float(df_raw["rating"].mean())
    print(f"Loaded {len(df_raw)} restaurants across {df_raw['location'].nunique()} localities.")
    print(f"Global average rating: {global_mean_rating:.4f}")

    # Check 0km degenerate distance
    zero_dist_pct = (df_raw["nearest_competitor_km"] == 0).mean() * 100
    print(f"Audit finding: nearest_competitor_km == 0 for {zero_dist_pct:.1f}% of rows.")

    # 2. Filter localities >= 15 restaurants (matching Phase 4)
    df = df_raw.copy()
    df["price_tier"] = df["cost_for_two"].apply(price_tier)
    locality_counts = df.groupby("location").size()
    valid_localities = locality_counts[locality_counts >= MIN_LOCALITY_RESTAURANTS].index
    df = df[df["location"].isin(valid_localities)].copy()

    # Calculate locality totals
    locality_total_map = locality_counts.to_dict()

    # 3. Aggregate locality x primary_cuisine
    grouped = (
        df.groupby(["location", "primary_cuisine"])
        .agg(
            restaurant_count=("name", "count"),
            avg_rating=("rating", "mean"),
            avg_competitor_distance=("nearest_competitor_km", "mean"),
            avg_cost=("cost_for_two", "mean"),
            tiers_present=("price_tier", lambda x: sorted(list(set(x)))),
        )
        .reset_index()
    )

    # Fill NaNs in avg_competitor_distance with 0.0
    grouped["avg_competitor_distance"] = grouped["avg_competitor_distance"].fillna(0.0)

    # Attach locality_total and share_of_locality
    grouped["locality_total"] = grouped["location"].map(locality_total_map)
    grouped["share_of_locality"] = grouped["restaurant_count"] / grouped["locality_total"]

    # Bayesian shrinkage for avg_rating
    grouped["avg_rating_adj"] = (
        grouped["restaurant_count"] * grouped["avg_rating"] + BAYES_PRIOR_WEIGHT * global_mean_rating
    ) / (grouped["restaurant_count"] + BAYES_PRIOR_WEIGHT)

    grouped["confidence"] = grouped["restaurant_count"].apply(get_confidence_tier)

    # Missing price tiers
    all_tiers = {"budget", "mid", "premium"}
    grouped["missing_tiers"] = grouped["tiers_present"].apply(lambda tp: sorted(list(all_tiers - set(tp))))
    grouped["missing_price_tiers"] = grouped["missing_tiers"].apply(len)

    # 4. Compute v1 scores
    grouped["score_low_competition"] = 1.0 - norm(grouped["restaurant_count"])
    grouped["score_demand_v1"] = norm(grouped["avg_rating"])
    grouped["score_room_to_breathe_v1"] = norm(grouped["avg_competitor_distance"].fillna(0))
    grouped["score_price_gap"] = norm(grouped["missing_price_tiers"])

    grouped["opportunity_score_v1"] = (
        0.35 * grouped["score_low_competition"]
        + 0.35 * grouped["score_demand_v1"]
        + 0.15 * grouped["score_room_to_breathe_v1"]
        + 0.15 * grouped["score_price_gap"]
    )

    # Verify v1 reproducibility against outputs/market_gap_scores.csv
    orig_scores = pd.read_csv(IN_SCORES_V1)
    merged_check = pd.merge(
        grouped, orig_scores, on=["location", "primary_cuisine"], suffixes=("_calc", "_orig")
    )
    max_diff = np.abs(merged_check["opportunity_score_v1"] - merged_check["opportunity_score"]).max()
    print(f"Reproducibility check: Max abs diff for v1 = {max_diff:.2e}")
    if max_diff > 1e-9:
        raise ValueError(f"v1 score reproduction failed with max diff {max_diff} > 1e-9!")
    print("SUCCESS: v1 score reproduces outputs/market_gap_scores.csv within 1e-9.")

    # 5. Compute v2 scores
    # v2 replaces degenerate distance with 1 - norm(share_of_locality)
    # and uses Bayesian shrunk rating for demand
    grouped["score_demand_v2"] = norm(grouped["avg_rating_adj"])
    grouped["score_room_to_breathe_v2"] = 1.0 - norm(grouped["share_of_locality"])

    grouped["opportunity_score_v2"] = (
        0.35 * grouped["score_low_competition"]
        + 0.35 * grouped["score_demand_v2"]
        + 0.15 * grouped["score_room_to_breathe_v2"]
        + 0.15 * grouped["score_price_gap"]
    )

    # Add coordinates
    def get_coords(loc: str):
        if loc in coords_df.index:
            row = coords_df.loc[loc]
            return float(row["lat"]), float(row["lon"])
        return 12.9716, 77.5946  # Bangalore center fallback

    coords_series = grouped["location"].apply(get_coords)
    grouped["lat"] = coords_series.apply(lambda c: c[0])
    grouped["lon"] = coords_series.apply(lambda c: c[1])

    # 6. Train Models & Persist
    print("\n--- Training Models & Checking Target Leakage ---")
    df_model = df_raw.copy()
    median_distance = float(df_model["nearest_competitor_km"].median())
    df_model["nearest_competitor_km"] = df_model["nearest_competitor_km"].fillna(median_distance)
    df_model["is_successful"] = (df_model["rating"] >= 4.0).astype(int)

    le_location = LabelEncoder()
    le_cuisine = LabelEncoder()
    le_resttype = LabelEncoder()
    df_model["location_enc"] = le_location.fit_transform(df_model["location"])
    df_model["cuisine_enc"] = le_cuisine.fit_transform(df_model["primary_cuisine"])
    df_model["rest_type_enc"] = le_resttype.fit_transform(df_model["rest_type"].fillna("Unknown"))

    features_with_votes = [
        "location_enc", "cuisine_enc", "rest_type_enc",
        "cost_for_two", "online_order", "book_table",
        "nearest_competitor_km", "votes"
    ]
    features_no_votes = [
        "location_enc", "cuisine_enc", "rest_type_enc",
        "cost_for_two", "online_order", "book_table",
        "nearest_competitor_km"
    ]

    X1 = df_model[features_with_votes]
    X2 = df_model[features_no_votes]
    y = df_model["is_successful"]

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y, test_size=0.2, random_state=42, stratify=y
    )
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Legacy Model (With Votes)
    m_with_votes = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    m_with_votes.fit(X1_train, y1_train)
    p1 = m_with_votes.predict(X1_test)
    prob1 = m_with_votes.predict_proba(X1_test)[:, 1]

    # Train Honest Zero-Day Model (Without Votes)
    m_no_votes = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    m_no_votes.fit(X2_train, y2_train)
    p2 = m_no_votes.predict(X2_test)
    prob2 = m_no_votes.predict_proba(X2_test)[:, 1]

    eval_with_votes = ModelEvaluation(
        name="RandomForest (with votes / legacy leakage)",
        description="Original model from pipeline. Target leakage via 'votes' (r~0.43 with rating) inflates accuracy to ~91%.",
        features=features_with_votes,
        accuracy=float(accuracy_score(y1_test, p1)),
        precision=float(precision_score(y1_test, p1)),
        recall=float(recall_score(y1_test, p1)),
        f1=float(f1_score(y1_test, p1)),
        roc_auc=float(roc_auc_score(y1_test, prob1)),
        feature_importance={
            feat: float(imp)
            for feat, imp in zip(features_with_votes, m_with_votes.feature_importances_)
        },
    )

    eval_no_votes = ModelEvaluation(
        name="RandomForest (zero-day / honest without votes)",
        description="Production model for prospective site selection. Dropping 'votes' provides realistic zero-day success probability without post-opening leakage.",
        features=features_no_votes,
        accuracy=float(accuracy_score(y2_test, p2)),
        precision=float(precision_score(y2_test, p2)),
        recall=float(recall_score(y2_test, p2)),
        f1=float(f1_score(y2_test, p2)),
        roc_auc=float(roc_auc_score(y2_test, prob2)),
        feature_importance={
            feat: float(imp)
            for feat, imp in zip(features_no_votes, m_no_votes.feature_importances_)
        },
    )

    print("Model Comparison:")
    print(f"  With Votes:    Acc={eval_with_votes.accuracy:.4f}, Prec={eval_with_votes.precision:.4f}, Rec={eval_with_votes.recall:.4f}, ROC-AUC={eval_with_votes.roc_auc:.4f}")
    print(f"  Without Votes: Acc={eval_no_votes.accuracy:.4f}, Prec={eval_no_votes.precision:.4f}, Rec={eval_no_votes.recall:.4f}, ROC-AUC={eval_no_votes.roc_auc:.4f}")

    # Save models and encoders
    joblib.dump(m_no_votes, DATA_DIR / "model_no_votes.joblib")
    joblib.dump(m_with_votes, DATA_DIR / "model_with_votes.joblib")
    joblib.dump({
        "location": le_location,
        "cuisine": le_cuisine,
        "rest_type": le_resttype
    }, DATA_DIR / "label_encoders.joblib")
    print(f"Saved models and encoders to {DATA_DIR}")

    # 7. Compute Global SHAP on No-Votes Model (2,000 points sample)
    print("\n--- Precomputing Global SHAP Data (2,000 samples) ---")
    explainer = shap.TreeExplainer(m_no_votes)
    # Persist explainer for fast single-row inference in API
    joblib.dump(explainer, DATA_DIR / "explainer_no_votes.joblib")

    sample_X2 = X2_test.sample(2000, random_state=42)
    shap_vals = explainer.shap_values(sample_X2)
    sv_pos = shap_vals[1] if isinstance(shap_vals, list) else (
        shap_vals[:, :, 1] if shap_vals.ndim == 3 else shap_vals
    )

    mean_abs_shap = {
        feat: float(np.mean(np.abs(sv_pos[:, idx])))
        for idx, feat in enumerate(features_no_votes)
    }

    display_name_map = {
        "location_enc": "Locality",
        "cuisine_enc": "Primary Cuisine",
        "rest_type_enc": "Restaurant Type",
        "cost_for_two": "Cost for Two (INR)",
        "online_order": "Online Ordering",
        "book_table": "Table Booking",
        "nearest_competitor_km": "Competitor Distance (km)"
    }

    sample_points_list: List[ShapSamplePoint] = []
    sample_df = sample_X2.reset_index(drop=True)
    for row_i in range(len(sample_df)):
        f_vals = {feat: float(sample_df.iloc[row_i][feat]) for feat in features_no_votes}
        s_vals = {feat: float(sv_pos[row_i, idx]) for idx, feat in enumerate(features_no_votes)}
        sample_points_list.append(ShapSamplePoint(feature_values=f_vals, shap_values=s_vals))

    shap_global_contract = ShapGlobal(
        feature_names=features_no_votes,
        feature_display_names=display_name_map,
        mean_abs_shap=mean_abs_shap,
        sample_points=sample_points_list,
    )

    # 8. Build Data Payloads
    print("\n--- Constructing Export Payloads ---")

    # A. opportunities.json
    opp_items: List[OpportunityItem] = []
    for row in grouped.itertuples():
        item_id = f"{row.location.lower().replace(' ', '_')}_{row.primary_cuisine.lower().replace(' ', '_')}"
        opp_items.append(
            OpportunityItem(
                id=item_id,
                location=row.location,
                primary_cuisine=row.primary_cuisine,
                restaurant_count=int(row.restaurant_count),
                locality_total=int(row.locality_total),
                share_of_locality=round(float(row.share_of_locality), 6),
                avg_rating=round(float(row.avg_rating), 3),
                avg_rating_adj=round(float(row.avg_rating_adj), 3),
                avg_cost=round(float(row.avg_cost), 1),
                avg_competitor_distance=0.0 if pd.isna(row.avg_competitor_distance) else round(float(row.avg_competitor_distance), 4),
                missing_price_tiers=int(row.missing_price_tiers),
                price_tiers_present=row.tiers_present,
                missing_tiers=row.missing_tiers,
                confidence=row.confidence,
                scores_v1=ScoreBreakdown(
                    score_low_competition=float(row.score_low_competition),
                    score_demand=float(row.score_demand_v1),
                    score_room_to_breathe=float(row.score_room_to_breathe_v1),
                    score_price_gap=float(row.score_price_gap),
                    opportunity_score=float(row.opportunity_score_v1),
                ),
                scores_v2=ScoreBreakdown(
                    score_low_competition=float(row.score_low_competition),
                    score_demand=float(row.score_demand_v2),
                    score_room_to_breathe=float(row.score_room_to_breathe_v2),
                    score_price_gap=float(row.score_price_gap),
                    opportunity_score=float(row.opportunity_score_v2),
                ),
                lat=round(float(row.lat), 6),
                lon=round(float(row.lon), 6),
            )
        )

    # B. localities.json
    loc_items: List[LocalityItem] = []
    for loc_name in sorted(valid_localities):
        sub = df[df["location"] == loc_name]
        coords = get_coords(loc_name)
        cuisine_counts = sub["primary_cuisine"].value_counts()
        total = len(sub)
        top_cuisines = [
            LocalityCuisineShare(
                cuisine=c,
                count=int(cnt),
                share=round(float(cnt / total), 4)
            )
            for c, cnt in cuisine_counts.head(8).items()
        ]
        loc_items.append(
            LocalityItem(
                locality=loc_name,
                lat=coords[0],
                lon=coords[1],
                total_restaurants=total,
                avg_rating=round(float(sub["rating"].mean()), 2),
                avg_cost=round(float(sub["cost_for_two"].mean()), 1),
                top_cuisines=top_cuisines,
            )
        )

    # C. restaurants.min.json (compact columnar arrays)
    restaurants_min = RestaurantsMin(
        count=len(df_raw),
        lat=[round(float(x), 5) for x in df_raw["lat"].tolist()],
        lon=[round(float(x), 5) for x in df_raw["lon"].tolist()],
        name=df_raw["name"].fillna("Unknown").tolist(),
        location=df_raw["location"].fillna("Unknown").tolist(),
        cuisine=df_raw["primary_cuisine"].fillna("Unknown").tolist(),
        rating=[round(float(x), 2) for x in df_raw["rating"].fillna(0.0).tolist()],
        cost=[round(float(x), 1) for x in df_raw["cost_for_two"].fillna(0.0).tolist()],
        online_order=df_raw["online_order"].astype(int).tolist(),
        book_table=df_raw["book_table"].astype(int).tolist(),
    )

    # D. cuisine_matrix.json
    all_cuisines = sorted(sat_df["primary_cuisine"].unique().tolist())
    all_localities_sat = sorted(sat_df["location"].unique().tolist())
    sat_pivot_counts = sat_df.pivot(index="location", columns="primary_cuisine", values="restaurant_count").fillna(0).astype(int)
    sat_pivot_shares = sat_df.pivot(index="location", columns="primary_cuisine", values="share_of_locality").fillna(0.0)
    loc_totals = sat_df.groupby("location")["locality_total"].first().to_dict()

    matrix_contract = CuisineMatrix(
        cuisines=all_cuisines,
        localities=all_localities_sat,
        locality_totals={k: int(v) for k, v in loc_totals.items()},
        counts={loc: {c: int(sat_pivot_counts.loc[loc, c]) for c in all_cuisines if c in sat_pivot_counts.columns} for loc in all_localities_sat},
        shares={loc: {c: round(float(sat_pivot_shares.loc[loc, c]), 5) for c in all_cuisines if c in sat_pivot_shares.columns} for loc in all_localities_sat},
    )

    # E. model_meta.json
    model_meta_contract = ModelMeta(
        target_threshold=4.0,
        models={
            "with_votes": eval_with_votes,
            "no_votes": eval_no_votes,
        },
        categorical_options={
            "locations": sorted(df_raw["location"].dropna().unique().tolist()),
            "cuisines": sorted(df_raw["primary_cuisine"].dropna().unique().tolist()),
            "rest_types": sorted(df_raw["rest_type"].dropna().unique().tolist()),
        },
        median_defaults={
            "cost_for_two": float(df_raw["cost_for_two"].median()),
            "nearest_competitor_km": median_distance,
        }
    )

    # 9. Validate with Pydantic & Write JSONs
    print("\n--- Validating Payloads with Pydantic Schema ---")
    # Pydantic instances are already validated through instantiation!
    # Double check dumpability:
    files_to_export = [
        ("opportunities.json", [item.model_dump() for item in opp_items]),
        ("localities.json", [item.model_dump() for item in loc_items]),
        ("restaurants.min.json", restaurants_min.model_dump()),
        ("cuisine_matrix.json", matrix_contract.model_dump()),
        ("model_meta.json", model_meta_contract.model_dump()),
        ("shap_global.json", shap_global_contract.model_dump()),
    ]

    for dest_dir in [OUTPUTS_WEB_DIR, WEB_DATA_DIR]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for filename, data_dict in files_to_export:
            out_file = dest_dir / filename
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, separators=(",", ":"))
            print(f"Exported {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 60)
    print("PHASE 0 EXPORT COMPLETED SUCCESSFULLY WITH STRICT VALIDATION")
    print("=" * 60)


if __name__ == "__main__":
    run_export()
