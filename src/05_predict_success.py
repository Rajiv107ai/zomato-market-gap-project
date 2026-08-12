"""
Phase 5: Predict whether a restaurant will be "successful" (rating >= 4.0)
based on location, cuisine, price, and competition features — then explain
the model with SHAP so we can say *why*, not just *what*.

Input:  data/zomato_with_competition.csv
Output: outputs/feature_importance.png
        outputs/shap_summary.png
        Prints model accuracy + top features to console
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

IN_PATH = "data/zomato_with_competition.csv"
FEATURE_IMPORTANCE_OUT = "outputs/feature_importance.png"
SHAP_SUMMARY_OUT = "outputs/shap_summary.png"

SUCCESS_THRESHOLD = 4.0


def main():
    df = pd.read_csv(IN_PATH)
    df["nearest_competitor_km"] = df["nearest_competitor_km"].fillna(
        df["nearest_competitor_km"].median()
    )

    # Target: is this restaurant "successful" (rating >= threshold)?
    df["is_successful"] = (df["rating"] >= SUCCESS_THRESHOLD).astype(int)

    # Encode categoricals
    le_location = LabelEncoder()
    le_cuisine = LabelEncoder()
    le_resttype = LabelEncoder()
    df["location_enc"] = le_location.fit_transform(df["location"])
    df["cuisine_enc"] = le_cuisine.fit_transform(df["primary_cuisine"])
    df["rest_type_enc"] = le_resttype.fit_transform(df["rest_type"].fillna("Unknown"))

    features = [
        "location_enc", "cuisine_enc", "rest_type_enc",
        "cost_for_two", "online_order", "book_table",
        "nearest_competitor_km", "votes",
    ]
    X = df[features]
    y = df["is_successful"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    # Plain feature importance (quick sanity check)
    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    importances.plot(kind="barh", figsize=(8, 5), title="Feature Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_OUT)
    plt.close()
    print(f"Saved -> {FEATURE_IMPORTANCE_OUT}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle both old and new SHAP output formats
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

    shap.summary_plot(sv, X_test, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_OUT)
    plt.close()
    print(f"Saved -> {SHAP_SUMMARY_OUT}")


if __name__ == "__main__":
    main()
