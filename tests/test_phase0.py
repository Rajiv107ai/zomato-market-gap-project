import json
from pathlib import Path
import importlib
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
WEB_DATA_DIR = ROOT_DIR / "web" / "public" / "data"

export_module = importlib.import_module("src.06_export_for_web")
ModelMeta = export_module.ModelMeta


def test_v1_reproducibility():
    """Verify v1 score exactly reproduces outputs/market_gap_scores.csv within 1e-9."""
    orig = pd.read_csv(OUTPUTS_DIR / "market_gap_scores.csv")
    with open(WEB_DATA_DIR / "opportunities.json", "r", encoding="utf-8") as f:
        exported_data = json.load(f)
    
    exported_df = pd.DataFrame([
        {
            "location": item["location"],
            "primary_cuisine": item["primary_cuisine"],
            "opportunity_score_v1": item["scores_v1"]["opportunity_score"],
        }
        for item in exported_data
    ])
    
    merged = pd.merge(exported_df, orig, on=["location", "primary_cuisine"])
    assert len(merged) == len(orig)
    diff = np.abs(merged["opportunity_score_v1"] - merged["opportunity_score"]).max()
    assert diff < 1e-9, f"Max diff {diff} exceeds 1e-9"


def test_v2_room_to_breathe_and_confidence():
    """Verify that v2 does not suffer from 0-km degeneration and has confidence tiers."""
    with open(WEB_DATA_DIR / "opportunities.json", "r", encoding="utf-8") as f:
        items = json.load(f)

    v2_breathe = [it["scores_v2"]["score_room_to_breathe"] for it in items]
    confidences = [it["confidence"] for it in items]

    # Non-degenerate distribution
    assert np.mean(v2_breathe) > 0.5
    assert set(confidences) == {"high", "medium", "low"}

    # Verify shrinkage: for n=1, adjusted rating must be closer to global mean
    n1_items = [it for it in items if it["restaurant_count"] == 1]
    assert len(n1_items) > 0
    for it in n1_items[:10]:
        # If raw rating is high (>4.0), adj must be smaller
        if it["avg_rating"] > 3.8:
            assert it["avg_rating_adj"] < it["avg_rating"]


def test_exported_json_contracts():
    """Verify that all exported web data files exist and parse with Pydantic contracts."""
    files_to_check = [
        ("localities.json", list),
        ("opportunities.json", list),
        ("restaurants.min.json", dict),
        ("cuisine_matrix.json", dict),
        ("model_meta.json", dict),
        ("shap_global.json", dict),
    ]

    for filename, expected_type in files_to_check:
        filepath = WEB_DATA_DIR / filename
        assert filepath.exists(), f"Missing file: {filepath}"
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, expected_type)

    # Validate model_meta
    with open(WEB_DATA_DIR / "model_meta.json", "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    meta = ModelMeta(**meta_dict)
    assert "no_votes" in meta.models
    assert "with_votes" in meta.models
    assert meta.models["no_votes"].accuracy > 0.80
    assert meta.models["no_votes"].roc_auc > 0.85


def test_saved_model_inference():
    """Verify joblib model and explainer can be loaded and run single-sample inference."""
    model = joblib.load(DATA_DIR / "model_no_votes.joblib")
    encoders = joblib.load(DATA_DIR / "label_encoders.joblib")
    explainer = joblib.load(DATA_DIR / "explainer_no_votes.joblib")

    loc_enc = encoders["location"].transform(["Koramangala 5th Block"])[0]
    cui_enc = encoders["cuisine"].transform(["North Indian"])[0]
    rest_enc = encoders["rest_type"].transform(["Casual Dining"])[0]

    sample_features = np.array([[loc_enc, cui_enc, rest_enc, 600.0, 1, 0, 0.5]])
    prob = model.predict_proba(sample_features)[0, 1]
    assert 0.0 <= prob <= 1.0

    shap_val = explainer.shap_values(sample_features)
    assert shap_val is not None
