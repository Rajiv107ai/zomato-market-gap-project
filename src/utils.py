"""Shared helpers for the Zomato market-gap project."""
import pandas as pd
import numpy as np


def parse_rate(value):
    """Convert '4.1/5' -> 4.1, handle 'NEW', '-', NaN."""
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value in ("NEW", "-", ""):
        return np.nan
    try:
        return float(value.split("/")[0])
    except (ValueError, IndexError):
        return np.nan


def parse_cost(value):
    """Convert '1,200' -> 1200.0"""
    if pd.isna(value):
        return np.nan
    value = str(value).replace(",", "").strip()
    try:
        return float(value)
    except ValueError:
        return np.nan


def split_cuisines(value):
    """'North Indian, Chinese' -> ['North Indian', 'Chinese']"""
    if pd.isna(value):
        return []
    return [c.strip() for c in str(value).split(",") if c.strip()]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in km (vectorized)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))
