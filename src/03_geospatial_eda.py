"""
Phase 3: Geospatial EDA 

Produces:
- An interactive Folium heatmap of restaurant density across Bangalore
- A cuisine-saturation table (which localities are oversaturated for which cuisine)
- Nearest-competitor distance per restaurant (same primary cuisine)

Input:  data/zomato_geo.csv
Output: outputs/density_heatmap.html
        outputs/cuisine_saturation.csv
        data/zomato_with_competition.csv
"""
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from utils import haversine_km

GEO_PATH = "data/zomato_geo.csv"
HEATMAP_OUT = "outputs/density_heatmap.html"
SATURATION_OUT = "outputs/cuisine_saturation.csv"
COMPETITION_OUT = "data/zomato_with_competition.csv"


def build_heatmap(df):
    center = [df["lat"].mean(), df["lon"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")
    heat_data = df[["lat", "lon"]].values.tolist()
    HeatMap(heat_data, radius=12, blur=15).add_to(m)
    m.save(HEATMAP_OUT)
    print(f"Saved heatmap -> {HEATMAP_OUT}")


def cuisine_saturation(df):
    """Count restaurants per locality x primary_cuisine, plus locality totals."""
    sat = (
        df.groupby(["location", "primary_cuisine"])
        .size()
        .reset_index(name="restaurant_count")
    )
    locality_totals = df.groupby("location").size().rename("locality_total")
    sat = sat.merge(locality_totals, on="location")
    sat["share_of_locality"] = sat["restaurant_count"] / sat["locality_total"]
    sat = sat.sort_values("restaurant_count", ascending=False)
    sat.to_csv(SATURATION_OUT, index=False)
    print(f"Saved cuisine saturation table -> {SATURATION_OUT}")
    return sat


def nearest_competitor_distance(df):
    """
    For each restaurant, find the distance (km) to the nearest other
    restaurant serving the same primary cuisine. This is O(n^2) within
    each cuisine group.
    """
    df = df.copy()
    df["nearest_competitor_km"] = np.nan

    for cuisine, group in df.groupby("primary_cuisine"):
        if len(group) < 2:
            continue
        lats = group["lat"].values
        lons = group["lon"].values
        idx = group.index.values

        for i in range(len(group)):
            dists = haversine_km(lats[i], lons[i], lats, lons)
            dists[i] = np.inf  # exclude self
            df.loc[idx[i], "nearest_competitor_km"] = dists.min()

    return df


def main():
    df = pd.read_csv(GEO_PATH)

    print("Building density heatmap...")
    build_heatmap(df)

    print("Computing cuisine saturation...")
    cuisine_saturation(df)

    print("Computing nearest-competitor distance (this may take a minute)...")
    df = nearest_competitor_distance(df)
    df.to_csv(COMPETITION_OUT, index=False)
    print(f"Saved -> {COMPETITION_OUT}")


if __name__ == "__main__":
    main()
