"""
Phase 2: Geocode each unique locality name into latitude/longitude.

This is the step that turns a plain-text 'location' column into something
we can actually map. Uses Nominatim (OpenStreetMap) via geopy, with a
1-request/sec rate limit (their usage policy) and local caching so we
only ever do this once.

Input:  data/zomato_clean.csv
Output: data/locality_coords.csv   (cache: locality -> lat, lon)
        data/zomato_geo.csv        (clean data merged with coordinates)
"""
import pandas as pd
import time
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

CLEAN_PATH = "data/zomato_clean.csv"
CACHE_PATH = "data/locality_coords.csv"
OUT_PATH = "data/zomato_geo.csv"


def geocode_localities(localities):
    geolocator = Nominatim(user_agent="zomato_market_gap_project")
    results = []
    for loc in localities:
        query = f"{loc}, Bangalore, India"
        lat, lon = None, None
        for attempt in range(3):
            try:
                location = geolocator.geocode(query, timeout=10)
                if location:
                    lat, lon = location.latitude, location.longitude
                break
            except GeocoderTimedOut:
                time.sleep(2)
                continue
        results.append({"location": loc, "lat": lat, "lon": lon})
        print(f"  {loc:30s} -> {lat}, {lon}")
        time.sleep(1)  # respect Nominatim's 1 req/sec rate limit
    return pd.DataFrame(results)


def main():
    df = pd.read_csv(CLEAN_PATH)
    unique_localities = sorted(df["location"].unique())
    print(f"Found {len(unique_localities)} unique localities to geocode.")

    if os.path.exists(CACHE_PATH):
        print("Cache found — loading existing coordinates.")
        coords = pd.read_csv(CACHE_PATH)
        already_done = set(coords["location"])
        remaining = [l for l in unique_localities if l not in already_done]
    else:
        coords = pd.DataFrame(columns=["location", "lat", "lon"])
        remaining = unique_localities

    if remaining:
        print(f"Geocoding {len(remaining)} new localities (~1 sec each)...")
        new_coords = geocode_localities(remaining)
        coords = pd.concat([coords, new_coords], ignore_index=True)
        coords.to_csv(CACHE_PATH, index=False)

    # Merge coordinates back into the main dataset
    merged = df.merge(coords, on="location", how="left")
    before = len(merged)
    merged = merged.dropna(subset=["lat", "lon"])
    print(f"Dropped {before - len(merged)} rows with failed geocoding.")

    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
