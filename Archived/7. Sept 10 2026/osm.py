"""
osm.py
======
Fetches restaurant POI data from OpenStreetMap via the Overpass API
and saves each city to a JSON file for use in experiments.py.

Usage:
    python osm.py                  # fetch all three cities
    python osm.py --city atlanta   # fetch one city only

Dependencies: requests
"""

import requests
import json
import time
import argparse
import os


# ── Bounding boxes (min_lat, min_lon, max_lat, max_lon) ──────────────────────
CITIES = {
    "atlanta": {
        "bbox":        (33.6490, -84.5510, 33.8860, -84.2890),
        "description": "Atlanta Metro (~26km × 24km)",
        "output":      "atlanta_restaurants_osm.json",
    },
    "nyc": {
        "bbox":        (40.4774, -74.2591, 40.9176, -73.7004),
        "description": "New York City (~58km × 49km)",
        "output":      "nyc_restaurants_osm.json",
    },
    "chicago": {
        "bbox":        (41.6445, -87.9401, 42.0230, -87.5240),
        "description": "Chicago Metro (~42km × 46km)",
        "output":      "chicago_restaurants_osm.json",
    },
}


def fetch_restaurants_osm(bbox, timeout=180):
    """
    Fetch restaurant locations from OpenStreetMap using the Overpass API.

    Args:
        bbox:    (min_lat, min_lon, max_lat, max_lon)
        timeout: HTTP timeout in seconds

    Returns:
        list of dicts with keys: lon, lat, name
    """
    min_lat, min_lon, max_lat, max_lon = bbox

    overpass_url   = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:{timeout}];
    (
      node["amenity"="restaurant"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["amenity"="restaurant"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["amenity"="restaurant"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center;
    """

    print(f"  Querying Overpass API … (may take 30–90 s)")
    try:
        response = requests.post(
            overpass_url,
            data={"data": overpass_query},
            headers={
                "User-Agent": "osm-restaurant-fetcher/1.0 (research project)",
                "Accept": "*/*",
            },
            timeout=timeout + 30,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        print(f"  ✗ Request timed out after {timeout}s. Try a smaller bbox.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error: {e}")
        return []

    restaurants = []
    for el in data["elements"]:
        if el["type"] == "node":
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        name = el.get("tags", {}).get("name", "Unknown")
        restaurants.append({"lon": lon, "lat": lat, "name": name})

    print(f"  ✓ Found {len(restaurants)} restaurants")
    return restaurants


def save_json(restaurants, filepath):
    with open(filepath, "w") as f:
        json.dump(restaurants, f, indent=2)
    print(f"  ✓ Saved to {filepath}")


def fetch_city(city_key):
    city = CITIES[city_key]
    print(f"\n{'='*60}")
    print(f"City: {city['description']}")
    print(f"Output: {city['output']}")
    print(f"{'='*60}")

    if os.path.exists(city["output"]):
        print(f"  File already exists — skipping download.")
        print(f"  Delete {city['output']} to re-fetch.")
        return

    restaurants = fetch_restaurants_osm(city["bbox"])
    if restaurants:
        save_json(restaurants, city["output"])
        print(f"\n  First 3 entries:")
        for r in restaurants[:3]:
            print(f"    {r['name']}  ({r['lat']:.4f}, {r['lon']:.4f})")
    else:
        print(f"  ✗ No data fetched for {city_key}.")


def main():
    parser = argparse.ArgumentParser(description="Fetch OSM restaurant data")
    parser.add_argument(
        "--city",
        choices=list(CITIES.keys()) + ["all"],
        default="all",
        help="Which city to fetch (default: all)",
    )
    args = parser.parse_args()

    cities_to_fetch = list(CITIES.keys()) if args.city == "all" else [args.city]

    print("=" * 60)
    print("OpenStreetMap Restaurant Fetcher")
    print("=" * 60)

    for city_key in cities_to_fetch:
        fetch_city(city_key)
        if city_key != cities_to_fetch[-1]:
            print("  Waiting 5 s between requests …")
            time.sleep(5)

    print(f"\n{'='*60}")
    print("Done. JSON files are ready for experiments.py.")
    print("=" * 60)


if __name__ == "__main__":
    main()