"""
Requirements:
    pip install requests

Usage:
    python fetch_osm_data.py
"""

import requests
import json
import time


def fetch_restaurants_osm(bbox, timeout=180):
    """
    Fetch restaurant locations from OpenStreetMap using Overpass API.
    
    Args:
        bbox: tuple of (min_lat, min_lon, max_lat, max_lon)
        timeout: API timeout in seconds
    
    Returns:
        list of (lon, lat) tuples
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    # Overpass API query for restaurants
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:{timeout}];
    (
      node["amenity"="restaurant"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["amenity"="restaurant"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["amenity"="restaurant"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center;
    """
    
    print(f"Fetching restaurants from OpenStreetMap...")
    print(f"Bounding box: lat [{min_lat}, {max_lat}], lon [{min_lon}, {max_lon}]")
    print(f"30-60 seconds")
    
    try:
        response = requests.post(overpass_url, data={"data": overpass_query}, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        restaurants = []
        for element in data['elements']:
            if element['type'] == 'node':
                lat = element['lat']
                lon = element['lon']
            elif 'center' in element:
                lat = element['center']['lat']
                lon = element['center']['lon']
            else:
                continue
            
            # extract name and other metadata
            name = element.get('tags', {}).get('name', 'Unknown')
            restaurants.append({
                'lon': lon,
                'lat': lat,
                'name': name
            })
        
        print(f" Found {len(restaurants)} restaurants")
        return restaurants
    
    # Error handling
    except requests.exceptions.Timeout:
        print(f" Request timed out after {timeout} seconds")
        print("  Try reducing the bounding box size or increasing timeout")
        return []
    except requests.exceptions.RequestException as e:
        print(f" Error fetching data: {e}")
        return []

# Write info to JSON file
def save_to_file(restaurants, filename='atlanta_restaurants.json'):
    """Save restaurant data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(restaurants, f, indent=2)
    print(f" Saved {len(restaurants)} restaurants to {filename}")


def main():
    print("="*60)
    print("OpenStreetMap Restaurant Data Fetcher")
    print("="*60)
    
    # Atlanta bounding box - metro area
    atlanta_bbox = (
        33.6490,  # min_lat (south) - Hartsfield-Jackson Airport
        -84.5510,  # min_lon (west) - Six Flags
        33.8860,  # max_lat (north) - Roswell
        -84.2890   # max_lon (east) - Stone Mountain
    )
    
    print(f"\nArea: Atlanta Metro")
    print(f"Coverage: ~26km x ~24km")
    print(f"Expected restaurants: 1000-3000\n")
    
    restaurants = fetch_restaurants_osm(atlanta_bbox)
    
    if restaurants:
        save_to_file(restaurants, 'atlanta_restaurants_osm.json')
        
        print(f"\n{'='*60}")
        print("Success! Data ready for clustering.")
        print("="*60)
        
        print("\nFirst 5 restaurants:")
        for i, r in enumerate(restaurants[:5], 1):
            print(f"  {i}. {r['name']} ({r['lat']:.4f}, {r['lon']:.4f})")
    else:
        print("\nFailed to fetch data.")


if __name__ == '__main__':
    main()