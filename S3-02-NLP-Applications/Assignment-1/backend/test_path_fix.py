#!/usr/bin/env python3
"""
Test script to verify path finding fix
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_path(source, destination, path_type="shortest_distance"):
    """Test path finding between two stations"""
    url = f"{BASE_URL}/paths?source={source}&destination={destination}&type={path_type}"

    print(f"\n{'='*70}")
    print(f"Testing: {source} → {destination}")
    print(f"Type: {path_type}")
    print(f"{'='*70}")

    try:
        response = requests.get(url)
        data = response.json()

        if "error" in data:
            print(f"❌ Error: {data['error']}")
            return

        print(f"✓ Path found!")
        print(f"  Length: {data['length']} stops")
        print(f"  Total Distance: {data.get('total_distance_km', 'N/A')} km")
        print(f"  Line Changes: {data.get('line_changes', 'N/A')}")

        # Print the path
        print(f"\n  Path ({len(data['path'])} stations):")
        for i, station in enumerate(data['path']):
            print(f"    {i+1}. {station}")

        # Print route details
        if 'route_details' in data:
            print(f"\n  Route Details:")
            for segment in data['route_details']:
                print(f"    {segment['from']}")
                print(f"      → {segment['to']} ({segment['line']}, {segment['distance_km']} km)")

        return data

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("\nTesting Path Finding Bug Fix")
    print("="*70)

    # Test Case 1: Hindon River to Major Mohit Sharma (should be 4 stops on Red line)
    test_path("Hindon River", "Major Mohit Sharma", "shortest_distance")

    # Test Case 2: Same path with shortest_path (by stops)
    test_path("Hindon River", "Major Mohit Sharma", "shortest_path")

    # Test Case 3: Longer path with potential line changes
    test_path("Rajiv Chowk", "Kashmere Gate", "shortest_distance")

    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)
