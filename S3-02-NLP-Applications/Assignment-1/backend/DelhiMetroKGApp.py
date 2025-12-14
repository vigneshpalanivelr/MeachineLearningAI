"""
  Example API calls to test:

  # Welcome page (help)
  http://127.0.0.1:5000/

  # Get all stations and connections
  http://127.0.0.1:5000/full_graph

  # Search for stations
  curl "http://127.0.0.1:5000/search_station?q=Kashmere"

  # Get neighbors of a specific station (fuzzy matching)
  curl "http://127.0.0.1:5000/neighbors/Kashmere%20Gate"

  # Find shortest path (fuzzy matching)
  curl "http://127.0.0.1:5000/shortest_path?source=Rajiv%20Chowk&target=Kashmere%20Gate"

  # Upload CSV
  curl -X POST -F "file=@test_generic_relationships.csv" http://127.0.0.1:5000/upload_csv
  {
  "message": "CSV uploaded successfully",
  "relationships_added": 10,
  "total_connections": 294,
  "total_stations": 289
  }
  curl -X POST -F "file=@test_metro_relationships.csv" http://127.0.0.1:5000/upload_csv
  {
  "message": "CSV uploaded successfully",
  "relationships_added": 10,
  "total_connections": 303,
  "total_stations": 299
  }

  # Smart Error Handling: When station not found, API returns suggestions:
  {
    "error": "Station not found",
    "query": "invalid station",
    "suggestions": ["Station1", "Station2", "Station3"]
  }

  # Generated Test Files
  1. test_metro_relationships.csv (Metro-Specific Format)
  Contains 10 relationships:
  - New Delhi Station → Connaught Place → Aerocity → Terminal 3 Airport
  - Gurgaon Sector 21 → Multiple Cyber City phases
  - Noida City Center → Multiple Noida sectors

  Format: source,target,line,distance

  2. test_generic_relationships.csv (Generic Knowledge Graph Format)
  Contains 10 entity relationships:
  - Delhi metro lines and their connections
  - Interchange points at major stations

  Format: entity1,relationship,entity2
"""
from flask import Flask, request, jsonify
import networkx as nx
import pandas as pd
from difflib import get_close_matches
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load Delhi Metro Single-Line Dataset
KG = nx.Graph()   # Metro network is undirected

def load_delhi_metro_graph(csv_path="delhi_metro.csv"):
    df = pd.read_csv(csv_path)

    # Rename columns for consistency (optional)
    df.columns = ["ID", "Station", "Distance", "Line", "Opened", "Layout", "Latitude", "Longitude"]

    # Sort by ID (ensures correct station order)
    df = df.sort_values("ID")

    # Create nodes with attributes
    for _, row in df.iterrows():
        KG.add_node(
            row["Station"],
            station_id=row["ID"],
            line=row["Line"],
            opened=row["Opened"],
            layout=row["Layout"],
            latitude=row["Latitude"],
            longitude=row["Longitude"],
            distance_from_start=row["Distance"]
        )

    # Create edges between consecutive stations
    for i in range(len(df) - 1):
        s1 = df.iloc[i]
        s2 = df.iloc[i + 1]

        station1 = s1["Station"]
        station2 = s2["Station"]

        # Distance between two stations = difference in cumulative distance
        distance_km = abs(s2["Distance"] - s1["Distance"])

        KG.add_edge(station1, station2, line=s1["Line"], distance=distance_km)

    print(f"Graph Loaded: {KG.number_of_nodes()} stations, {KG.number_of_edges()} edges")


# Load graph when server starts
load_delhi_metro_graph()


# Helper function for fuzzy station matching
def find_station(query):
    """
    Find station using improved fuzzy matching.
    Returns (matched_station, confidence) or (None, 0) if no match found.

    Matching strategy:
    1. Exact match (case-sensitive)
    2. Case-insensitive exact match
    3. Case-insensitive substring match
    4. Fuzzy match with higher threshold (0.6)
    """
    if not query:
        return None, 0

    # 1. Exact match (case-sensitive)
    if query in KG:
        return query, 1.0

    query_lower = query.lower()

    # 2. Case-insensitive exact match
    for station in KG.nodes():
        if station.lower() == query_lower:
            return station, 1.0

    # 3. Case-insensitive substring match (prioritize this over fuzzy)
    substring_matches = []
    for station in KG.nodes():
        station_lower = station.lower()
        # Check if query is in station name or vice versa
        if query_lower in station_lower or station_lower in query_lower:
            # Prefer matches where query is at the start
            if station_lower.startswith(query_lower):
                substring_matches.insert(0, station)
            else:
                substring_matches.append(station)

    if substring_matches:
        # Return the best substring match (preferring those that start with query)
        return substring_matches[0], 0.9

    # 4. Fuzzy match with higher cutoff (0.6 instead of 0.4 for better precision)
    matches = get_close_matches(query, KG.nodes(), n=1, cutoff=0.6)
    if matches:
        return matches[0], 0.7

    return None, 0


# API: Home/Welcome Page
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Delhi Metro Knowledge Graph API",
        "total_stations": KG.number_of_nodes(),
        "total_connections": KG.number_of_edges(),
        "endpoints": {
            "GET /": "This help message",
            "GET /search_station?q=<query>": "Search for stations (supports fuzzy matching)",
            "GET /neighbors/<station>": "Get neighboring stations (supports fuzzy matching)",
            "GET /shortest_path?source=<s1>&target=<s2>": "Find shortest path by stops (supports fuzzy matching)",
            "GET /shortest_distance?source=<s1>&target=<s2>": "Find shortest path by distance (supports fuzzy matching)",
            "GET /full_graph": "Get complete graph data",
            "POST /add_relationship": "Add new station connection",
            "POST /upload_csv": "Bulk upload relationships from CSV file"
        }
    })


# API: Search for stations
@app.route("/search_station", methods=["GET"])
def search_station():
    """
    Search for stations using fuzzy matching.
    Query param: q (query string)
    Returns: List of matching stations with confidence scores
    """
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    # Get multiple matches for better UX
    matches = get_close_matches(query, KG.nodes(), n=5, cutoff=0.3)

    results = []
    for match in matches:
        node_data = KG.nodes[match]
        results.append({
            "station": match,
            "line": node_data.get("line"),
            "latitude": node_data.get("latitude"),
            "longitude": node_data.get("longitude")
        })

    return jsonify({
        "query": query,
        "matches": results,
        "total": len(results)
    })


# API: Add Manual Relationship
@app.route("/add_relationship", methods=["POST"])
def add_relationship():
    data = request.get_json()

    s1 = data.get("source")
    s2 = data.get("target")
    line = data.get("line", "UnknownLine")
    distance = float(data.get("distance", 0.0))

    if not s1 or not s2:
        return jsonify({"error": "source and target required"}), 400

    KG.add_edge(s1, s2, line=line, distance=distance)
    return jsonify({"message": "Relationship added"})


# API: Upload CSV for Bulk Relationships
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    """
    Upload CSV file to bulk add relationships.
    Expected CSV format:
    - Option 1: source,target,line,distance
    - Option 2: source,relationship,target (generic format)
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be a CSV"}), 400

    try:
        # Read CSV file
        csv_data = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))

        # Check CSV format and process accordingly
        added_count = 0

        # Format 1: source, target, line, distance (metro-specific)
        if all(col in df.columns for col in ['source', 'target']):
            for _, row in df.iterrows():
                source = row.get('source')
                target = row.get('target')
                line = row.get('line', 'Unknown')
                distance = float(row.get('distance', 0.0))

                if pd.notna(source) and pd.notna(target):
                    # Add nodes if they don't exist
                    if source not in KG:
                        KG.add_node(source)
                    if target not in KG:
                        KG.add_node(target)

                    # Add edge
                    KG.add_edge(source, target, line=line, distance=distance)
                    added_count += 1

        # Format 2: entity1, relationship, entity2 (generic format)
        elif all(col in df.columns for col in ['entity1', 'relationship', 'entity2']):
            for _, row in df.iterrows():
                entity1 = row.get('entity1')
                relationship = row.get('relationship')
                entity2 = row.get('entity2')

                if pd.notna(entity1) and pd.notna(entity2):
                    # Add nodes if they don't exist
                    if entity1 not in KG:
                        KG.add_node(entity1)
                    if entity2 not in KG:
                        KG.add_node(entity2)

                    # Add edge with relationship as attribute
                    KG.add_edge(entity1, entity2, relationship=relationship)
                    added_count += 1

        else:
            return jsonify({
                "error": "Invalid CSV format",
                "expected_formats": [
                    "Format 1: source,target,line,distance",
                    "Format 2: entity1,relationship,entity2"
                ]
            }), 400

        return jsonify({
            "message": "CSV uploaded successfully",
            "relationships_added": added_count,
            "total_stations": KG.number_of_nodes(),
            "total_connections": KG.number_of_edges()
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process CSV: {str(e)}"}), 500


# API: Get Neighbors of Station
@app.route("/neighbors/<station>", methods=["GET"])
def neighbors(station):
    # Try fuzzy matching
    matched_station, confidence = find_station(station)

    if not matched_station:
        # Provide suggestions
        suggestions = get_close_matches(station, KG.nodes(), n=3, cutoff=0.3)
        return jsonify({
            "error": "Station not found",
            "query": station,
            "suggestions": suggestions
        }), 404

    result = []
    for neigh in KG.neighbors(matched_station):
        edge = KG.get_edge_data(matched_station, neigh)
        result.append({"station": neigh, "line": edge.get("line"), "distance_km": edge.get("distance")})

    return jsonify({
        "query": station,
        "matched_station": matched_station,
        "fuzzy_match": confidence < 1.0,
        "neighbors": result
    })


# API: Shortest Path (Fewest Stops)
@app.route("/shortest_path", methods=["GET"])
def shortest_path():
    s1 = request.args.get("source")
    s2 = request.args.get("target")

    # Fuzzy match both stations
    matched_s1, conf1 = find_station(s1)
    matched_s2, conf2 = find_station(s2)

    if not matched_s1 or not matched_s2:
        errors = {}
        if not matched_s1:
            errors["source"] = {
                "query": s1,
                "suggestions": get_close_matches(s1, KG.nodes(), n=3, cutoff=0.3)
            }
        if not matched_s2:
            errors["target"] = {
                "query": s2,
                "suggestions": get_close_matches(s2, KG.nodes(), n=3, cutoff=0.3)
            }
        return jsonify({"error": "Stations not found", "details": errors}), 404

    try:
        path = nx.shortest_path(KG, matched_s1, matched_s2)
        return jsonify({
            "query": {"source": s1, "target": s2},
            "matched": {"source": matched_s1, "target": matched_s2},
            "fuzzy_match": conf1 < 1.0 or conf2 < 1.0,
            "shortest_path": path,
            "stops": len(path) - 1
        })
    except nx.NetworkXNoPath:
        return jsonify({"error": "No path found"}), 404


# API: Shortest Path by Distance (Dijkstra)
@app.route("/shortest_distance", methods=["GET"])
def shortest_distance():
    s1 = request.args.get("source")
    s2 = request.args.get("target")

    # Fuzzy match both stations
    matched_s1, conf1 = find_station(s1)
    matched_s2, conf2 = find_station(s2)

    if not matched_s1 or not matched_s2:
        errors = {}
        if not matched_s1:
            errors["source"] = {
                "query": s1,
                "suggestions": get_close_matches(s1, KG.nodes(), n=3, cutoff=0.3)
            }
        if not matched_s2:
            errors["target"] = {
                "query": s2,
                "suggestions": get_close_matches(s2, KG.nodes(), n=3, cutoff=0.3)
            }
        return jsonify({"error": "Stations not found", "details": errors}), 404

    try:
        path = nx.shortest_path(KG, matched_s1, matched_s2, weight="distance")
        total = nx.shortest_path_length(KG, matched_s1, matched_s2, weight="distance")
        return jsonify({
            "query": {"source": s1, "target": s2},
            "matched": {"source": matched_s1, "target": matched_s2},
            "fuzzy_match": conf1 < 1.0 or conf2 < 1.0,
            "shortest_path": path,
            "total_distance_km": round(total, 2),
            "stops": len(path) - 1
        })
    except nx.NetworkXNoPath:
        return jsonify({"error": "No path found"}), 404


# API: Full Graph
@app.route("/full_graph", methods=["GET"])
def full_graph():
    nodes = [{
        "station": n,
        "attributes": KG.nodes[n]
    } for n in KG.nodes()]

    edges = [{
        "source": u,
        "target": v,
        "line": d.get("line"),
        "distance_km": d.get("distance")
    } for u, v, d in KG.edges(data=True)]

    return jsonify({"nodes": nodes, "edges": edges})


# ==================== FRONTEND API ENDPOINTS ====================
# These endpoints are specifically for the React frontend

# API: Get Graph Data (for visualization)
@app.route("/api/graph", methods=["GET"])
def get_graph():
    """
    Get graph data in format expected by vis-network.
    Supports optional subgraph query by node and radius.
    """
    node = request.args.get("node")
    radius = request.args.get("radius", type=int)

    if node and radius:
        # Return subgraph within radius
        try:
            # Find station with fuzzy matching
            matched_node, _ = find_station(node)
            if not matched_node:
                return jsonify({"nodes": [], "edges": [], "error": "Station not found"}), 404

            # Get ego graph (node + neighbors within radius)
            if radius == 1:
                # Direct neighbors only
                neighbors = list(KG.neighbors(matched_node))
                subgraph_nodes = [matched_node] + neighbors
            else:
                # Use NetworkX ego_graph for larger radius
                ego = nx.ego_graph(KG, matched_node, radius=radius)
                subgraph_nodes = list(ego.nodes())

            # Build node and edge lists
            nodes = [{"id": n, "label": n} for n in subgraph_nodes]

            edges = []
            for u, v, data in KG.edges(subgraph_nodes, data=True):
                edges.append({
                    "from": u,
                    "to": v,
                    "label": data.get("line", ""),
                    "title": f"{data.get('distance', 0)} km"
                })

            return jsonify({"nodes": nodes, "edges": edges})

        except Exception as e:
            return jsonify({"nodes": [], "edges": [], "error": str(e)}), 500

    # Return full graph
    nodes = [{"id": n, "label": n} for n in KG.nodes()]

    edges = []
    for u, v, data in KG.edges(data=True):
        edges.append({
            "from": u,
            "to": v,
            "label": data.get("line", data.get("relationship", "")),
            "title": f"{data.get('distance', 0)} km"
        })

    return jsonify({"nodes": nodes, "edges": edges})


# API: Add Relationship (Frontend format)
@app.route("/api/add", methods=["POST"])
def add_relation_api():
    """
    Add relationship using frontend's expected format.
    Accepts: {entity1, relationship, entity2}
    """
    data = request.get_json()

    entity1 = data.get("entity1")
    entity2 = data.get("entity2")
    relationship = data.get("relationship", "connected_to")

    if not entity1 or not entity2:
        return jsonify({"error": "entity1 and entity2 required"}), 400

    # Add nodes if they don't exist
    if entity1 not in KG:
        KG.add_node(entity1)
    if entity2 not in KG:
        KG.add_node(entity2)

    # Add edge
    KG.add_edge(entity1, entity2, relationship=relationship, line=relationship)

    return jsonify({
        "message": "Relationship added",
        "total_stations": KG.number_of_nodes(),
        "total_connections": KG.number_of_edges()
    })


# API: Query Paths (Frontend format)
@app.route("/api/query", methods=["GET"])
def query_api():
    """
    Query graph using frontend's expected format.
    Supports: ?type=path&src=<source>&dst=<destination>
    """
    query_type = request.args.get("type")
    src = request.args.get("src")
    dst = request.args.get("dst")

    if query_type == "path":
        if not src or not dst:
            return jsonify({"error": "src and dst required"}), 400

        # Fuzzy match both stations
        matched_src, _ = find_station(src)
        matched_dst, _ = find_station(dst)

        if not matched_src or not matched_dst:
            return jsonify({
                "error": "Stations not found",
                "path": []
            }), 404

        try:
            # Find shortest path
            path = nx.shortest_path(KG, matched_src, matched_dst)

            return jsonify({
                "path": path,
                "length": len(path) - 1,
                "matched": {
                    "source": matched_src,
                    "destination": matched_dst
                }
            })

        except nx.NetworkXNoPath:
            return jsonify({"error": "No path found", "path": []}), 404

    return jsonify({"error": "Invalid query type"}), 400


# API: Upload CSV (Frontend compatible)
@app.route("/api/upload_csv", methods=["POST"])
def upload_csv_api():
    """
    CSV upload endpoint with /api prefix for frontend.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be a CSV"}), 400

    try:
        # Read CSV file
        csv_data = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))

        added_count = 0

        # Format 1: source, target, line, distance (metro-specific)
        if all(col in df.columns for col in ['source', 'target']):
            for _, row in df.iterrows():
                source = row.get('source')
                target = row.get('target')
                line = row.get('line', 'Unknown')
                distance = float(row.get('distance', 0.0))

                if pd.notna(source) and pd.notna(target):
                    if source not in KG:
                        KG.add_node(source)
                    if target not in KG:
                        KG.add_node(target)

                    KG.add_edge(source, target, line=line, distance=distance)
                    added_count += 1

        # Format 2: entity1, relationship, entity2 (generic format)
        elif all(col in df.columns for col in ['entity1', 'relationship', 'entity2']):
            for _, row in df.iterrows():
                entity1 = row.get('entity1')
                relationship = row.get('relationship')
                entity2 = row.get('entity2')

                if pd.notna(entity1) and pd.notna(entity2):
                    if entity1 not in KG:
                        KG.add_node(entity1)
                    if entity2 not in KG:
                        KG.add_node(entity2)

                    KG.add_edge(entity1, entity2, relationship=relationship, line=relationship)
                    added_count += 1

        else:
            return jsonify({
                "error": "Invalid CSV format",
                "expected_formats": [
                    "Format 1: source,target,line,distance",
                    "Format 2: entity1,relationship,entity2"
                ]
            }), 400

        return jsonify({
            "message": "CSV uploaded successfully",
            "added": added_count,
            "total_stations": KG.number_of_nodes(),
            "total_connections": KG.number_of_edges()
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process CSV: {str(e)}"}), 500


# Run Server
if __name__ == "__main__":
    app.run(debug=True)
