"""
API ENDPOINTS:

1. INFORMATION & SEARCH
-----------------------
GET /
  → API info and available endpoints

GET /search_station?q=<query>
  → Search stations with fuzzy matching
  Example: curl "http://127.0.0.1:5000/search_station?q=Kashmere"
  Response: {"query": "Kashmere", "matches": [...], "total": 5}

GET /neighbors/<station>
  → Get neighboring stations (supports fuzzy matching)
  Example: curl "http://127.0.0.1:5000/neighbors/Rajiv%20Chowk"


2. GRAPH DATA
-------------
GET /full_graph
  → Get complete graph (all stations and connections)
  Example: http://127.0.0.1:5000/full_graph

GET /api/graph
GET /api/graph?node=<station>&radius=<N>
  → Get full graph OR subgraph within N stops of a station
  Examples:
    curl "http://127.0.0.1:5000/api/graph"
    curl "http://127.0.0.1:5000/api/graph?node=Rajiv%20Chowk&radius=2"
  Response: {"nodes": [...], "edges": [...]}


3. PATH FINDING
----------------------------------------------------
GET /paths?source=<s1>&destination=<s2>
GET /paths?source=<s1>&destination=<s2>&type=shortest_path
  → Find shortest path by number of stops (supports fuzzy matching)
  Example: curl "http://127.0.0.1:5000/paths?source=Rajiv%20Chowk&destination=Kashmere%20Gate"
  Response: {"path": [...], "length": 15, "route_details": [...]}

GET /paths?source=<s1>&destination=<s2>&type=shortest_distance
  → Find shortest path by distance with LINE-CHANGE PENALTY
  → Algorithm prioritizes:
      1. Staying on the same metro line when possible
      2. Changing lines only at valid interchange stations (stations with [Conn: ...])
      3. Minimizing total distance while avoiding unnecessary line changes
  → Penalty: 5.0 km equivalent per line change (configurable)
  → Heavy penalty (50 km) for impossible line changes (non-interchange stations)
  Example: curl "http://127.0.0.1:5000/paths?source=New%20Delhi&destination=Airport&type=shortest_distance"
  Response: {
    "path": [...],
    "length": 15,
    "total_distance_km": 25.3,
    "line_changes": 2,
    "route_details": [
      {"from": "Station A", "to": "Station B", "line": "Yellow line", "distance_km": 1.5},
      ...
    ]
  }

Legacy endpoints (still supported):
  GET /shortest_path?source=<s1>&target=<s2>
  GET /shortest_distance?source=<s1>&target=<s2>


4. ADD RELATIONSHIPS
--------------------
POST /add_relationship
  → Add single relationship (metro format)
  Body: {"source": "StationA", "target": "StationB", "line": "Red Line", "distance": 1.5}
  Example: curl -X POST -H "Content-Type: application/json" \
           -d '{"source":"TestStation1","target":"TestStation2","line":"Test Line","distance":2.0}' \
           http://127.0.0.1:5000/add_relationship

POST /api/add
  → Add single relationship (generic format for frontend)
  Body: {"entity1": "StationA", "entity2": "StationB", "relationship": "connects_to"}
  Example: curl -X POST -H "Content-Type: application/json" \
           -d '{"entity1":"Rajiv Chowk","entity2":"Connaught Place","relationship":"near"}' \
           http://127.0.0.1:5000/api/add
  Response: {"message": "Relationship added", "total_stations": 284, "total_connections": 285}

USAGE WITH FRONTEND:
====================
Start backend:  cd backend && python3 DelhiMetroKGApp.py
Start frontend: cd frontend && npm start
Access UI:      http://localhost:3000
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

    # Create edges between consecutive stations ON THE SAME LINE
    # Group stations by metro line first, then sort by distance within each line
    edges_created = 0

    # Group by line
    lines = df["Line"].unique()
    print(f"Found {len(lines)} unique metro lines")

    for line in lines:
        # Get all stations on this line, sorted by distance
        line_df = df[df["Line"] == line].sort_values("Distance")

        # Create edges between consecutive stations on this line
        for i in range(len(line_df) - 1):
            s1 = line_df.iloc[i]
            s2 = line_df.iloc[i + 1]

            station1 = s1["Station"]
            station2 = s2["Station"]

            # Distance between two stations = difference in cumulative distance
            distance_km = abs(s2["Distance"] - s1["Distance"])

            KG.add_edge(station1, station2, line=line, distance=distance_km)
            edges_created += 1

        print(f"  {line}: {len(line_df)} stations, {len(line_df)-1} edges")

    # Create edges between interchange stations (stations with [Conn: ...] suffix)
    # Find all interchange stations and connect them
    interchange_groups = {}
    for station in KG.nodes():
        # Extract base station name (before [Conn: ...])
        if '[Conn:' in station:
            base_name = station.split('[Conn:')[0].strip()
            if base_name not in interchange_groups:
                interchange_groups[base_name] = []
            interchange_groups[base_name].append(station)

    # Connect all variants of each interchange station
    interchange_edges = 0
    for base_name, stations in interchange_groups.items():
        if len(stations) > 1:
            # Connect each pair of stations at this interchange
            for i in range(len(stations)):
                for j in range(i + 1, len(stations)):
                    # Add zero-distance edge for interchange transfer
                    KG.add_edge(stations[i], stations[j], line="interchange", distance=0.0)
                    interchange_edges += 1

    print(f"Graph Loaded: {KG.number_of_nodes()} stations, {edges_created} line edges, {interchange_edges} interchange edges, {KG.number_of_edges()} total edges")


# Load graph when server starts
# Uncomment the line below to auto-load delhi_metro.csv on startup
# load_delhi_metro_graph()

# By default, server starts with an empty graph.
# You can upload delhi_metro.csv (or any CSV) via the UI after starting the server.
print("🚇 Delhi Metro Knowledge Graph API Started")
print(f"Current Graph: {KG.number_of_nodes()} stations, {KG.number_of_edges()} edges")
print("Upload delhi_metro.csv via the UI to load the full metro network")


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
    """
    Generic add relationship endpoint - replaces /add_relationship and /api/add.

    Supports two formats:
    1. Metro format: {"source": "A", "target": "B", "line": "Red Line", "distance": 1.5}
    2. Generic format: {"entity1": "A", "entity2": "B", "relationship": "connects_to"}

    Auto-detects format and handles both.
    """
    data = request.get_json()

    # Auto-detect format
    if "entity1" in data and "entity2" in data:
        # Generic format (entity1, entity2, relationship)
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

        # Add edge with relationship
        KG.add_edge(entity1, entity2, relationship=relationship, line=relationship)

        return jsonify({
            "message": "Relationship added",
            "total_stations": KG.number_of_nodes(),
            "total_connections": KG.number_of_edges()
        })

    else:
        # Metro format (source, target, line, distance)
        source = data.get("source")
        target = data.get("target")
        line = data.get("line", "UnknownLine")
        distance = float(data.get("distance", 0.0))

        if not source or not target:
            return jsonify({"error": "source and target required"}), 400

        # Add edge with metro attributes
        KG.add_edge(source, target, line=line, distance=distance)

        return jsonify({
            "message": "Relationship added",
            "total_stations": KG.number_of_nodes(),
            "total_connections": KG.number_of_edges()
        })


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
                    KG.add_edge(entity1, entity2, relationship=relationship, line=relationship)
                    added_count += 1

        # Format 3: Full Delhi Metro dataset (Station Names, Metro Line, etc.)
        elif 'Station Names' in df.columns or 'Station' in df.columns:
            print(f"[DEBUG] Format 3 detected. Columns: {list(df.columns)}")

            # Normalize column names
            station_col = 'Station Names' if 'Station Names' in df.columns else 'Station'
            line_col = 'Metro Line' if 'Metro Line' in df.columns else 'Line'
            distance_col = 'Dist. From First Station(km)' if 'Dist. From First Station(km)' in df.columns else 'Distance'

            print(f"[DEBUG] Mapping: {station_col} -> Station, {line_col} -> Line, {distance_col} -> Distance")

            # Rename for consistency
            df = df.rename(columns={
                station_col: 'Station',
                line_col: 'Line',
                distance_col: 'Distance'
            })

            # Sort by ID to ensure correct order
            if 'ID (Station ID)' in df.columns:
                df = df.rename(columns={'ID (Station ID)': 'ID'})
            if 'ID' in df.columns:
                df = df.sort_values('ID')

            print(f"[DEBUG] After rename, columns: {list(df.columns)}")
            print(f"[DEBUG] Total rows: {len(df)}")

            # Create nodes with attributes
            for _, row in df.iterrows():
                station = row.get('Station')
                if pd.notna(station):
                    KG.add_node(
                        station,
                        line=row.get('Line'),
                        opened=row.get('Opened(Year)') or row.get('Opened'),
                        layout=row.get('Layout'),
                        latitude=row.get('Latitude'),
                        longitude=row.get('Longitude')
                    )

            # Create edges between consecutive stations ON THE SAME LINE
            # Group stations by metro line first, then sort by distance within each line
            edges_created = 0

            # Group by line
            lines = df['Line'].unique()
            print(f"[DEBUG] Found {len(lines)} unique metro lines")

            for line in lines:
                if pd.isna(line):
                    continue

                # Get all stations on this line, sorted by distance
                line_df = df[df['Line'] == line].sort_values('Distance')

                # Create edges between consecutive stations on this line
                for i in range(len(line_df) - 1):
                    s1 = line_df.iloc[i]
                    s2 = line_df.iloc[i + 1]

                    station1 = s1['Station']
                    station2 = s2['Station']

                    if pd.notna(station1) and pd.notna(station2):
                        distance_km = abs(s2.get('Distance', 0) - s1.get('Distance', 0))

                        KG.add_edge(station1, station2, line=line, distance=distance_km)
                        added_count += 1
                        edges_created += 1

                print(f"[DEBUG] Line '{line}': {len(line_df)} stations, {len(line_df)-1} edges")

            print(f"[DEBUG] Total: Created {edges_created} edges within lines")

            # Create edges between interchange stations (stations with [Conn: ...] suffix)
            interchange_groups = {}
            for station in KG.nodes():
                if '[Conn:' in station:
                    base_name = station.split('[Conn:')[0].strip()
                    if base_name not in interchange_groups:
                        interchange_groups[base_name] = []
                    interchange_groups[base_name].append(station)

            print(f"[DEBUG] Found {len(interchange_groups)} interchange station groups")

            # Connect all variants of each interchange station
            interchange_edges = 0
            for base_name, stations in interchange_groups.items():
                if len(stations) > 1:
                    print(f"[DEBUG] Interchange '{base_name}': {len(stations)} variants")
                    for i in range(len(stations)):
                        for j in range(i + 1, len(stations)):
                            KG.add_edge(stations[i], stations[j], line="interchange", distance=0.0)
                            added_count += 1
                            interchange_edges += 1

            print(f"[DEBUG] Created {interchange_edges} interchange edges")

        else:
            return jsonify({
                "error": "Invalid CSV format",
                "expected_formats": [
                    "Format 1: source,target,line,distance",
                    "Format 2: entity1,relationship,entity2",
                    "Format 3: Station Names,Metro Line,... (Delhi Metro dataset)"
                ],
                "found_columns": list(df.columns)
            }), 400

        return jsonify({
            "message": "CSV uploaded successfully",
            "added": added_count,
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


# API: Generic Paths Endpoint
@app.route("/paths", methods=["GET"])
def find_paths():
    """
    Generic path finding endpoint - replaces /shortest_path, /shortest_distance, and /api/query.

    Supports:
    - GET /paths?source=A&destination=B → Shortest path by stops (default)
    - GET /paths?source=A&destination=B&type=shortest_path → Shortest path by stops
    - GET /paths?source=A&destination=B&type=shortest_distance → Shortest path by distance

    Returns path with fuzzy matching and helpful error messages.
    """
    source = request.args.get("source") or request.args.get("src")
    destination = request.args.get("destination") or request.args.get("dst")
    path_type = request.args.get("type", "shortest_path")

    if not source or not destination:
        return jsonify({"error": "source and destination parameters required"}), 400

    # Fuzzy match both stations
    matched_source, conf1 = find_station(source)
    matched_destination, conf2 = find_station(destination)

    if not matched_source or not matched_destination:
        errors = {}
        if not matched_source:
            errors["source"] = {
                "query": source,
                "suggestions": get_close_matches(source, KG.nodes(), n=3, cutoff=0.3)
            }
        if not matched_destination:
            errors["destination"] = {
                "query": destination,
                "suggestions": get_close_matches(destination, KG.nodes(), n=3, cutoff=0.3)
            }
        return jsonify({"error": "Stations not found", "details": errors}), 404

    try:
        if path_type == "shortest_distance":
            # Enhanced path finding with line-change penalty
            # Strategy: Use Dijkstra with modified edge weights that include line-change cost

            LINE_CHANGE_PENALTY = 5.0  # km equivalent penalty for changing lines

            # Create a modified graph with line-aware weights
            # We'll use A* or custom Dijkstra that tracks the current line

            # First, get the basic shortest path by distance
            base_path = nx.shortest_path(KG, matched_source, matched_destination, weight="distance")

            # Calculate the cost (distance + line changes) for the base path
            def calculate_path_cost(path_nodes):
                """Calculate total cost including line-change penalties"""
                total_distance = 0
                line_changes = 0
                previous_line = None

                for i in range(len(path_nodes) - 1):
                    u, v = path_nodes[i], path_nodes[i+1]
                    edge_data = KG[u][v]
                    distance = edge_data.get('distance', 0)
                    line = edge_data.get('line', '')

                    total_distance += distance

                    # Count line changes
                    if line and previous_line and line != previous_line:
                        line_changes += 1

                    previous_line = line if line else previous_line

                # Cost = distance + penalty for line changes
                return total_distance + (line_changes * LINE_CHANGE_PENALTY), total_distance, line_changes

            # Check if there's a better path with fewer line changes
            # Try to find alternative paths using BFS on same line first
            best_path = base_path
            best_cost, _, _ = calculate_path_cost(base_path)

            # Get the line of the source station
            source_edges = list(KG.edges(matched_source, data=True))
            if source_edges:
                source_line = source_edges[0][2].get('line', '')

                # Try to find a path that stays on the same line as much as possible
                # Use shortest path with custom weight that heavily penalizes line changes
                def line_aware_weight(u, v, data):
                    """Weight function that penalizes line changes"""
                    distance = data.get('distance', 1.0)
                    edge_line = data.get('line', '')

                    # If edge has no line data, just return distance
                    if not edge_line:
                        return distance

                    # Base weight is the distance
                    weight = distance

                    # Add penalty if this edge is on a different line than source
                    # This encourages staying on the same line
                    if edge_line != source_line:
                        weight += LINE_CHANGE_PENALTY

                    return weight

                try:
                    # Find path with line-aware weights
                    line_aware_path = nx.shortest_path(KG, matched_source, matched_destination,
                                                      weight=line_aware_weight)
                    line_aware_cost, _, _ = calculate_path_cost(line_aware_path)

                    # Use this path if it's better
                    if line_aware_cost < best_cost:
                        best_path = line_aware_path
                        best_cost = line_aware_cost
                except:
                    pass

            path = best_path

            # Calculate actual route details for the selected path
            total_distance = 0
            line_changes = 0
            previous_line = None
            route_details = []

            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = KG[u][v]
                distance = edge_data.get('distance', 0)
                line = edge_data.get('line', edge_data.get('relationship', ''))

                total_distance += distance

                # Track line changes
                if previous_line and line != previous_line and line != '':
                    line_changes += 1

                route_details.append({
                    "from": u,
                    "to": v,
                    "line": line,
                    "distance_km": round(distance, 2)
                })

                previous_line = line if line else previous_line

            return jsonify({
                "query": {"source": source, "destination": destination},
                "matched": {"source": matched_source, "destination": matched_destination},
                "fuzzy_match": conf1 < 1.0 or conf2 < 1.0,
                "path": path,
                "length": len(path) - 1,  # Number of stops
                "total_distance_km": round(total_distance, 2),
                "line_changes": line_changes,
                "route_details": route_details,
                "type": "shortest_distance"
            })
        else:
            # Find shortest path by number of stops (default)
            path = nx.shortest_path(KG, matched_source, matched_destination)

            # Calculate route details for display
            route_details = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = KG[u][v]
                route_details.append({
                    "from": u,
                    "to": v,
                    "line": edge_data.get('line', edge_data.get('relationship', '')),
                    "distance_km": round(edge_data.get('distance', 0), 2)
                })

            return jsonify({
                "query": {"source": source, "destination": destination},
                "matched": {"source": matched_source, "destination": matched_destination},
                "fuzzy_match": conf1 < 1.0 or conf2 < 1.0,
                "path": path,
                "length": len(path) - 1,  # Number of stops
                "route_details": route_details,
                "type": "shortest_path"
            })

    except nx.NetworkXNoPath:
        return jsonify({
            "error": "No path found",
            "query": {"source": source, "destination": destination},
            "matched": {"source": matched_source, "destination": matched_destination},
            "path": []
        }), 404


# API: Full Graph
@app.route("/graph", methods=["GET"])
def get_graph_generic():
    """
    Generic graph endpoint - replaces /full_graph and /api/graph.

    Supports:
    - GET /graph → Full graph in vis-network format
    - GET /graph?node=<station>&radius=<N> → Subgraph within N hops
    - GET /graph?format=detailed → Full graph with all attributes

    Returns vis-network compatible format by default: {nodes: [{id, label}], edges: [{from, to, label, title}]}
    """
    node = request.args.get("node")
    radius = request.args.get("radius", type=int)
    format_type = request.args.get("format", "vis-network")

    # Subgraph query
    if node and radius:
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

            # Build node and edge lists in vis-network format
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

    # Full graph
    if format_type == "detailed":
        # Detailed format with all attributes (for analysis/debugging)
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

    else:
        # Default: vis-network format
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


# Run Server
if __name__ == "__main__":
    app.run(debug=True)
