#!/bin/bash
# Test script for generic API endpoints

echo "==================================================="
echo "Testing Generic API Endpoints"
echo "==================================================="
echo ""

echo "1. Testing GET /graph (should show empty graph)..."
curl -s "http://localhost:5000/graph" | python3 -c "import json, sys; data=json.load(sys.stdin); print(f'✓ Nodes: {len(data[\"nodes\"])}, Edges: {len(data[\"edges\"])}, Stations: {data[\"total_stations\"]}, Connections: {data[\"total_connections\"]}')"
echo ""

echo "2. Testing POST /upload_csv (uploading delhi_metro.csv)..."
curl -s -X POST -F "file=@delhi_metro.csv" http://localhost:5000/upload_csv | python3 -c "import json, sys; data=json.load(sys.stdin); print(f'✓ Added: {data.get(\"added\", 0)} relationships, Total Stations: {data.get(\"total_stations\", 0)}, Total Connections: {data.get(\"total_connections\", 0)}')"
echo ""

echo "3. Testing GET /graph (should show full metro network)..."
curl -s "http://localhost:5000/graph" | python3 -c "import json, sys; data=json.load(sys.stdin); print(f'✓ Nodes: {len(data[\"nodes\"])}, Edges: {len(data[\"edges\"])}, Stations: {data[\"total_stations\"]}, Connections: {data[\"total_connections\"]}')"
echo ""

echo "4. Testing GET /graph?node=Kashmere Gate&radius=2 (subgraph query)..."
curl -s "http://localhost:5000/graph?node=Kashmere%20Gate&radius=2" | python3 -c "import json, sys; data=json.load(sys.stdin); print(f'✓ Subgraph: {len(data[\"nodes\"])} stations within 2 stops')"
echo ""

echo "5. Testing GET /paths (shortest path)..."
curl -s "http://localhost:5000/paths?source=Rajiv%20Chowk&destination=Kashmere%20Gate" | python3 -c "import json, sys; data=json.load(sys.stdin); print(f'✓ Path length: {data.get(\"length\", \"N/A\")} stops, Path: {\" → \".join(data.get(\"path\", [])[:3])}...')"
echo ""

echo "6. Testing POST /add_relationship (generic format)..."
curl -s -X POST http://localhost:5000/add_relationship \
  -H "Content-Type: application/json" \
  -d '{"entity1":"Test Station A","relationship":"connects_to","entity2":"Test Station B"}' \
  | python3 -c "import json, sys; data=json.load(sys.stdin); print(f'✓ {data.get(\"message\", \"Added\")}')"
echo ""

echo "7. Testing POST /add_relationship (metro format)..."
curl -s -X POST http://localhost:5000/add_relationship \
  -H "Content-Type: application/json" \
  -d '{"source":"Metro Test A","target":"Metro Test B","line":"Test Line","distance":2.5}' \
  | python3 -c "import json, sys; data=json.load(sys.stdin); print(f'✓ {data.get(\"message\", \"Added\")}')"
echo ""

echo "==================================================="
echo "All tests completed!"
echo "==================================================="
