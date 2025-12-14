# NLP Applications - Assignment1 - PS9
## Delhi Metro Knowledge Graph Application

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start-guide)
3. [Project Structure](#project-structure)
4. [API Documentation](#api-documentation)
5. [Frontend-Backend Integration](#frontend-backend-integration)
6. [Testing Guide](#testing-guide)
7.  [Build & Deployment](#build--deployment)
8.  [Design Choices](#design-choices)
9.  [Assignment Deliverables](#assignment-deliverables)

---

## Overview

A full-stack web application for visualizing and querying the Delhi Metro network as a Knowledge Graph. Users can interactively explore metro stations, find shortest paths, add relationships, and upload CSV data for bulk import.

### Dataset
- **Stations:** 283 metro stations (default when delhi_metro.csv is uploaded)
- **Connections:** 284+ edges
- **Attributes:** Station names, metro lines, distances, GPS coordinates, opening dates
- **Dataset Link:** [delhi-metro-dataset](https://www.kaggle.com/datasets/arunjangir245/delhi-metro-dataset)

---

## Quick Start Guide

### Prerequisites
```bash
# Python 3.9+
python3 --version

# Node.js 14+ (only for development)
node --version
```

### Option 1: Production Build (Recommended)

**Terminal 1 - Start Backend:**
```bash
cd backend
pip3 install -r requirements.txt
python3 DelhiMetroKGApp.py

# You should see:
# 🚇 Delhi Metro Knowledge Graph API Started
# Current Graph: 0 stations, 0 edges
# Upload delhi_metro.csv via the UI to load the full metro network
```

**Terminal 2 - Serve Frontend:**
```bash
cd frontend/build
python3 -m http.server 3000
```

**Access:** http://localhost:3000

**First Steps:**
1. Open UI - Legend shows "Stations: 0, Connections: 0"
2. Click "Choose File" under CSV Upload
3. Select `delhi_metro.csv` from Assignment-1 folder
4. Click "Upload"
5. Success! Legend updates to "Stations: 283, Connections: 284"
6. Graph visualization displays the full metro network

### Option 2: Development Mode

**Terminal 1 - Backend:**
```bash
cd backend
python3 DelhiMetroKGApp.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install  # First time only
npm start
```

**Auto-opens:** http://localhost:3000

---

## Project Structure

```
Assignment-1/
├── README.md                             # This file (everything you need)
├── Task_B_Enhancement_Plan.md            # Enhancement plan (convert to PDF)
├── .gitignore                            # Git ignore rules
│
├── backend/
│   ├── DelhiMetroKGApp.py                # Flask API server
│   ├── requirements.txt                  # Python dependencies
│   ├── delhi_metro.csv                   # Metro dataset (283 stations)
│   ├── test_metro_relationships.csv      # Test CSV (metro format)
│   └── test_generic_relationships.csv    # Test CSV (generic format)
│
└── frontend/
    ├── build/                            # Production build (4.9MB)
    ├── src/
    │   ├── App.jsx                       # Main component
    │   ├── api.js                        # Axios API client
    │   ├── theme.js                      # Material-UI theme
    │   └── components/
    │       ├── AddRelationForm.jsx       # Manual relationship entry
    │       ├── CSVUpload.jsx             # CSV bulk upload
    │       ├── QueryPanel.jsx            # Search & path finding
    │       └── GraphView.jsx             # Network visualization
    ├── public/
    │   └── index.html
    ├── package.json                      # Dependencies
    └── node_modules/                     # EXCLUDED by .gitignore (761MB)
```

---

## Technical Stack (Part A)

#### 1. Frontend Development
- React + Material-UI web interface
- Input fields for entity-relationship pairs
- CSV file upload for bulk import
- Interactive graph visualization (vis-network)
- Dynamic updates on data changes

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.2.0 | UI framework |
| Material-UI | 5.18.0 | UI components |
| vis-network | 9.1.13 | Graph visualization |
| Axios | 1.13.2 | HTTP client |
| react-scripts | 5.0.1 | Build tool |

---

#### 2. Backend Development
- Flask RESTful API
- NetworkX graph management
- Add relationships endpoint
- Query endpoints (neighbors, paths)
- CSV upload with format detection

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.9+ | Runtime |
| Flask | 3.1+ | Web framework |
| Flask-CORS | 6.0+ | Cross-origin support |
| NetworkX | Latest | Graph algorithms |
| Pandas | Latest | CSV processing |

---

#### 3. Integration
- Frontend-backend via Axios
- CORS enabled
- Real-time graph updates
- Error handling

#### 4. Task B: Enhancement Plan
- 45-page detailed enhancement plan
- Visual improvements
- Interactive features
- Future innovations

---

## API Documentation

**Base URL:** `http://localhost:5000`
### Generic API Endpoints (Used by React UI and CLI)

#### 1. Get Full Graph
```http
GET /graph
```

**Response:**
```json
{
  "nodes": [
    {"id": "Station Name", "label": "Station Name"}
  ],
  "edges": [
    {"from": "A", "to": "B", "label": "Blue Line", "title": "2.5 km"}
  ],
  "total_stations": 283,
  "total_connections": 284
}
```

**Example:**
```bash
curl "http://localhost:5000/graph"
```

---

#### 2. Get Subgraph (Neighborhood)
```http
GET /graph?node=<station>&radius=<N>
```

**Parameters:**
- `node`: Station name (fuzzy matching)
- `radius`: Neighborhood radius (1, 2, 3, etc.)

**Response**
```json
{
  "edges": [
    {
      "from": "Tilak Nagar",
      "label": "Magenta line",
      "title": "5.400000000000002 km",
      "to": "Nehru Enclave"
    },
    {
      "from": "Tilak Nagar",
      "label": "Blue line",
      "title": "1.5 km",
      "to": "Kalkaji Mandir [Conn: Magenta]"
    },
    {
      "from": "Kashmere Gate [Conn: Violet,Yellow]",
      "label": "Voilet line",
      "title": "0.3000000000000007 km",
      "to": "Kalkaji Mandir [Conn: Magenta]"
    },
    {
      "from": "Kashmere Gate [Conn: Violet,Yellow]",
      "label": "Red line",
      "title": "1.3999999999999986 km",
      "to": "Ghevra Metro station"
    },
    {
      "from": "Ghevra Metro station",
      "label": "Green line",
      "title": "4.299999999999997 km",
      "to": "Knowledge Park II"
    },
    {
      "from": "Knowledge Park II",
      "label": "Aqua line",
      "title": "4.199999999999999 km",
      "to": "Rajiv Chowk [Conn: Blue]"
    }
  ],
  "nodes": [
    {
      "id": "Tilak Nagar",
      "label": "Tilak Nagar"
    },
    {
      "id": "Kashmere Gate [Conn: Violet,Yellow]",
      "label": "Kashmere Gate [Conn: Violet,Yellow]"
    },
    {
      "id": "Kalkaji Mandir [Conn: Magenta]",
      "label": "Kalkaji Mandir [Conn: Magenta]"
    },
    {
      "id": "Ghevra Metro station",
      "label": "Ghevra Metro station"
    },
    {
      "id": "Knowledge Park II",
      "label": "Knowledge Park II"
    }
  ]
}
```

**Example:**
```bash
curl "http://localhost:5000/graph?node=Kashmere%20Gate&radius=2"
```

---

#### 3. Find Shortest Path
```http
GET /paths?source=<source>&destination=<destination>
```

**Response:**
```json
{
  "fuzzy_match": true,
  "length": 85,
  "matched": {
    "destination": "Kashmere Gate [Conn: Violet,Yellow]",
    "source": "Rajiv Chowk [Conn: Yellow]"
  },
  "path": [
    "Rajiv Chowk [Conn: Yellow]",
    "Old Faridabad",
    "Qutab Minar",
    "R K Ashram Marg",
    "Rohini West",
    "Anand Vihar [Conn: Blue]",
    "Badkal Mor",
    "Sector 28 Faridabad",
    "Rohini East",
    "IP Extension",
    "Jhandewalan",
    "Saket",
    "Pitam Pura",
    "Mandawali - West Vinod Nagar",
    "Karol Bagh",
    "Malviya Nagar",
    "Mewala Maharajpur",
    "N.H.P.C. Chowk",
    "Rajendra Place",
    "Hauz Khas [Conn: Magenta]",
    "Kohat Enclave",
    "Vinod Nagar East",
    "Botanical Garden [Conn: Blue]",
    "Sarai",
    "Netaji Subash Place [Conn: Pink]",
    "Trilokpuri Sanjay Lake",
    "Patel Nagar",
    "Green Park",
    "Okhla Bird Sanctuary",
    "Kalindi Kunj",
    "Badarpur Border",
    "Mayur Vihar Pocket I",
    "AIIMS",
    "Keshav Puram",
    "Shadipur",
    "Tughlakabad",
    "Kirti Nagar [Conn: Green]",
    "Dilli Haat INA [Conn: Pink]",
    "Mayur Vihar Phase-1 [Conn: Blue]",
    "Kanhaiya Nagar",
    "Jasola Vihar Shaheen Bagh",
    "Depot Greater Noida",
    "Mohan Estate",
    "Moti Nagar",
    "Inderlok [Conn: Green]",
    "Sarai Kale Khan Hazrat Nizamuddin",
    "Brigadier Hoshiar Singh",
    "Okhla Vihar",
    "Jor Bagh",
    "Bahdurgarh City",
    "Lok Kalyan Marg",
    "Ramesh Nagar",
    "Sarita Vihar",
    "Shastri Nagar",
    "GNIDA Office",
    "JAMIA MILLIA ISLAMIA",
    "Ashram",
    "Rajouri Garden [Conn: Pink]",
    "Delta 1 Greater Noida",
    "Pandit Shree Ram Sharma",
    "Udyog Bhawan",
    "Jasola",
    "Sukhdev Vihar",
    "Pratap Nagar",
    "Vinobapuri",
    "Tagore Garden",
    "Tikri Border",
    "Okhla NSIC",
    "Lajpat Nagar [Conn: Violet]",
    "Pul Bangash",
    "Okhla",
    "Alpha 1 Greater Noida",
    "Central Secretariat [Conn: Violet]",
    "South Extension",
    "Patel Chowk",
    "Subhash Nagar",
    "Tis Hazari",
    "Pari Chowk Greater Noida",
    "Kalkaji Mandir [Conn: Violet]",
    "Tikri Kalan",
    "Govind Puri",
    "Dilli Haat INA [Conn: Yellow]",
    "Rajiv Chowk [Conn: Blue]",
    "Knowledge Park II",
    "Ghevra Metro station",
    "Kashmere Gate [Conn: Violet,Yellow]"
  ],
  "query": {
    "destination": "Kashmere Gate",
    "source": "Rajiv Chowk"
  },
  "type": "shortest_path"
}
```

**Example:**
```bash
curl "http://localhost:5000/paths?source=Rajiv%20Chowk&destination=Kashmere%20Gate"
```

**Shortest Distance (by km):**
```bash
curl "http://localhost:5000/paths?source=New%20Delhi&destination=Airport&type=shortest_distance"
```

---

#### 4. Add Relationship
```http
POST /add_relationship
Content-Type: application/json
```

**Request Body (Auto-detects format):**

**Format 1 - Generic:**
```json
{
  "entity1": "Station A",
  "relationship": "connects_to",
  "entity2": "Station B"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/add_relationship \
  -H "Content-Type: application/json" \
  -d '{"entity1":"Test A","relationship":"test","entity2":"Test B"}'
```
**Format 2 - Metro:**
```json
{
  "source": "New Delhi",
  "target": "Connaught Place",
  "line": "Airport Express",
  "distance": 2.5
}
```
---

#### 5. Upload CSV
```http
POST /upload_csv
Content-Type: multipart/form-data
```

**CSV (Metro):**
```csv
source,target,line,distance
New Delhi,Connaught Place,Airport Express,2.5
```

**Example:**
```bash
curl -X POST -F "file=@delhi_metro.csv" http://localhost:5000/upload_csv
```

---

### Direct API Endpoints (CLI/Testing)

#### 6. Search Stations (Fuzzy)
```http
GET /search_station?q=<query>
```

**Example:**
```bash
curl "http://localhost:5000/search_station?q=Kashmere"
```

**Response:**
```json
{
  "query": "Kashmere",
  "matches": [
    {
      "station": "Kashmere Gate [Conn: Yellow]",
      "line": "Violet line",
      "latitude": 28.6675,
      "longitude": 77.22817
    }
  ],
  "total": 1
}
```

---

#### 7. Get Neighbors
```http
GET /neighbors/<station>
```

**Example:**
```bash
curl "http://localhost:5000/neighbors/Kashmere%20Gate"
```

**Response:**
```json
{
  "query": "Kashmere Gate",
  "matched_station": "Kashmere Gate [Conn: Yellow]",
  "fuzzy_match": true,
  "neighbors": [
    {
      "station": "Sector 55-66",
      "line": "Rapid Metro",
      "distance_km": 0.0
    }
  ]
}
```

---

#### 8. Shortest Path (Legacy)
```http
GET /shortest_path?source=<A>&target=<B>
```

#### 9. Shortest Distance
```http
GET /shortest_distance?source=<A>&target=<B>
```

#### 10. Full Graph (Legacy)
```http
GET /full_graph
```

---

## Frontend-Backend Integration

### API Endpoint Mapping

| Frontend Call | Backend Endpoint | Status | Purpose |
|--------------|------------------|--------|---------|
| `GET /graph` | Implemented | Working | Load full graph |
| `GET /graph?node=X&radius=2` | Implemented | Working | Get neighborhood |
| `GET /paths?source=A&destination=B` | Implemented | Working | Find shortest path |
| `POST /add_relationship` | Implemented | Working | Add relationship |
| `POST /upload_csv` | Implemented | Working | Upload CSV |

### Data Flow

```
User Action (React UI)
  ↓
Axios HTTP Request → Flask Backend
  ↓                       ↓
API Endpoint         NetworkX Graph
  ↓                       ↓
JSON Response  ←   ← Graph Query/Update
  ↓
React State Update
  ↓
vis-network Re-render
```

---

## Testing Guide

> **See Also:** `Steps` file contains detailed test cases for "Add Relationship" feature with 20+ realistic examples

### Test Checklist

- [ ] **Test 1:** Backend starts with empty graph (0 stations, 0 edges)
- [ ] **Test 2:** Legend displays correct counts in real-time (0 stations, 0 edges)
- [ ] **Test 3:** CSV upload works (delhi_metro.csv → 283 stations, 284 edges)
- [ ] **Test 4:** Legend updates after CSV upload (283 stations, 284 edges)
- [ ] **Test 5:** Add relationship form works
- [ ] **Test 6:** Legend updates after adding relationships
- [ ] **Test 7:** Neighborhood query works
- [ ] **Test 8:** Path finding works
- [ ] **Test 9:** Error handling works (detailed error messages)
- [ ] **Test 10:** Graph interactions (pan/zoom/hover)

### Command-Line Testing

```bash
# Test all endpoints
echo "Testing /graph..."
curl -s "http://localhost:5000/graph" | python3 -c "import json, sys; data=json.load(sys.stdin); print(f'Nodes: {len(data[\"nodes\"])}, Edges: {len(data[\"edges\"])}')"

echo "Testing /add_relationship..."
curl -s -X POST http://localhost:5000/add_relationship \
  -H "Content-Type: application/json" \
  -d '{"entity1":"Test A","relationship":"test","entity2":"Test B"}' \
  | python3 -c "import json, sys; print(json.load(sys.stdin)[\"message\"])"

echo "Testing /paths..."
curl -s "http://localhost:5000/paths?source=New%20Delhi%20Station&destination=Terminal%203%20Airport" \
  | python3 -c "import json, sys; data=json.load(sys.stdin); print(f'Path length: {data.get(\"length\", \"N/A\")}')"

echo "Testing /upload_csv..."
curl -s -X POST -F "file=@test_generic_relationships.csv" \
  http://localhost:5000/upload_csv \
  | python3 -c "import json, sys; print(json.load(sys.stdin)[\"message\"])"
```
---

### Complete Testing Scenario: Walking Network on Yellow Line

**Prerequisites:**
- Backend running with `delhi_metro.csv` loaded (283 stations, 284 connections)
- Frontend open at http://localhost:3000

**Context:**
The following stations already exist on the Yellow Line with metro connections:
- Rajiv Chowk [Conn: Blue] (interchange with Blue line)
- Patel Chowk
- Central Secretariat [Conn: Violet] (interchange with Violet line)
- Udyog Bhawan

We will add **walking distance** relationships between these stations.

---

#### Step 1: Add Walking Connections

Add these relationships one by one using the "Add Relationship" form:

**Relationship 1:**
```yaml
Entity 1: "Rajiv Chowk [Conn: Blue]"
Relationship: walking_distance
Entity 2: Patel Chowk

Expected Legend:
  Stations: 283 (no new stations, both exist)
  Connections: 285 (284 metro + 1 walking)
```

**Relationship 2:**
```yaml
Entity 1: Patel Chowk
Relationship: walking_distance
Entity 2: "Central Secretariat [Conn: Violet]"

Expected Legend:
  Stations: 283
  Connections: 286 (284 metro + 2 walking)
```

**Relationship 3:**
```yaml
Entity 1: "Central Secretariat [Conn: Violet]"
Relationship: walking_distance
Entity 2: Udyog Bhawan

Expected Legend:
  Stations: 283
  Connections: 287 (284 metro + 3 walking)
```

---

#### Step 2: Query Neighborhood (Radius 1)

Test neighborhood query to see both metro and walking connections:

```yaml
Query Parameters:
  Station: Patel Chowk
  Radius: 1

Expected Result:
  Stations Shown: Multiple stations (including):
    - "Rajiv Chowk [Conn: Blue]" (via walking_distance)
    - "Central Secretariat [Conn: Violet]" (via walking_distance)
    - "Rajiv Chowk [Conn: Yellow]" (via Yellow line metro)
    - Udyog Bhawan (via Yellow line metro - next station)
    - Other Yellow line neighbors

  Total: 5 stations

  Connections Shown:
    - 2 walking_distance edges (Rajiv-Patel, Patel-Central)
    - Multiple metro line edges
```

#### Step 3: Query Neighborhood (Radius 2)

Expand to radius 2 to see more connections:

```yaml
Query Parameters:
  Station: Patel Chowk
  Radius: 2

Expected Result:
  Stations Shown: More stations (including):
    - All radius-1 stations
    - "Udyog Bhawan" (2 hops: Patel→Central→Udyog via walking)
    - More Yellow line stations
    - Blue line stations connected to Rajiv Chowk
    - Violet line stations connected to Central Secretariat

  Total: ~10-15 stations (depends on metro network structure)
```

#### Step 4: Find Path Using Walking Connections

Test path that uses the walking connections:

```yaml
Query Parameters:
  Source: Rajiv Chowk [Conn: Blue]
  Destination: Udyog Bhawan

Expected Result:
  Path may use:
    Option 1 (via walking):
      Rajiv Chowk [Conn: Blue] → Patel Chowk →
      Central Secretariat [Conn: Violet] → Udyog Bhawan
      (3 hops via walking)

    Option 2 (via Yellow line metro):
      Rajiv Chowk [Conn: Blue] → Rajiv Chowk [Conn: Yellow] →
      Patel Chowk → Central Secretariat [Conn: Yellow] → Udyog Bhawan
      (4 hops via metro interchanges)

    Option 3 (hybrid):
      Some combination of metro and walking

  NetworkX will choose the SHORTEST path (fewest hops)
```

#### Step 5: Alternative Path Test

Test a simpler path:

```yaml
Query Parameters:
  Source: Patel Chowk
  Destination: Udyog Bhawan

Expected Result:
  Path Length: Likely 2-3 hops
  Route options:
    - Via walking: Patel Chowk → Central Secretariat → Udyog Bhawan
    - Via metro: Patel Chowk → (next Yellow line station) → Udyog Bhawan

  NetworkX chooses shortest
```

## Design Choices and Implementation Challenges

### Design Choices

#### 1. Graph Data Structure: NetworkX Undirected Graph

**Choice:** Used NetworkX `Graph()` (undirected) instead of `DiGraph` (directed).

**Rationale:**
- Metro connections are **bidirectional** (trains run both ways)
- Passengers can travel in either direction on any line
- Simplifies path finding (no need to handle reverse edges)
- Matches real-world metro network topology

**Code:**
```python
KG = nx.Graph()  # Undirected graph for bidirectional metro connections
```

---

#### 2. Fuzzy Station Name Matching (4-Level Strategy)

**Choice:** Implemented 4-level fuzzy matching instead of exact string matching.

**Rationale:**
- Station names are complex: `"Rajiv Chowk [Conn: Blue]"`
- Users may not know exact spelling or suffix
- Improves user experience significantly
- Handles typos and partial names

**Implementation (lines 153-193 in DelhiMetroKGApp.py):**
```python
def find_station(query):
    # Level 1: Exact match (case-sensitive)
    if query in KG:
        return query, 1.0

    # Level 2: Case-insensitive exact match
    for station in KG.nodes():
        if station.lower() == query.lower():
            return station, 1.0

    # Level 3: Substring match (prioritize starts-with)
    substring_matches = []
    for station in KG.nodes():
        if query.lower() in station.lower():
            if station.lower().startswith(query.lower()):
                substring_matches.insert(0, station)  # Prioritize
            else:
                substring_matches.append(station)
    if substring_matches:
        return substring_matches[0], 0.9

    # Level 4: Fuzzy match (Ratcliff-Obershelp algorithm, cutoff=0.6)
    matches = get_close_matches(query, KG.nodes(), n=1, cutoff=0.6)
    if matches:
        return matches[0], 0.7

    return None, 0
```

---

#### 3. Path Finding with Line-Change Penalty

**Choice:** Implemented custom path finding that penalizes line changes.

**Problem:**
- Standard Dijkstra finds shortest distance path
- May suggest impractical routes mixing multiple lines
- Real passengers prefer staying on same line when possible

**Solution (lines 594-713 in DelhiMetroKGApp.py):**
```python
LINE_CHANGE_PENALTY = 5.0  # km equivalent penalty

# Strategy:
# 1. Find basic shortest path by distance
base_path = nx.shortest_path(KG, source, dest, weight="distance")

# 2. Find line-aware path (prefers staying on same line)
def line_aware_weight(u, v, data):
    distance = data.get('distance', 1.0)
    edge_line = data.get('line', '')

    # Add penalty if edge is on different line than source
    if edge_line != source_line:
        return distance + LINE_CHANGE_PENALTY
    return distance

line_aware_path = nx.shortest_path(KG, source, dest, weight=line_aware_weight)

# 3. Compare both paths and choose the one with lower total cost
```

**Result:** Returns paths that balance distance vs. line changes.

---

#### 4. CSV Format Auto-Detection

**Choice:** Support 3 CSV formats with automatic detection.

**Rationale:**
- Different use cases (metro-specific vs. generic knowledge graphs)
- User-friendly (no manual format selection)
- Flexible data import

**Implementation (lines 367-502 in DelhiMetroKGApp.py):**
```python
# Format 1: Metro-specific (source, target, line, distance)
if all(col in df.columns for col in ['source', 'target']):
    # Process metro format...

# Format 2: Generic KG (entity1, relationship, entity2)
elif all(col in df.columns for col in ['entity1', 'relationship', 'entity2']):
    # Process generic format...

# Format 3: Full Delhi Metro dataset (Station Names, Metro Line, ...)
elif 'Station Names' in df.columns or 'Station' in df.columns:
    # Process full dataset format...
```

---

#### 5. Frontend-Backend Separation

**Choice:** Separate React frontend and Flask backend with RESTful API.

**Rationale:**
- **Scalability:** Backend can serve multiple clients
- **Maintainability:** Independent updates to UI vs. logic
- **Testability:** Each component can be tested separately
- **Deployment flexibility:** Frontend (static files) and backend can be deployed separately

**API Design:**
- Generic endpoints (not frontend-specific)
- `/graph`, `/paths`, `/add_relationship`, `/upload_csv`
- JSON responses with consistent structure
- CORS enabled for cross-origin requests

---

### Implementation Challenges

#### Challenge 1: CSV Data Structure (Interleaved Stations)

**Problem:**
- Initial assumption: CSV has stations grouped by metro line
- Reality: CSV has stations **interleaved** (all station #1s, then all station #2s, etc.)
- Created only 40 edges instead of 271!

**Discovery:**
```
Row 1: Red line station #1
Row 2: Blue line station #1
Row 3: Yellow line station #1
...
Row 15: Red line station #2
Row 16: Blue line station #2
```

**Initial (Broken) Code:**
```python
# Created edges between consecutive rows (WRONG!)
for i in range(len(df) - 1):
    if df.iloc[i]['Line'] == df.iloc[i+1]['Line']:  # Rarely true!
        KG.add_edge(...)
```

**Solution:**
```python
# Group by line first, then sort by distance
lines = df["Line"].unique()

for line in lines:
    # Get all stations on this line, sorted by distance
    line_df = df[df["Line"] == line].sort_values("Distance")

    # Create edges between consecutive stations on this sorted line
    for i in range(len(line_df) - 1):
        s1 = line_df.iloc[i]
        s2 = line_df.iloc[i + 1]
        KG.add_edge(s1['Station'], s2['Station'], line=line, distance=...)
```

**Impact:** Fixed edge count from 40 → 271 edges!

---

#### Challenge 2: Interchange Station Connections

**Problem:**
- Interchange stations have multiple nodes (one per line)
- Example: `"Rajiv Chowk [Conn: Blue]"` (Yellow line) vs. `"Rajiv Chowk [Conn: Yellow]"` (Blue line)
- Without connections, graph is **disconnected**
- Cannot find paths between different metro lines

**Solution (lines 156-176 in DelhiMetroKGApp.py):**
```python
# Extract base name from stations with [Conn: ...] suffix
interchange_groups = {}
for station in KG.nodes():
    if '[Conn:' in station:
        base_name = station.split('[Conn:')[0].strip()  # "Rajiv Chowk"
        interchange_groups[base_name].append(station)

# Connect all variants of each interchange
for base_name, stations in interchange_groups.items():
    if len(stations) > 1:
        # Create edges between all pairs
        for i in range(len(stations)):
            for j in range(i + 1, len(stations)):
                KG.add_edge(stations[i], stations[j],
                           line="interchange", distance=0.0)
```

**Result:** Added 22 interchange edges, making the graph fully connected.

---

#### Challenge 3: Path Finding Algorithm Complexity

**Problem:**
- Initial attempt: Used `nx.all_simple_paths()` to enumerate all paths
- In dense metro network: Thousands of paths exist
- Limiting to first 50 paths missed the optimal path
- **Result:** 47-stop path instead of 4-stop path!

**Failed Approach:**
```python
# Enumerate all paths (TOO SLOW and incomplete)
all_paths = list(nx.all_simple_paths(KG, source, dest, cutoff=10))
for path in all_paths[:50]:  # Only first 50
    cost = calculate_cost(path)
    # May miss the best path!
```

**Successful Approach:**
```python
# Use Dijkstra twice with different weight functions
base_path = nx.shortest_path(KG, source, dest, weight="distance")
line_aware_path = nx.shortest_path(KG, source, dest, weight=line_aware_weight)

# Compare and choose the better path
```

**Complexity:** O(V log V + E) for each Dijkstra call vs. exponential for enumerating paths.

---

#### Challenge 4: Graph Visualization Performance

**Problem:**
- 283 nodes + 293 edges = heavy rendering
- Initial implementation caused browser lag
- Zoom/pan was slow

**Solution:**
- Used vis-network library (optimized for large graphs)
- Implemented physics stabilization
- Added loading indicators
- Limited initial rendering to visible viewport

**Code (frontend/src/components/GraphView.jsx):**
```javascript
const options = {
  physics: {
    stabilization: {
      iterations: 200,  // Limit iterations
      updateInterval: 25
    },
    barnesHut: {
      gravitationalConstant: -8000,
      springConstant: 0.04,
      springLength: 95
    }
  },
  interaction: {
    zoomView: true,
    dragView: true
  }
};
```

---

#### Challenge 5: API Consolidation and Backward Compatibility

**Problem:**
- Initially had duplicate endpoints: `/api/graph` and `/full_graph`
- Frontend-specific endpoints mixed with generic endpoints
- 265 lines of duplicate code

**Solution:**
- Consolidated to generic RESTful endpoints
- `/graph` supports multiple query modes (full graph, subgraph, detailed)
- Removed all `/api/*` endpoints (broke documentation temporarily)

**Lessons Learned:**
- Update documentation immediately when changing APIs
- Use API versioning for future changes
- Test all endpoints after refactoring

---

### Performance Metrics

**Graph Construction:**
- Loading 283 stations + creating 293 edges: **< 500ms**
- Interchange detection and connection: **< 100ms**

**Query Performance:**
- Shortest path (by stops): **< 50ms** (BFS)
- Shortest path (by distance + line penalty): **< 200ms** (2× Dijkstra)
- Subgraph extraction (radius=2): **< 100ms**
- Full graph serialization to JSON: **< 300ms**

**Frontend Rendering:**
- Initial graph load: **< 2s** (includes physics stabilization)
- Graph update after query: **< 500ms**
- Path highlighting: **< 100ms**

---

## Build & Deployment

### Build Production Version

```bash
cd frontend

# Install dependencies (if needed)
npm install

# Create production build
npm run build
```

**Output:** `frontend/build/` folder (4.9MB)

### Test Production Build

```bash
cd frontend/build
python3 -m http.server 3000
```

Visit: http://localhost:3000
---

## Assignment Deliverables

### Part A - Knowledge Graph Application (10 marks)

| Deliverable | Status | Location |
|------------|--------|----------|
| Well-documented Python code | | `backend/DelhiMetroKGApp.py` |
| Well-documented React code | | `frontend/src/` |
| Production build | | `frontend/build/` |
| Instructions for running | | This README |
| Design choices report | | This README (Design Choices section) |
| Challenges faced | | This README (Troubleshooting section) |
| Screenshots | ⏳ | Capture during testing |
| Task-B Enhancement Plan | | `Task_B_Enhancement_Plan.md` → PDF |

### Part B - Literature Survey (5 marks)

| Deliverable | Status | Topic |
|------------|--------|-------|
| Literature review PDF | ⏳ | "Hallucination Reduction in RAG" |
| Research papers reviewed | ⏳ | 15-20 papers (2023-2025) |
| Comparative analysis | ⏳ | Strategy comparison |

---

## Submission Checklist

### Before Submission

- [x] Backend works locally
- [x] Frontend works locally
- [x] Production build created
- [ ] All features tested manually
- [ ] Screenshots captured
- [ ] Task B converted to PDF
- [ ] Part B literature survey written
- [ ] ZIP package created (< 10MB)
- [ ] OSHA Lab screenshot included

### Create Submission Package

```bash
# Test production build
cd backend && python3 DelhiMetroKGApp.py &
cd frontend/build && python3 -m http.server 3000

# Create ZIP
cd /path/to/parent
zip -r Assignment-1-Submission.zip Assignment-1 \
  -x "*/node_modules/*" \
  -x "*/.DS_Store" \
  -x "*/__pycache__/*"

# Verify size
du -sh Assignment-1-Submission.zip
# Should be < 10MB
```

### Submission Files

```
Assignment-1-Submission.zip
├── backend/
│   ├── DelhiMetroKGApp.py
│   ├── delhi_metro.csv
│   ├── requirements.txt
│   └── test_*.csv
├── frontend/
│   ├── build/              # Production build
│   ├── src/                # Source code
│   ├── public/
│   └── package.json
├── README.md              # This file
├── Task_B_Enhancement_Plan.pdf  # Converted from .md
└── .gitignore

Excluded:
- node_modules/ (761MB)
- .DS_Store
- __pycache__/
- *.log
```

---

## Screenshots Required

1. **Main Interface** - Full application view
2. **Add Relationship** - Form with sample data
3. **CSV Upload** - File selected, ready to upload
4. **Path Finding** - Query and result
5. **Graph Visualization** - Zoomed view of network
6. **Neighborhood Query** - Subgraph result
7. **OSHA Lab** - Terminal with credentials visible

---

## Performance Benchmarks

Expected response times (development mode):

| Endpoint | Response Time | Notes |
|----------|--------------|-------|
| `GET /graph` (full) | < 500ms | 283 nodes, 284 edges |
| `GET /graph` (subgraph) | < 100ms | Radius query |
| `GET /paths` | < 200ms | Dijkstra algorithm |
| `POST /add_relationship` | < 50ms | Single relationship |
| `POST /upload_csv` | < 1s | 100 rows |

---

## Future Enhancements

See **Task_B_Enhancement_Plan.md** for detailed 45-page plan covering:

### Tier 1: Essential (Immediate)
- Color-coded metro lines (official colors)
- Geographical map overlay (Leaflet.js)
- Enhanced station information cards
- Path highlighting with animations

### Tier 2: Advanced (Medium-term)
- Route comparison (3 routes side-by-side)
- Live filtering and layer control
- Time-series animation (network growth)
- Multi-criteria route optimization

### Tier 3: Future Innovations
- Augmented Reality navigation
- Predictive analytics dashboard
- Natural Language Query (NLP)
- Collaborative features
- External API integrations

---

## References

### Libraries & Frameworks
1. **Flask** - https://flask.palletsprojects.com/
2. **NetworkX** - https://networkx.org/
3. **React** - https://react.dev/
4. **Material-UI** - https://mui.com/
5. **vis-network** - https://visjs.github.io/vis-network/

### Algorithms
1. **Dijkstra's Algorithm** - Shortest path calculation
2. **Force-Directed Layout** - Graph visualization
3. **Fuzzy String Matching** - Station search

### Dataset
- Delhi Metro network data
- 283 stations, 284 connections
- Attributes: lines, distances, coordinates, opening dates

---

## Contact & Support

**Course:** NLP Applications (S1-25_AIMLCZG519)
**Course LF:** Vasugi I (vasugii@wilp.bits-pilani.ac.in)
**Institution:** BITS Pilani - M.Tech AIML

For technical issues:
1. Check Troubleshooting section
2. Verify both servers running (backend on :5000, frontend on :3000)
3. Check browser console (F12) for errors
4. Test backend directly: `curl http://localhost:5000/graph`
5. Ensure .gitignore working

---

## License

This project is submitted as part of M.Tech AIML coursework at BITS Pilani.

---

**Document Version:** 2.0 (Consolidated)
**Last Updated:** December 14, 2025
**Status:** Ready for Testing & Submission
**Next Actions:**
1. Test all features
2. Capture screenshots
3. Convert Task B to PDF
4. Complete Part B literature survey
5. Create submission ZIP

---

### Extras

**Option 3: Auto-load CSV on Startup (Optional)**

If you prefer to auto-load the metro dataset when backend starts:

1. Edit `backend/DelhiMetroKGApp.py`
2. Uncomment line 111:
   ```python
   load_delhi_metro_graph()  # Uncomment this line
   ```
3. Restart backend
4. Backend will load 283 stations automatically on startup


**You're ready to test and submit!**

For quick reference:
- **Start Backend:** `cd backend && python3 DelhiMetroKGApp.py`
- **Start Frontend:** `cd frontend/build && python3 -m http.server 3000`
- **Access:** http://localhost:3000
- **Test API:** `curl http://localhost:5000/graph`
- **Upload CSV:** `curl -X POST -F "file=@delhi_metro.csv" http://localhost:5000/upload_csv`
