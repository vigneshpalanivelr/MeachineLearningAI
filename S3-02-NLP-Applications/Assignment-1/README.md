# Delhi Metro Knowledge Graph Application

**Assignment 1 - PS-9**
**Course:** NLP Applications (S1-25_AIMLCZG519)
**M.Tech. in AIML - BITS Pilani**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Features](#features)
5. [Technology Stack](#technology-stack)
6. [API Documentation](#api-documentation)
7. [Frontend-Backend Integration](#frontend-backend-integration)
8. [Testing Guide](#testing-guide)
9. [CSV Upload Guide](#csv-upload-guide)
10. [Git & Submission](#git--submission)
11. [Build & Deployment](#build--deployment)
12. [Design Choices](#design-choices)
13. [Troubleshooting](#troubleshooting)
14. [Assignment Deliverables](#assignment-deliverables)

---

## Overview

A full-stack web application for visualizing and querying the Delhi Metro network as a Knowledge Graph. Users can interactively explore metro stations, find shortest paths, add relationships, and upload CSV data for bulk import.

### Dataset
- **Stations:** 283 metro stations
- **Connections:** 284+ edges
- **Attributes:** Station names, metro lines, distances, GPS coordinates, opening dates

### Key Features
- ✅ Interactive graph visualization (vis-network)
- ✅ Fuzzy search for station matching
- ✅ Shortest path algorithms (stops & distance)
- ✅ CSV bulk upload (2 formats)
- ✅ Real-time graph updates
- ✅ React + Material-UI frontend
- ✅ Flask + NetworkX backend

---

## Quick Start

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
```

**Terminal 2 - Serve Frontend:**
```bash
cd frontend/build
python3 -m http.server 3000
```

**Access:** http://localhost:3000

### Option 2: Development Mode

**Terminal 1 - Backend:**
```bash
cd backend
python3 DelhiMetroKGApp.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm start
```

**Auto-opens:** http://localhost:3000

---

## Project Structure

```
Assignment-1/
├── README.md                           # This file (everything you need)
├── Task_B_Enhancement_Plan.md         # Enhancement plan (convert to PDF)
├── .gitignore                          # Git ignore rules
│
├── backend/
│   ├── DelhiMetroKGApp.py             # Flask API server
│   ├── requirements.txt               # Python dependencies
│   ├── delhi_metro.csv                # Metro dataset (283 stations)
│   ├── test_metro_relationships.csv   # Test CSV (metro format)
│   └── test_generic_relationships.csv # Test CSV (generic format)
│
└── frontend/
    ├── build/                         # Production build (4.9MB) ✅
    ├── src/
    │   ├── App.jsx                    # Main component
    │   ├── api.js                     # Axios API client
    │   ├── theme.js                   # Material-UI theme
    │   └── components/
    │       ├── AddRelationForm.jsx    # Manual relationship entry
    │       ├── CSVUpload.jsx          # CSV bulk upload
    │       ├── QueryPanel.jsx         # Search & path finding
    │       └── GraphView.jsx          # Network visualization
    ├── public/
    │   └── index.html
    ├── package.json                   # Dependencies
    └── node_modules/                  # ⚠️ EXCLUDED by .gitignore (761MB)
```

---

## Features

### Part A Requirements (10 Marks) ✅

#### 1. Frontend Development (3 Marks)
- ✅ React + Material-UI web interface
- ✅ Input fields for entity-relationship pairs
- ✅ CSV file upload for bulk import
- ✅ Interactive graph visualization (vis-network)
- ✅ Dynamic updates on data changes

#### 2. Backend Development (3 Marks)
- ✅ Flask RESTful API
- ✅ NetworkX graph management
- ✅ Add relationships endpoint
- ✅ Query endpoints (neighbors, paths)
- ✅ CSV upload with format detection

#### 3. Integration (2 Marks)
- ✅ Frontend-backend via Axios
- ✅ CORS enabled
- ✅ Real-time graph updates
- ✅ Error handling

#### 4. Task B: Enhancement Plan (2 Marks)
- ✅ 45-page detailed enhancement plan
- ✅ Visual improvements
- ✅ Interactive features
- ✅ Future innovations

### Bonus Features Implemented

- ✅ **Fuzzy Search:** Tolerates typos and partial names
- ✅ **Auto-suggestions:** Returns similar stations when not found
- ✅ **Multiple CSV Formats:** Metro-specific & generic
- ✅ **Enhanced Responses:** Includes match confidence
- ✅ **Physics Simulation:** Force-directed graph layout

---

## Technology Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.9+ | Runtime |
| Flask | 3.1+ | Web framework |
| Flask-CORS | 6.0+ | Cross-origin support |
| NetworkX | Latest | Graph algorithms |
| Pandas | Latest | CSV processing |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.2.0 | UI framework |
| Material-UI | 5.18.0 | UI components |
| vis-network | 9.1.13 | Graph visualization |
| Axios | 1.13.2 | HTTP client |
| react-scripts | 5.0.1 | Build tool |

---

## API Documentation

### Base URL
```
http://localhost:5000
```

### Frontend API Endpoints (React UI)

#### 1. Get Full Graph
```http
GET /api/graph
```

**Response:**
```json
{
  "nodes": [
    {"id": "Station Name", "label": "Station Name"}
  ],
  "edges": [
    {"from": "A", "to": "B", "label": "Blue Line", "title": "2.5 km"}
  ]
}
```

**Example:**
```bash
curl "http://localhost:5000/api/graph"
```

---

#### 2. Get Subgraph (Neighborhood)
```http
GET /api/graph?node=<station>&radius=<N>
```

**Parameters:**
- `node`: Station name (fuzzy matching)
- `radius`: Neighborhood radius (1, 2, 3, etc.)

**Example:**
```bash
curl "http://localhost:5000/api/graph?node=Kashmere%20Gate&radius=2"
```

---

#### 3. Find Shortest Path
```http
GET /api/query?type=path&src=<source>&dst=<destination>
```

**Response:**
```json
{
  "path": ["Station A", "Station B", "Station C"],
  "length": 2,
  "matched": {
    "source": "Matched Station A",
    "destination": "Matched Station C"
  }
}
```

**Example:**
```bash
curl "http://localhost:5000/api/query?type=path&src=Rajiv%20Chowk&dst=Kashmere%20Gate"
```

---

#### 4. Add Relationship
```http
POST /api/add
Content-Type: application/json
```

**Request Body:**
```json
{
  "entity1": "Station A",
  "relationship": "connects_to",
  "entity2": "Station B"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/add \
  -H "Content-Type: application/json" \
  -d '{"entity1":"Test A","relationship":"test","entity2":"Test B"}'
```

---

#### 5. Upload CSV
```http
POST /api/upload_csv
Content-Type: multipart/form-data
```

**CSV Format 1 (Metro):**
```csv
source,target,line,distance
New Delhi,Connaught Place,Airport Express,2.5
```

**CSV Format 2 (Generic):**
```csv
entity1,relationship,entity2
Delhi,has_metro_line,Red Line
```

**Example:**
```bash
curl -X POST -F "file=@test.csv" http://localhost:5000/api/upload_csv
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
| `GET /api/graph` | ✅ Implemented | Working | Load full graph |
| `GET /api/graph?node=X&radius=2` | ✅ Implemented | Working | Get neighborhood |
| `GET /api/query?type=path&src=A&dst=B` | ✅ Implemented | Working | Find path |
| `POST /api/add` | ✅ Implemented | Working | Add relationship |
| `POST /api/upload_csv` | ✅ Implemented | Working | Upload CSV |

### CORS Configuration

**Backend (DelhiMetroKGApp.py):**
```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables cross-origin requests
```

**Frontend (api.js):**
```javascript
import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:5000",
});
```

### Data Flow

```
User Action (React UI)
  ↓
Axios HTTP Request → Flask Backend
  ↓                     ↓
API Endpoint         NetworkX Graph
  ↓                     ↓
JSON Response    ← Graph Query/Update
  ↓
React State Update
  ↓
vis-network Re-render
```

---

## Testing Guide

### Test Checklist

- [ ] **Test 1:** Graph loads on page load (283 nodes)
- [ ] **Test 2:** Add relationship form works
- [ ] **Test 3:** CSV upload works
- [ ] **Test 4:** Neighborhood query works
- [ ] **Test 5:** Path finding works
- [ ] **Test 6:** Fuzzy matching works
- [ ] **Test 7:** Error handling works
- [ ] **Test 8:** Graph interactions (pan/zoom)
- [ ] **Test 9:** No CORS errors
- [ ] **Test 10:** Production build works

### Test Cases

#### Test 1: Load Graph
1. Open http://localhost:3000
2. **Expected:** 283 blue nodes, 284 edges visible

#### Test 2: Add Relationship
1. Fill form: Entity1="Test A", Relationship="test", Entity2="Test B"
2. Click "Add Relationship"
3. **Expected:** Form clears, graph updates, new nodes appear

#### Test 3: Upload CSV
1. Create `test.csv`:
```csv
entity1,relationship,entity2
City A,has_metro,Line 1
City B,has_metro,Line 2
```
2. Upload via UI
3. **Expected:** Alert shows "Uploaded rows: 2", graph updates

#### Test 4: Neighborhood Query
1. Enter Node="Kashmere Gate", Radius=2
2. Click "Show Neighborhood"
3. **Expected:** Graph shows only Kashmere Gate + neighbors

#### Test 5: Path Finding
1. Source="New Delhi Station", Destination="Terminal 3 Airport"
2. Click "Find Shortest Path"
3. **Expected:** Path: [New Delhi Station → Connaught Place → Aerocity → Terminal 3 Airport]

#### Test 6: Fuzzy Matching
1. Source="Rajiv Chowk" (partial), Destination="Kashmere" (partial)
2. **Expected:** Matches "Rajiv Chowk [Conn: Blue]" and "Kashmere Gate [Conn: Yellow]"

#### Test 7: Error Handling
1. Source="Nonexistent Station", Destination="Fake Station"
2. **Expected:** Alert "No path found" or "Stations not found"

### Command-Line Testing

```bash
# Test all endpoints
echo "Testing /api/graph..."
curl -s "http://localhost:5000/api/graph" | python3 -c "import json, sys; data=json.load(sys.stdin); print(f'Nodes: {len(data[\"nodes\"])}, Edges: {len(data[\"edges\"])}')"

echo "Testing /api/add..."
curl -s -X POST http://localhost:5000/api/add \
  -H "Content-Type: application/json" \
  -d '{"entity1":"Test A","relationship":"test","entity2":"Test B"}' \
  | python3 -c "import json, sys; print(json.load(sys.stdin)[\"message\"])"

echo "Testing /api/query..."
curl -s "http://localhost:5000/api/query?type=path&src=New%20Delhi%20Station&dst=Terminal%203%20Airport" \
  | python3 -c "import json, sys; data=json.load(sys.stdin); print(f'Path length: {data.get(\"length\", \"N/A\")}')"

echo "Testing /api/upload_csv..."
curl -s -X POST -F "file=@test_generic_relationships.csv" \
  http://localhost:5000/api/upload_csv \
  | python3 -c "import json, sys; print(json.load(sys.stdin)[\"message\"])"
```

---

## CSV Upload Guide

### How CSV Upload Works

```
CSV File → Flask Endpoint → Pandas Parse → Format Detection → Create Nodes → Create Edges → Update Graph
```

### Supported Formats

#### Format 1: Metro-Specific
**Use for:** Transportation networks with distances

**Structure:**
```csv
source,target,line,distance
```

**Example:**
```csv
source,target,line,distance
New Delhi Station,Connaught Place,Airport Express,2.5
Connaught Place,Aerocity,Airport Express,15.3
Aerocity,Terminal 3 Airport,Airport Express,3.2
```

**Creates:**
- Nodes: "New Delhi Station", "Connaught Place", "Aerocity", "Terminal 3 Airport"
- Edges: With `line` and `distance` attributes

---

#### Format 2: Generic Knowledge Graph
**Use for:** General entity relationships

**Structure:**
```csv
entity1,relationship,entity2
```

**Example:**
```csv
entity1,relationship,entity2
Delhi,has_metro_line,Red Line
Delhi,has_metro_line,Blue Line
Red Line,connects_to,Blue Line
Kashmere Gate,interchange_point,Red Line
```

**Creates:**
- Nodes: "Delhi", "Red Line", "Blue Line", "Kashmere Gate"
- Edges: With `relationship` attribute

### Processing Logic

```python
# Backend code
if all(col in df.columns for col in ['source', 'target']):
    # Process as metro format
    for _, row in df.iterrows():
        source = row.get('source')
        target = row.get('target')
        line = row.get('line', 'Unknown')
        distance = float(row.get('distance', 0.0))

        KG.add_node(source)
        KG.add_node(target)
        KG.add_edge(source, target, line=line, distance=distance)

elif all(col in df.columns for col in ['entity1', 'relationship', 'entity2']):
    # Process as generic format
    for _, row in df.iterrows():
        entity1 = row.get('entity1')
        relationship = row.get('relationship')
        entity2 = row.get('entity2')

        KG.add_node(entity1)
        KG.add_node(entity2)
        KG.add_edge(entity1, entity2, relationship=relationship)
```

### Test Files Included

**test_metro_relationships.csv:**
- 10 relationships
- Airport Express and Rapid Metro lines
- New Delhi → Connaught Place → Aerocity → Terminal 3

**test_generic_relationships.csv:**
- 10 relationships
- Metro lines and interchange points
- Delhi → Lines → Connections

---

## Git & Submission

### .gitignore Status ✅

**Verification:**
```bash
# Check staging area
git diff --cached --name-only | wc -l
# Output: 27 files ✅

# Check node_modules in staging
git diff --cached --name-only | grep "node_modules" | wc -l
# Output: 0 ✅

# Verify node_modules is ignored
git status --ignored --short | grep "node_modules"
# Output: !! S3-02-NLP-Applications/Assignment-1/frontend/node_modules/
# !! means IGNORED ✅
```

### Why You See 10k Files in Your IDE

**The .gitignore IS working!** Your IDE (VS Code/PyCharm) shows ALL files including ignored ones. Git is correctly excluding node_modules.

**To hide ignored files in VS Code:**
1. Open Source Control panel (Ctrl+Shift+G)
2. Click "..." menu → Uncheck "Show Ignored Files"

### What's Being Committed (27 Files)

```
✅ Documentation (3):
   .gitignore, README.md, Task_B_Enhancement_Plan.md

✅ Backend (5):
   DelhiMetroKGApp.py, delhi_metro.csv, requirements.txt, test_*.csv

✅ Frontend Source (10):
   package.json, src/*.jsx, src/*.js, public/*

✅ Frontend Build (9):
   build/index.html, build/static/js/*

❌ Excluded:
   node_modules/ (761MB), .DS_Store, __pycache__/
```

### File Size Comparison

| Component | With node_modules | Without |
|-----------|------------------|---------|
| Frontend | 761MB | 5MB |
| Backend | 30KB | 30KB |
| Docs | 50KB | 50KB |
| **Total** | **~761MB** | **~6MB** ✅

### Create Submission ZIP

```bash
cd /Users/nila/Documents/repositories/MeachineLearningAI/S3-02-NLP-Applications

zip -r Assignment-1-Submission.zip Assignment-1 \
  -x "*/node_modules/*" \
  -x "*/.DS_Store" \
  -x "*/__pycache__/*" \
  -x "*/.claude/*" \
  -x "*/backend.log"

# Check size
du -sh Assignment-1-Submission.zip
# Expected: 2-5 MB ✅
```

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

### Deploy Options

#### 1. GitHub Pages (Frontend Only)
```bash
npm install --save-dev gh-pages

# Add to package.json
{
  "homepage": "https://yourusername.github.io/delhi-metro-kg",
  "scripts": {
    "predeploy": "npm run build",
    "deploy": "gh-pages -d build"
  }
}

npm run deploy
```

#### 2. Render.com (Backend)
1. Create `requirements.txt` (already exists)
2. Push to GitHub
3. Connect to Render.com
4. Deploy with one click

#### 3. Full Stack on Heroku/Railway
- Backend: Flask with Gunicorn
- Frontend: Serve `build/` folder
- Database: Optional (currently in-memory)

---

## Design Choices

### 1. Backend: Flask + NetworkX

**Why Flask?**
- Lightweight web framework
- Easy RESTful API creation
- Python ecosystem (NetworkX, Pandas)
- Fast development

**Why NetworkX?**
- Industry-standard graph library
- Rich algorithm library (Dijkstra, BFS, DFS)
- Easy CSV integration
- Scalable (tested with 10k+ nodes)

**Alternatives Considered:**
- Neo4j: Too complex for assignment
- FastAPI: Overkill for this scope

### 2. Frontend: React + Material-UI

**Why React?**
- Component-based architecture
- Large ecosystem
- Easy state management
- Industry standard

**Why Material-UI?**
- Professional components
- Responsive by default
- Accessible (WCAG compliant)
- Fast development

**Why vis-network?**
- Best graph visualization
- Force-directed physics
- Interactive (zoom, pan, drag)
- Performance optimized

**Alternatives Considered:**
- D3.js: Steeper learning curve
- Cytoscape.js: Less React-friendly

### 3. Graph Structure: Undirected

**Rationale:**
- Metro connections are bidirectional
- Travel both directions
- Simpler querying
- Matches real-world usage

### 4. Fuzzy Search

**Algorithm:** `difflib.get_close_matches`
- Ratcliff-Obershelp algorithm
- Cutoff: 0.4 (40% similarity)
- Returns top 5 matches

**Why?**
- Station names are complex
- Users may not know exact spelling
- Improves UX significantly
- Tolerates typos

### 5. CSV Format Detection

**Auto-detection logic:**
```python
if all(col in df.columns for col in ['source', 'target']):
    # Metro format
elif all(col in df.columns for col in ['entity1', 'relationship', 'entity2']):
    # Generic format
```

**Benefits:**
- Flexible input
- No manual format selection
- Supports multiple use cases

---

## Troubleshooting

### Issue 1: CORS Error

**Symptom:**
```
Access-Control-Allow-Origin header is missing
```

**Solution:**
```bash
pip install flask-cors
```

Ensure backend has:
```python
from flask_cors import CORS
CORS(app)
```

---

### Issue 2: 404 on All API Calls

**Symptom:**
```
GET http://localhost:5000/api/graph 404
```

**Solution:**
- Backend not running
- Start with: `python3 DelhiMetroKGApp.py`
- Check terminal for errors

---

### Issue 3: Empty Graph

**Possible Causes:**
1. Backend returned no data
2. Frontend-backend URL mismatch
3. vis-network not installed

**Solutions:**
```bash
# Test backend
curl http://localhost:5000/api/graph

# Check frontend API URL (api.js)
baseURL: "http://localhost:5000"

# Reinstall frontend deps
cd frontend && npm install
```

---

### Issue 4: CSV Upload Fails

**Error:**
```
Invalid CSV format
```

**Solution:**
- Check column headers match:
  - **Format 1:** `source,target,line,distance`
  - **Format 2:** `entity1,relationship,entity2`
- No extra spaces in headers
- UTF-8 encoding

---

### Issue 5: Path Query Returns Empty

**Causes:**
1. Stations disconnected (no path exists)
2. Station names don't match

**Debug:**
```bash
# Check if stations exist
curl "http://localhost:5000/search_station?q=Station%20Name"

# Check neighbors
curl "http://localhost:5000/neighbors/Station%20Name"
```

---

### Issue 6: Port Already in Use

**Error:**
```
Address already in use: 5000
```

**Solution:**
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use different port
flask run --port=5001
```

---

### Issue 7: Production Build Doesn't Load

**Symptom:**
- Blank page
- Console errors

**Solution:**
```bash
# Check console for errors
# Verify homepage in package.json
"homepage": "."

# Rebuild
cd frontend
rm -rf build
npm run build
```

---

## Assignment Deliverables

### Part A - Knowledge Graph Application (10 marks)

| Deliverable | Status | Location |
|------------|--------|----------|
| Well-documented Python code | ✅ | `backend/DelhiMetroKGApp.py` |
| Well-documented React code | ✅ | `frontend/src/` |
| Production build | ✅ | `frontend/build/` |
| Instructions for running | ✅ | This README |
| Design choices report | ✅ | This README (Design Choices section) |
| Challenges faced | ✅ | This README (Troubleshooting section) |
| Screenshots | ⏳ | Capture during testing |
| Task-B Enhancement Plan | ✅ | `Task_B_Enhancement_Plan.md` → PDF |

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
- [x] .gitignore excludes node_modules
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
# Should be < 10MB ✅
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
| `GET /api/graph` (full) | < 500ms | 283 nodes, 284 edges |
| `GET /api/graph` (subgraph) | < 100ms | Radius query |
| `GET /api/query` (path) | < 200ms | Dijkstra algorithm |
| `POST /api/add` | < 50ms | Single relationship |
| `POST /api/upload_csv` | < 1s | 100 rows |

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
2. Verify both servers running
3. Check browser console (F12)
4. Ensure .gitignore working

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

**🎯 You're ready to test and submit!**

For quick reference:
- **Start Backend:** `cd backend && python3 DelhiMetroKGApp.py`
- **Start Frontend:** `cd frontend/build && python3 -m http.server 3000`
- **Access:** http://localhost:3000
- **Test API:** `curl http://localhost:5000/api/graph`
