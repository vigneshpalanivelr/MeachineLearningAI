# CSV Relationship Documentation

## Table of Contents
1. [Overview](#overview)
2. [How CSV Upload Creates Relationships](#how-csv-upload-creates-relationships)
3. [Supported CSV Formats](#supported-csv-formats)
4. [Data Model and Graph Structure](#data-model-and-graph-structure)
5. [CSV File Creation Process](#csv-file-creation-process)
6. [Usage Examples](#usage-examples)
7. [Technical Implementation](#technical-implementation)

---

## Overview

The Delhi Metro Knowledge Graph application uses **NetworkX** to model metro stations and their connections as a graph data structure. The CSV upload feature allows bulk import of relationships, making it easy to populate the knowledge graph with multiple entities and connections at once.

**Key Concepts:**
- **Nodes**: Represent entities (e.g., metro stations, lines, cities)
- **Edges**: Represent relationships between entities (e.g., connections between stations)
- **Attributes**: Additional metadata stored on nodes/edges (e.g., distance, line name)

---

## How CSV Upload Creates Relationships

### Process Flow

```
CSV File Upload → Flask Endpoint → Parse CSV → Validate Format → Create Nodes → Create Edges → Update Graph
```

### Step-by-Step Breakdown

#### 1. **File Reception**
```python
file = request.files['file']  # Receive uploaded CSV file
```
- Flask receives the CSV file via HTTP POST request
- File is validated (must be .csv extension)

#### 2. **CSV Parsing**
```python
csv_data = file.read().decode('utf-8')
df = pd.read_csv(io.StringIO(csv_data))
```
- File content is decoded from bytes to UTF-8 string
- Pandas reads the CSV into a DataFrame for easy manipulation

#### 3. **Format Detection**
The system automatically detects which CSV format is being used:

**Format 1: Metro-Specific** (columns: `source`, `target`, `line`, `distance`)
```python
if all(col in df.columns for col in ['source', 'target']):
    # Process as metro-specific format
```

**Format 2: Generic** (columns: `entity1`, `relationship`, `entity2`)
```python
elif all(col in df.columns for col in ['entity1', 'relationship', 'entity2']):
    # Process as generic knowledge graph format
```

#### 4. **Node Creation**
For each row in the CSV:
```python
if source not in KG:
    KG.add_node(source)  # Create node if it doesn't exist
if target not in KG:
    KG.add_node(target)  # Create node if it doesn't exist
```

**Why this matters:**
- Ensures all entities exist in the graph before creating relationships
- Prevents orphaned edges (edges without corresponding nodes)
- Automatically handles duplicate nodes (NetworkX ignores duplicate add_node calls)

#### 5. **Edge Creation**
After nodes exist, relationships are created:

**Metro Format:**
```python
KG.add_edge(source, target, line=line, distance=distance)
```
Creates a bidirectional edge with metadata:
- `line`: Metro line name (e.g., "Blue Line")
- `distance`: Distance in kilometers

**Generic Format:**
```python
KG.add_edge(entity1, entity2, relationship=relationship)
```
Creates a relationship with a semantic label:
- `relationship`: Type of connection (e.g., "connects_to", "interchange_point")

#### 6. **Response**
```python
return jsonify({
    "message": "CSV uploaded successfully",
    "relationships_added": added_count,
    "total_stations": KG.number_of_nodes(),
    "total_connections": KG.number_of_edges()
})
```

---

## Supported CSV Formats

### Format 1: Metro-Specific Format

**Structure:**
```csv
source,target,line,distance
```

**Columns:**
- `source` (required): Starting station name
- `target` (required): Destination station name
- `line` (optional): Metro line name (defaults to "Unknown")
- `distance` (optional): Distance between stations in km (defaults to 0.0)

**Example:**
```csv
source,target,line,distance
New Delhi Station,Connaught Place,Airport Express,2.5
Connaught Place,Aerocity,Airport Express,15.3
Aerocity,Terminal 3 Airport,Airport Express,3.2
```

**Graph Representation:**
```
New Delhi Station --[2.5 km, Airport Express]--> Connaught Place
Connaught Place --[15.3 km, Airport Express]--> Aerocity
Aerocity --[3.2 km, Airport Express]--> Terminal 3 Airport
```

**Use Case:**
- Transportation networks
- Route planning
- Distance-based path finding

---

### Format 2: Generic Knowledge Graph Format

**Structure:**
```csv
entity1,relationship,entity2
```

**Columns:**
- `entity1` (required): First entity/node
- `relationship` (required): Semantic relationship type
- `entity2` (required): Second entity/node

**Example:**
```csv
entity1,relationship,entity2
Delhi,has_metro_line,Red Line
Red Line,connects_to,Blue Line
Kashmere Gate,interchange_point,Red Line
```

**Graph Representation:**
```
Delhi --[has_metro_line]--> Red Line
Red Line --[connects_to]--> Blue Line
Kashmere Gate --[interchange_point]--> Red Line
```

**Use Case:**
- General knowledge graphs
- Semantic networks
- Ontology modeling
- Entity relationship mapping

---

## Data Model and Graph Structure

### NetworkX Graph Properties

The application uses an **undirected graph** (`nx.Graph()`):

```python
KG = nx.Graph()  # Undirected graph
```

**Why Undirected?**
- Metro connections are bidirectional (can travel both ways)
- Symmetric relationships (if A connects to B, then B connects to A)

### Node Structure

**Node Representation:**
```python
{
    "station_name": {
        "station_id": int,
        "line": str,
        "opened": str,
        "layout": str,
        "latitude": float,
        "longitude": float,
        "distance_from_start": float
    }
}
```

**Example:**
```python
{
    "Connaught Place": {
        "station_id": None,  # Not from original dataset
        "line": None,
        "latitude": None,
        "longitude": None
    }
}
```

### Edge Structure

**Edge Representation (Metro Format):**
```python
{
    ("Station A", "Station B"): {
        "line": "Blue Line",
        "distance": 2.5
    }
}
```

**Edge Representation (Generic Format):**
```python
{
    ("Entity 1", "Entity 2"): {
        "relationship": "connects_to"
    }
}
```

---

## CSV File Creation Process

### Original Dataset: `delhi_metro.csv`

The base dataset contains Delhi Metro station information:

**Structure:**
```csv
ID (Station ID),Station Names,Dist. From First Station(km),Metro Line,Opened(Year),Layout,Latitude,Longitude
```

**How it's loaded:**
```python
def load_delhi_metro_graph(csv_path="delhi_metro.csv"):
    df = pd.read_csv(csv_path)

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
        distance_km = abs(s2["Distance"] - s1["Distance"])
        KG.add_edge(s1["Station"], s2["Station"], line=s1["Line"], distance=distance_km)
```

**Result:**
- 283 stations (nodes)
- 284 connections (edges)

---

### Test File 1: `test_metro_relationships.csv`

**Purpose:** Add new metro connections for Airport Express and Rapid Metro lines

**Creation Process:**

1. **Identify New Routes:**
   - Airport Express: New Delhi → Connaught Place → Aerocity → Terminal 3
   - Rapid Metro: Gurgaon Sector 21 → Cyber City phases
   - Blue Line Extension: Noida sectors

2. **Calculate Distances:**
   - Based on approximate real-world distances
   - Measured in kilometers

3. **Format Data:**
```csv
source,target,line,distance
New Delhi Station,Connaught Place,Airport Express,2.5
Connaught Place,Aerocity,Airport Express,15.3
Aerocity,Terminal 3 Airport,Airport Express,3.2
Gurgaon Sector 21,Sikanderpur Metro,Rapid Metro,1.8
Sikanderpur Metro,Phase 1 Cyber City,Rapid Metro,2.1
Phase 1 Cyber City,Phase 2 Cyber City,Rapid Metro,1.5
Phase 2 Cyber City,Phase 3 Cyber City,Rapid Metro,1.3
Noida City Center,Noida Sector 34,Blue Line Extension,2.7
Noida Sector 34,Noida Sector 52,Blue Line Extension,3.1
Noida Sector 52,Noida Sector 61,Blue Line Extension,2.9
```

4. **Upload Result:**
   - 10 new relationships added
   - 10 new stations created
   - Total: 293 stations, 293 connections

---

### Test File 2: `test_generic_relationships.csv`

**Purpose:** Add semantic relationships about metro lines and interchange points

**Creation Process:**

1. **Define Entity Relationships:**
   - Cities and their metro lines
   - Line-to-line connections
   - Interchange point mappings

2. **Choose Relationship Types:**
   - `has_metro_line`: City owns a metro line
   - `connects_to`: Lines that intersect
   - `interchange_point`: Stations where lines meet

3. **Format Data:**
```csv
entity1,relationship,entity2
Delhi,has_metro_line,Red Line
Delhi,has_metro_line,Blue Line
Delhi,has_metro_line,Yellow Line
Red Line,connects_to,Blue Line
Red Line,connects_to,Yellow Line
Blue Line,connects_to,Yellow Line
Kashmere Gate,interchange_point,Red Line
Kashmere Gate,interchange_point,Yellow Line
Rajiv Chowk,interchange_point,Blue Line
Rajiv Chowk,interchange_point,Yellow Line
```

4. **Upload Result:**
   - 10 new relationships added
   - 6 new entities created (Delhi, Red Line, Blue Line, Yellow Line, etc.)
   - Total: 299 stations, 303 connections

---

## Usage Examples

### Example 1: Upload Metro Relationships

**Command:**
```bash
curl -X POST -F "file=@test_metro_relationships.csv" \
  http://127.0.0.1:5000/upload_csv
```

**Response:**
```json
{
    "message": "CSV uploaded successfully",
    "relationships_added": 10,
    "total_stations": 293,
    "total_connections": 293
}
```

**What Happened:**
1. 10 rows read from CSV
2. 20 nodes checked/created (source + target for each row)
3. 10 edges created with line and distance attributes
4. Graph updated in memory

---

### Example 2: Query Uploaded Data

**Check Neighbors:**
```bash
curl "http://127.0.0.1:5000/neighbors/Connaught%20Place"
```

**Response:**
```json
{
    "fuzzy_match": false,
    "matched_station": "Connaught Place",
    "neighbors": [
        {
            "station": "New Delhi Station",
            "line": "Airport Express",
            "distance_km": 2.5
        },
        {
            "station": "Aerocity",
            "line": "Airport Express",
            "distance_km": 15.3
        }
    ],
    "query": "Connaught Place"
}
```

---

### Example 3: Find Path Using Uploaded Data

**Command:**
```bash
curl "http://127.0.0.1:5000/shortest_path?source=New%20Delhi%20Station&target=Terminal%203%20Airport"
```

**Response:**
```json
{
    "query": {
        "source": "New Delhi Station",
        "target": "Terminal 3 Airport"
    },
    "matched": {
        "source": "New Delhi Station",
        "target": "Terminal 3 Airport"
    },
    "fuzzy_match": false,
    "shortest_path": [
        "New Delhi Station",
        "Connaught Place",
        "Aerocity",
        "Terminal 3 Airport"
    ],
    "stops": 3
}
```

**Explanation:**
- CSV created a linear route: New Delhi → Connaught → Aerocity → Terminal 3
- Shortest path algorithm found this exact route
- 3 intermediate stops required

---

## Technical Implementation

### Code Flow for CSV Upload

```python
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    # Step 1: Validate file
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be a CSV"}), 400

    # Step 2: Parse CSV
    csv_data = file.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_data))

    # Step 3: Detect format and process
    added_count = 0

    # Metro-specific format
    if all(col in df.columns for col in ['source', 'target']):
        for _, row in df.iterrows():
            source = row.get('source')
            target = row.get('target')
            line = row.get('line', 'Unknown')
            distance = float(row.get('distance', 0.0))

            # Create nodes
            if source not in KG:
                KG.add_node(source)
            if target not in KG:
                KG.add_node(target)

            # Create edge
            KG.add_edge(source, target, line=line, distance=distance)
            added_count += 1

    # Generic format
    elif all(col in df.columns for col in ['entity1', 'relationship', 'entity2']):
        for _, row in df.iterrows():
            entity1 = row.get('entity1')
            relationship = row.get('relationship')
            entity2 = row.get('entity2')

            # Create nodes
            if entity1 not in KG:
                KG.add_node(entity1)
            if entity2 not in KG:
                KG.add_node(entity2)

            # Create edge
            KG.add_edge(entity1, entity2, relationship=relationship)
            added_count += 1

    # Step 4: Return results
    return jsonify({
        "message": "CSV uploaded successfully",
        "relationships_added": added_count,
        "total_stations": KG.number_of_nodes(),
        "total_connections": KG.number_of_edges()
    })
```

### Key Design Decisions

1. **Automatic Node Creation:**
   - Nodes are created automatically if they don't exist
   - Simplifies CSV creation (no need to pre-register entities)

2. **Flexible Format Support:**
   - Two formats supported based on use case
   - Format detected automatically via column inspection

3. **Attribute Handling:**
   - Optional attributes default to sensible values
   - Missing columns don't cause errors

4. **Error Handling:**
   - Validates file type
   - Catches parsing exceptions
   - Returns clear error messages

5. **In-Memory Graph:**
   - Graph stored in memory (KG variable)
   - Fast lookups and updates
   - Note: Data lost on server restart (consider persistence for production)

---

## Benefits of CSV Upload Approach

### 1. **Bulk Data Import**
- Add hundreds of relationships in seconds
- No need to call API individually for each relationship

### 2. **Data Portability**
- CSV is universal format
- Easy to create in Excel, Google Sheets, Python, etc.

### 3. **Version Control**
- CSV files can be tracked in git
- Changes are visible in diffs

### 4. **Collaboration**
- Non-technical users can create CSV files
- Domain experts can contribute data

### 5. **Testing**
- Easy to create test datasets
- Repeatable imports for development

---

## Conclusion

The CSV upload feature enables rapid population of the knowledge graph with minimal effort. By supporting both metro-specific and generic formats, it accommodates different use cases while maintaining a simple, intuitive interface.

**Key Takeaways:**
- CSV rows → Graph edges + nodes
- Automatic format detection
- Bidirectional relationships (undirected graph)
- Metadata preserved on edges
- Scalable to thousands of relationships

This approach makes the Delhi Metro Knowledge Graph accessible to users without programming knowledge while providing powerful capabilities for data modeling and querying.
