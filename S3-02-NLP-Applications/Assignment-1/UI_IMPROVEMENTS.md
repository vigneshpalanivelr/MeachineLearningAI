# UI Improvements (Optional Enhancements)

## Quick Fixes to Add Before Testing

### 1. Error Handling in AddRelationForm.jsx

**Replace lines 17-27:**
```javascript
const handleAdd = async () => {
  try {
    if (!e1 || !rel || !e2) {
      alert("Please fill all fields");
      return;
    }
    await api.post("/api/add", {
      entity1: e1,
      relationship: rel,
      entity2: e2,
    });
    alert("Relationship added successfully!");
    setE1("");
    setRel("");
    setE2("");
    refreshGraph();
  } catch (error) {
    alert("Error: " + error.message);
  }
};
```

### 2. Error Handling in CSVUpload.jsx

**Replace lines 8-14:**
```javascript
const uploadCSV = async () => {
  try {
    if (!file) {
      alert("Please select a file");
      return;
    }
    const form = new FormData();
    form.append("file", file);
    const res = await api.post("/api/upload_csv", form);
    alert("✅ Uploaded " + res.data.added + " relationships successfully!");
    setFile(null);
    refreshGraph();
  } catch (error) {
    alert("❌ Upload failed: " + error.message);
  }
};
```

### 3. Error Handling in App.jsx

**Replace loadGraph function (lines 15-18):**
```javascript
const loadGraph = async () => {
  try {
    const res = await api.get("/api/graph");
    setGraphData(res.data);
  } catch (error) {
    console.error("Failed to load graph:", error);
    alert("Failed to load graph. Is the backend running?");
  }
};
```

**Replace showSubgraph function (lines 24-27):**
```javascript
const showSubgraph = async (node, radius) => {
  try {
    if (!node) {
      alert("Please enter a node name");
      return;
    }
    const res = await api.get(`/api/graph?node=${node}&radius=${radius}`);
    setGraphData(res.data);
  } catch (error) {
    alert("Station not found or error: " + error.message);
  }
};
```

**Replace showPath function (lines 29-45):**
```javascript
const showPath = async (src, dst) => {
  try {
    if (!src || !dst) {
      alert("Please enter both source and destination");
      return;
    }
    const res = await api.get(`/api/query?type=path&src=${src}&dst=${dst}`);
    if (!res.data.path || res.data.path.length === 0) {
      alert("No path found between these stations");
      return;
    }
    const nodes = res.data.path.map((n) => ({ id: n, label: n }));
    const edges = [];
    for (let i = 0; i < res.data.path.length - 1; i++) {
      edges.push({
        from: res.data.path[i],
        to: res.data.path[i + 1],
        label: "path",
      });
    }
    setGraphData({ nodes, edges });
    alert(`✅ Found path with ${res.data.path.length - 1} stops`);
  } catch (error) {
    alert("Error finding path: " + error.message);
  }
};
```

### 4. Add Loading State (Optional)

**Add to App.jsx after line 13:**
```javascript
const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
const [loading, setLoading] = useState(false);  // ADD THIS

const loadGraph = async () => {
  setLoading(true);  // ADD THIS
  try {
    const res = await api.get("/api/graph");
    setGraphData(res.data);
  } catch (error) {
    console.error("Failed to load graph:", error);
    alert("Failed to load graph. Is the backend running?");
  } finally {
    setLoading(false);  // ADD THIS
  }
};
```

**Then update GraphView line 62:**
```javascript
<Grid item xs={12} md={8}>
  {loading ? (
    <Typography variant="h6">Loading graph...</Typography>
  ) : (
    <GraphView graphData={graphData} />
  )}
</Grid>
```

## Priority

These improvements are OPTIONAL. Your current code works fine for the assignment.

**Recommended Action:**
1. ✅ Test the application AS-IS first
2. ✅ If everything works, leave it as-is
3. ⚠️ Only add error handling if you encounter issues during testing
