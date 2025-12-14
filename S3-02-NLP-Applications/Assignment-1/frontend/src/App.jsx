import React, { useState, useEffect } from "react";
import { ThemeProvider } from "@mui/material/styles";
import { Container, Grid, Typography } from "@mui/material";
import theme from "./theme";
import api from "./api";

import AddRelationForm from "./components/AddRelationForm";
import CSVUpload from "./components/CSVUpload";
import QueryPanel from "./components/QueryPanel";
import GraphView from "./components/GraphView";
import GraphLegend from "./components/GraphLegend";

function App() {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(false);
  const [nodeCount, setNodeCount] = useState(0);
  const [edgeCount, setEdgeCount] = useState(0);

  const loadGraph = async () => {
    try {
      setLoading(true);
      console.log("Loading full graph...");
      const res = await api.get("/graph");
      console.log("Graph loaded:", res.data.nodes.length, "nodes,", res.data.edges.length, "edges");
      setGraphData(res.data);
      setNodeCount(res.data.nodes.length);
      setEdgeCount(res.data.edges.length);
    } catch (error) {
      console.error("Failed to load graph:", error);
      alert("Failed to load graph. Is the backend running on port 5000?");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadGraph();
  }, []);

  const showSubgraph = async (node, radius) => {
    try {
      if (!node) {
        alert("Please enter a station name");
        return;
      }

      setLoading(true);
      console.log(`Querying subgraph: node=${node}, radius=${radius}`);
      const res = await api.get(`/graph?node=${node}&radius=${radius}`);
      console.log("Subgraph result:", res.data);

      if (res.data.error) {
        alert(`Error: ${res.data.error}`);
        return;
      }

      setGraphData(res.data);
      setNodeCount(res.data.nodes.length);
      setEdgeCount(res.data.edges.length);
      alert(`Showing ${res.data.nodes.length} stations within ${radius} stop(s) of "${res.data.nodes[0]?.label || node}"`);
    } catch (error) {
      console.error("Subgraph query error:", error);
      alert(`Station not found: ${error.response?.data?.error || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const showPath = async (src, dst) => {
    try {
      if (!src || !dst) {
        alert("Please enter both source and destination stations");
        return;
      }

      setLoading(true);
      console.log(`Finding path: ${src} → ${dst}`);
      const res = await api.get(`/paths?source=${src}&destination=${dst}`);
      console.log("Path result:", res.data);

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
          color: { color: "#4caf50" },
          width: 3,
        });
      }

      setGraphData({ nodes, edges });
      setNodeCount(nodes.length);
      setEdgeCount(edges.length);
      alert(
        `Path Found!\n\n` +
        `From: ${res.data.matched.source}\n` +
        `To: ${res.data.matched.destination}\n` +
        `Stops: ${res.data.length}\n\n` +
        `Route: ${res.data.path.join(" → ")}`
      );
    } catch (error) {
      console.error("Path finding error:", error);
      alert(`Error: ${error.response?.data?.error || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="xl" sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          🚇 Delhi Metro Knowledge Graph Visualizer
        </Typography>

        {loading && (
          <Typography variant="body2" color="primary" gutterBottom>
            ⏳ Loading...
          </Typography>
        )}

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <AddRelationForm refreshGraph={loadGraph} />
            <CSVUpload refreshGraph={loadGraph} />
            <QueryPanel
              showSubgraph={showSubgraph}
              showPath={showPath}
              resetGraph={loadGraph}
            />
          </Grid>

          <Grid item xs={12} md={8} sx={{ position: "relative" }}>
            <GraphView graphData={graphData} />
            <GraphLegend nodeCount={nodeCount} edgeCount={edgeCount} />
          </Grid>
        </Grid>
      </Container>
    </ThemeProvider>
  );
}

export default App;
