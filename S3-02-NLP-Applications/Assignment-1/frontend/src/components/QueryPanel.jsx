import React, { useState } from "react";
import {
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Stack,
} from "@mui/material";

export default function QueryPanel({ showSubgraph, showPath, resetGraph }) {
  const [node, setNode] = useState("");
  const [radius, setRadius] = useState(1);
  const [src, setSrc] = useState("");
  const [dst, setDst] = useState("");

  return (
    <Card elevation={4} sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          Query Graph
        </Typography>

        <Stack spacing={2} mt={2}>
          <Typography variant="subtitle2" color="text.secondary">
            Neighborhood Query
          </Typography>
          <TextField
            label="Station Name"
            placeholder="e.g., Rajiv Chowk"
            value={node}
            onChange={(e) => setNode(e.target.value)}
            size="small"
          />
          <TextField
            label="Radius (stops)"
            type="number"
            value={radius}
            onChange={(e) => setRadius(parseInt(e.target.value) || 1)}
            size="small"
            inputProps={{ min: 1, max: 10 }}
          />

          <Button
            variant="contained"
            onClick={() => showSubgraph(node, radius)}
            disabled={!node}
          >
            Show Neighborhood
          </Button>

          <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 2 }}>
            Path Finding
          </Typography>
          <TextField
            label="Source Station"
            placeholder="e.g., Kashmere Gate"
            value={src}
            onChange={(e) => setSrc(e.target.value)}
            size="small"
          />
          <TextField
            label="Destination Station"
            placeholder="e.g., Botanical Garden"
            value={dst}
            onChange={(e) => setDst(e.target.value)}
            size="small"
          />

          <Button
            variant="contained"
            onClick={() => showPath(src, dst)}
            disabled={!src || !dst}
          >
            Find Shortest Path
          </Button>

          <Button
            variant="outlined"
            color="secondary"
            onClick={resetGraph}
            sx={{ mt: 2 }}
          >
            Reset to Full Graph
          </Button>
        </Stack>
      </CardContent>
    </Card>
  );
}
