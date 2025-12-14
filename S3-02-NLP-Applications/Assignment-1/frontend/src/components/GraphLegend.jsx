import React from "react";
import { Box, Typography, Paper } from "@mui/material";

export default function GraphLegend({ nodeCount, edgeCount }) {
  return (
    <Paper
      elevation={3}
      sx={{
        position: "absolute",
        top: 16,
        right: 16,
        padding: 2,
        backgroundColor: "rgba(255, 255, 255, 0.95)",
        borderRadius: 2,
        minWidth: 200,
      }}
    >
      <Typography variant="h6" gutterBottom sx={{ fontSize: 16, fontWeight: 600 }}>
        📊 Graph Statistics
      </Typography>

      <Box sx={{ mt: 1 }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            mb: 1,
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center" }}>
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                backgroundColor: "#64b5f6",
                border: "2px solid #1976d2",
                mr: 1,
              }}
            />
            <Typography variant="body2">Stations:</Typography>
          </Box>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            {nodeCount}
          </Typography>
        </Box>

        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center" }}>
            <Box
              sx={{
                width: 20,
                height: 2,
                backgroundColor: "#848484",
                mr: 1,
              }}
            />
            <Typography variant="body2">Connections:</Typography>
          </Box>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            {edgeCount}
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
}
