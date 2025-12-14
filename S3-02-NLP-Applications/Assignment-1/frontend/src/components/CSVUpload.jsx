import React, { useState } from "react";
import { Card, CardContent, Button, Typography } from "@mui/material";
import api from "../api";

export default function CSVUpload({ refreshGraph }) {
  const [file, setFile] = useState(null);

  const uploadCSV = async () => {
    try {
      if (!file) {
        alert("Please select a file first");
        return;
      }

      const form = new FormData();
      form.append("file", file);

      console.log("Uploading CSV file:", file.name);
      const res = await api.post("/api/upload_csv", form);

      console.log("Upload response:", res.data);

      if (res.data.error) {
        alert(`Upload failed: ${res.data.error}`);
        return;
      }

      alert(
        `✅ CSV Upload Successful!\n\n` +
        `Added: ${res.data.added} relationships\n` +
        `Total Stations: ${res.data.total_stations}\n` +
        `Total Connections: ${res.data.total_connections}`
      );

      setFile(null);

      // Reset file input
      const fileInput = document.querySelector('input[type="file"]');
      if (fileInput) fileInput.value = '';

      // Refresh graph to show new data
      await refreshGraph();

    } catch (error) {
      console.error("CSV upload error:", error);
      alert(`❌ Upload failed: ${error.response?.data?.error || error.message}`);
    }
  };

  return (
    <Card elevation={4} sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h5">Upload CSV</Typography>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0])}
          style={{ marginTop: "16px" }}
        />
        <Button
          variant="contained"
          sx={{ mt: 2 }}
          onClick={uploadCSV}
          disabled={!file}
        >
          Upload
        </Button>
      </CardContent>
    </Card>
  );
}
