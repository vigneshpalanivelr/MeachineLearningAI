import React, { useState } from "react";
import {
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Stack,
} from "@mui/material";
import api from "../api";

export default function AddRelationForm({ refreshGraph }) {
  const [e1, setE1] = useState("");
  const [rel, setRel] = useState("");
  const [e2, setE2] = useState("");

  const handleAdd = async () => {
    await api.post("/add_relationship", {
      entity1: e1,
      relationship: rel,
      entity2: e2,
    });
    setE1("");
    setRel("");
    setE2("");
    refreshGraph();
  };

  return (
    <Card elevation={4} sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h5">Add Relationship</Typography>
        <Stack spacing={2} mt={2}>
          <TextField
            label="Entity 1"
            value={e1}
            onChange={(e) => setE1(e.target.value)}
          />
          <TextField
            label="Relationship"
            value={rel}
            onChange={(e) => setRel(e.target.value)}
          />
          <TextField
            label="Entity 2"
            value={e2}
            onChange={(e) => setE2(e.target.value)}
          />
          <Button variant="contained" onClick={handleAdd}>
            Add Relationship
          </Button>
        </Stack>
      </CardContent>
    </Card>
  );
}
