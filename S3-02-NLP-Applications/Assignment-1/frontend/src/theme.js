import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    primary: {
      main: "#1976d2",
    },
    secondary: {
      main: "#9c27b0",
    },
    background: {
      default: "#f4f6f8",
    },
  },
  typography: {
    h4: { fontWeight: 700 },
    h5: { fontWeight: 600 },
    fontFamily: "Roboto, sans-serif",
  },
});

export default theme;
