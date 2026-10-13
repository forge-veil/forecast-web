import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "src/App";
import { ModelsProvider } from "src/lib/models";
import "src/styles.css";

// ?weights=<repo id or URL> loads other TimesFM weights, e.g. a local build served over HTTP.
const weights = new URLSearchParams(location.search).get("weights") ?? undefined;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ModelsProvider weights={weights}>
      <App />
    </ModelsProvider>
  </StrictMode>,
);
