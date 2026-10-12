import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { TimesFMProvider } from "timesfm-web/react";
import { App } from "src/App";
import "src/styles.css";

// ?model=<repo id or URL> loads other weights, e.g. a local build served over HTTP.
const model = new URLSearchParams(location.search).get("model") ?? undefined;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <TimesFMProvider model={model}>
      <App />
    </TimesFMProvider>
  </StrictMode>,
);
