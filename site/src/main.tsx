import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ForecastProvider } from "forecast-web/react";
import { App } from "src/App";
import "src/styles.css";

// ?weights=<repo id or URL> loads other weights, e.g. a local build served over HTTP.
const weights = new URLSearchParams(location.search).get("weights") ?? undefined;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ForecastProvider model="timesfm-2.5" weights={weights}>
      <App />
    </ForecastProvider>
  </StrictMode>,
);
