import { defineConfig } from "tsup";

export default defineConfig({
  entry: { index: "src/index.ts", react: "src/react.tsx", tabular: "src/tabular/index.ts", worker: "src/core/worker.ts" },
  format: ["esm"],
  dts: { entry: { index: "src/index.ts", react: "src/react.tsx", tabular: "src/tabular/index.ts" } },
  target: "es2022",
  platform: "browser",
  splitting: true,
  clean: true,
  sourcemap: true,
  external: ["react", "onnxruntime-web"],
});
