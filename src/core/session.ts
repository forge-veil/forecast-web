import * as ort from "onnxruntime-web";
import type { RunGraph } from "src/models/adapter";
import { ForecastError, type Backend } from "src/types";

export interface GraphSession {
  backend: Backend;
  run: RunGraph;
  release(): Promise<void>;
}

export interface SessionOptions {
  /** Backend to try first. */
  device: Backend;
  /** Fall back to WASM if WebGPU session creation fails. */
  fallback: boolean;
  wasmPaths?: string;
  externalData?: { path: string; data: Uint8Array }[];
}

/** Creates an onnxruntime-web session for the graph bytes. */
export async function createSession(bytes: Uint8Array, opts: SessionOptions): Promise<GraphSession> {
  ort.env.wasm.wasmPaths =
    opts.wasmPaths ?? `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;

  const make = (ep: Backend) =>
    ort.InferenceSession.create(bytes, {
      executionProviders: [ep],
      graphOptimizationLevel: "all",
      externalData: opts.externalData,
    });

  let session: ort.InferenceSession | undefined;
  let backend: Backend = opts.device;
  try {
    if (opts.device === "webgpu") {
      try {
        session = await make("webgpu");
      } catch (e) {
        if (!opts.fallback) throw e;
        console.warn("forecast-web: WebGPU session failed, falling back to WASM", e);
        backend = "wasm";
      }
    }
    session ??= await make("wasm");
  } catch (e) {
    throw new ForecastError("LOAD_FAILED", `could not create a ${backend} session`, { cause: e });
  }

  // onnxruntime-web sessions do not support overlapping run() calls; queue them.
  let queue: Promise<unknown> = Promise.resolve();
  const s = session;
  const run: RunGraph = (feeds) => {
    const job = queue.then(async () => {
      const tensors: Record<string, ort.Tensor> = {};
      for (const [name, f] of Object.entries(feeds)) tensors[name] = new ort.Tensor("float32", f.data, f.dims);
      const out = await s.run(tensors);
      const result: Record<string, Float32Array> = {};
      for (const [name, t] of Object.entries(out)) result[name] = (await t.getData()) as Float32Array;
      return result;
    });
    queue = job.catch(() => undefined);
    return job;
  };

  // Let queued runs finish before freeing the session.
  return { backend, run, release: () => queue.then(() => s.release()) };
}
