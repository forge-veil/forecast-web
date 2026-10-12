import * as ort from "onnxruntime-web";
import type { RunGraph } from "src/forecaster";
import { CONTEXT, TimesFMError, type Backend } from "src/types";

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
        console.warn("timesfm-web: WebGPU session failed, falling back to WASM", e);
        backend = "wasm";
      }
    }
    session ??= await make("wasm");
  } catch (e) {
    throw new TimesFMError("LOAD_FAILED", `could not create a ${backend} session`, { cause: e });
  }

  // onnxruntime-web sessions do not support overlapping run() calls; queue them.
  let queue: Promise<unknown> = Promise.resolve();
  const s = session;
  const run: RunGraph = (inputs, masks, batch) => {
    const job = queue.then(async () => {
      const out = await s.run({
        inputs: new ort.Tensor("float32", inputs, [batch, CONTEXT]),
        masks: new ort.Tensor("float32", masks, [batch, CONTEXT]),
      });
      return {
        point: (await out.point.getData()) as Float32Array,
        qspread: (await out.qspread.getData()) as Float32Array,
      };
    });
    queue = job.catch(() => undefined);
    return job;
  };

  return { backend, run, release: () => s.release() };
}
