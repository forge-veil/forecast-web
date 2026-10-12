import * as ort from "onnxruntime-web";
import type { RunGraph } from "src/forecaster";
import { CONTEXT, TimesFMError, type Backend, type Device } from "src/types";

export interface GraphSession {
  backend: Backend;
  run: RunGraph;
  release(): Promise<void>;
}

async function hasWebGPU(): Promise<boolean> {
  const gpu = (globalThis.navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } })?.gpu;
  if (!gpu) return false;
  try {
    return !!(await gpu.requestAdapter());
  } catch {
    return false;
  }
}

/**
 * Creates an onnxruntime-web session for the graph bytes. "auto" tries WebGPU
 * and falls back to WASM; an explicit device fails loudly instead.
 */
export async function createSession(
  bytes: Uint8Array,
  opts: { device?: Device; wasmPaths?: string } = {},
): Promise<GraphSession> {
  const device = opts.device ?? "auto";
  ort.env.wasm.wasmPaths =
    opts.wasmPaths ?? `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;

  const make = (ep: Backend) =>
    ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: "all" });

  let session: ort.InferenceSession | undefined;
  let backend: Backend = "wasm";
  try {
    if (device === "webgpu" || (device === "auto" && (await hasWebGPU()))) {
      try {
        session = await make("webgpu");
        backend = "webgpu";
      } catch (e) {
        if (device === "webgpu") throw e;
        console.warn("timesfm-web: WebGPU unavailable, falling back to WASM", e);
      }
    }
    session ??= await make("wasm");
  } catch (e) {
    throw new TimesFMError("LOAD_FAILED", `could not create a ${device} session`, { cause: e });
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
