import { DEFAULT_MODEL_ID, MODELS, getAdapter } from "src/models/index";

describe("model registry", () => {
  it("lists TimesFM 2.5 as the default", () => {
    expect(DEFAULT_MODEL_ID).toBe("timesfm-2.5");
    expect(MODELS["timesfm-2.5"]).toMatchObject({ name: "TimesFM 2.5", maxContext: 512, maxHorizon: 128 });
    expect(MODELS["timesfm-2.5"].defaults).toEqual({ webgpu: "fp16", wasm: "q8" });
  });

  it("keeps model info read-only so callers cannot change loader behavior", () => {
    const info = MODELS["timesfm-2.5"];
    expect(() => (info.dtypes as string[]).push("q4")).toThrow(TypeError);
    expect(() => ((info.defaults as { wasm: string }).wasm = "fp32")).toThrow(TypeError);
    expect(() => ((info as { maxHorizon: number }).maxHorizon = 999)).toThrow(TypeError);
    expect(() => ((info.sizes as { q8?: number }).q8 = 1)).toThrow(TypeError);
    expect(getAdapter("timesfm-2.5").info).toBe(info);
    expect(Object.isFrozen(MODELS)).toBe(true);
  });

  it("lists TimesFM and the four Chronos-Bolt sizes, all frozen", () => {
    expect(Object.keys(MODELS)).toEqual([
      "timesfm-2.5",
      "chronos-bolt-tiny",
      "chronos-bolt-mini",
      "chronos-bolt-small",
      "chronos-bolt-base",
    ]);
    for (const info of Object.values(MODELS)) expect(Object.isFrozen(info.dtypes)).toBe(true);
    expect(MODELS["chronos-bolt-small"]).toMatchObject({ name: "Chronos-Bolt Small", maxContext: 2048, maxHorizon: 512, horizon: 64 });
    expect(MODELS["timesfm-2.5"].horizon).toBe(128);
  });

  it("Chronos-Bolt defaults to fp16 on both backends; q8 drifts on multi-pass horizons", () => {
    for (const s of ["tiny", "mini", "small", "base"] as const) {
      expect(MODELS[`chronos-bolt-${s}`].defaults).toEqual({ webgpu: "fp16", wasm: "fp16" });
      expect(MODELS[`chronos-bolt-${s}`].dtypes).toContain("q8");
    }
  });

  it("Chronos-Bolt checkConfig rejects a mismatched graph", () => {
    const a = getAdapter("chronos-bolt-tiny");
    const ok = { model_type: "chronos-bolt", context: 2048, prediction_length: 64, patch: 16, quantiles: Array(9).fill(0) };
    expect(() => a.checkConfig(ok)).not.toThrow();
    expect(() => a.checkConfig({ ...ok, context: 512 })).toThrow(/unsupported Chronos-Bolt config/);
  });

  it("returns the adapter for a known id", () => {
    expect(getAdapter("timesfm-2.5").weights).toBe("bakshinghampalace/timesfm-2.5-200m-onnx");
  });

  it.each(["nope", "toString", "constructor", ""])("rejects %j as UNKNOWN_MODEL", (id) => {
    expect(() => getAdapter(id)).toThrow(expect.objectContaining({ code: "UNKNOWN_MODEL" }));
  });

  it.each([5, null, undefined, {}])("rejects non-string id %j as UNKNOWN_MODEL", (id) => {
    expect(() => getAdapter(id as never)).toThrow(expect.objectContaining({ code: "UNKNOWN_MODEL" }));
  });

  it("points 0.1 users who pass a repo id at the weights option", () => {
    expect(() => getAdapter("bakshinghampalace/timesfm-2.5-200m-onnx")).toThrow(/weights/);
  });
});
