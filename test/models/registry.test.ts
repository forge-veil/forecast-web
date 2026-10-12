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
