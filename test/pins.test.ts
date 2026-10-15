import { pinOf, resolvePin } from "src/pins";
import { getAdapter, MODELS } from "src/models/index";
import { STANDARD_FILES } from "src/models/files";
import { getTabularSpec, TABULAR_MODELS } from "src/tabular/models";
import type { ModelId } from "src/types";

const TFM = "bakshinghampalace/timesfm-2.5-200m-onnx";

describe("resolvePin", () => {
  it("uses the pinned commit and built-in hashes for the published weights", () => {
    const pin = pinOf(TFM);
    expect(resolvePin(TFM)).toEqual({ revision: pin.revision, hashes: pin.files });
    expect(resolvePin(TFM, TFM)).toEqual({ revision: pin.revision, hashes: pin.files });
    expect(resolvePin(TFM, undefined, pin.revision)).toEqual({ revision: pin.revision, hashes: pin.files });
  });

  it("falls back to config.json hashes for another revision or custom weights", () => {
    expect(resolvePin(TFM, undefined, "main")).toEqual({ revision: "main" });
    expect(resolvePin(TFM, "me/mine")).toEqual({ revision: "main" });
    expect(resolvePin(TFM, "https://cdn.example/w/", "v2")).toEqual({ revision: "v2" });
  });
});

describe("pins.json", () => {
  const sizeOf = (repo: string, dtype: keyof typeof STANDARD_FILES) => {
    const spec = STANDARD_FILES[dtype];
    const files = pinOf(repo).files;
    return [spec.graph, spec.half?.file].filter((f): f is string => !!f).reduce((n, f) => n + files[f].bytes, 0);
  };

  it.each(Object.keys(MODELS) as ModelId[])("covers %s, with registry sizes matching", (id) => {
    const { weights, info } = getAdapter(id);
    for (const dtype of info.dtypes) expect(info.sizes[dtype]).toBe(sizeOf(weights, dtype));
  });

  it.each(Object.keys(TABULAR_MODELS) as (keyof typeof TABULAR_MODELS)[])("covers %s, with registry sizes matching", (id) => {
    const spec = getTabularSpec(id);
    for (const dtype of spec.info.dtypes) expect(spec.info.sizes[dtype]).toBe(sizeOf(spec.weights, dtype));
  });
});
