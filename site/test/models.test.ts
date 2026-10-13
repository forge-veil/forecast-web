import { MODELS } from "forecast-web";
import { compareVerdict, downloadBytes, forecastKey, modelColor, rankByError, readChoice, resolvePrimary } from "src/lib/models";

describe("readChoice", () => {
  it("accepts a known model id and falls back to TimesFM otherwise", () => {
    expect(readChoice("chronos-bolt-small")).toBe("chronos-bolt-small");
    expect(readChoice("nope")).toBe("timesfm-2.5");
    expect(readChoice(null)).toBe("timesfm-2.5");
    expect(readChoice("toString")).toBe("timesfm-2.5");
  });
});

describe("downloadBytes", () => {
  it("uses the dtype auto would pick on each backend", () => {
    const info = MODELS["timesfm-2.5"];
    expect(downloadBytes(info, true)).toBe(info.sizes.fp16);
    expect(downloadBytes(info, false)).toBe(info.sizes[info.defaults.wasm]);
    expect(downloadBytes(MODELS["chronos-bolt-tiny"], false)).toBe(MODELS["chronos-bolt-tiny"].sizes.fp16);
  });
});

describe("forecastKey", () => {
  it("differs by model so a switch never reuses another model's forecast", () => {
    expect(forecastKey("timesfm-2.5", [1, 2, 3])).not.toBe(forecastKey("chronos-bolt-tiny", [1, 2, 3]));
    expect(forecastKey("timesfm-2.5", [1, 2, 3])).toBe(forecastKey("timesfm-2.5", [1, 2, 3]));
  });
});

describe("compareVerdict", () => {
  it("names the model with less error", () => {
    expect(compareVerdict({ name: "TimesFM 2.5", mae: 8 }, { name: "Chronos-Bolt Small", mae: 10 })).toBe(
      "TimesFM 2.5: 20% less error than Chronos-Bolt Small",
    );
    expect(compareVerdict({ name: "A", mae: 10 }, { name: "B", mae: 5 })).toBe("B: 50% less error than A");
    expect(compareVerdict({ name: "A", mae: 10 }, { name: "B", mae: 10.1 })).toBe("A and B: same error");
  });
});

describe("rankByError", () => {
  it("puts scored models first, best first, then the rest in registry order", () => {
    const ranked = rankByError([
      { id: "timesfm-2.5", mae: 5 },
      { id: "chronos-bolt-tiny" },
      { id: "chronos-bolt-mini", mae: 3 },
      { id: "chronos-bolt-small" },
    ]);
    expect(ranked.map((r) => r.id)).toEqual(["chronos-bolt-mini", "timesfm-2.5", "chronos-bolt-tiny", "chronos-bolt-small"]);
  });
});

describe("modelColor", () => {
  it("gives each model its own fixed color token", () => {
    expect(modelColor("timesfm-2.5")).toBe("var(--m1)");
    expect(modelColor("chronos-bolt-base")).toBe("var(--m5)");
  });
});

describe("resolvePrimary", () => {
  it("keeps the selected model when it is loaded or nothing is", () => {
    expect(resolvePrimary("chronos-bolt-mini", ["timesfm-2.5", "chronos-bolt-mini"])).toBe("chronos-bolt-mini");
    expect(resolvePrimary("chronos-bolt-mini", [])).toBe("chronos-bolt-mini");
  });

  it("falls back to the first loaded model while the selected one is not ready", () => {
    expect(resolvePrimary("chronos-bolt-base", ["timesfm-2.5", "chronos-bolt-tiny"])).toBe("timesfm-2.5");
  });
});
