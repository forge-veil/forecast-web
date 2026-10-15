import { MODELS } from "forecast-web";
import { compareVerdict, downloadBytes, forecastKey, isMobile, menuOrder, mobileNote, modelColor, rankByError, readChoice, resolvePrimary } from "src/lib/models";

describe("readChoice", () => {
  it("accepts a known model id and falls back to TimesFM otherwise", () => {
    expect(readChoice("chronos-bolt-small")).toBe("chronos-bolt-small");
    expect(readChoice("nope")).toBe("timesfm-2.5");
    expect(readChoice(null)).toBe("timesfm-2.5");
    expect(readChoice("toString")).toBe("timesfm-2.5");
  });

  it("defaults to Chronos-Bolt Tiny on mobile, but keeps a stored choice", () => {
    expect(readChoice(null, true)).toBe("chronos-bolt-tiny");
    expect(readChoice("nope", true)).toBe("chronos-bolt-tiny");
    expect(readChoice("timesfm-2.5", true)).toBe("timesfm-2.5");
  });
});

describe("isMobile", () => {
  const nav = (o: Record<string, unknown>) => o as unknown as Navigator;
  it("counts a device as mobile if either the hint or the user agent says so", () => {
    expect(isMobile(nav({ userAgentData: { mobile: true }, userAgent: "" }))).toBe(true);
    expect(isMobile(nav({ userAgentData: { mobile: false }, userAgent: "Mozilla/5.0 (Linux; Android 15)" }))).toBe(true);
    expect(isMobile(nav({ userAgentData: { mobile: false }, userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" }))).toBe(false);
  });
  it("falls back to the user agent, counting iPads that report a Mac", () => {
    expect(isMobile(nav({ userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)" }))).toBe(true);
    expect(isMobile(nav({ userAgent: "Mozilla/5.0 (Linux; Android 15; Pixel 9) Mobile" }))).toBe(true);
    expect(isMobile(nav({ userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", maxTouchPoints: 5 }))).toBe(true);
    expect(isMobile(nav({ userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", maxTouchPoints: 0 }))).toBe(false);
    expect(isMobile(undefined)).toBe(false);
  });
});

describe("mobile labels", () => {
  it("recommends Tiny and warns about larger models, only on mobile", () => {
    expect(mobileNote("chronos-bolt-tiny", true)).toEqual({ text: "Recommended on phones", warn: false });
    expect(mobileNote("chronos-bolt-small", true)).toEqual({ text: "May crash on phones", warn: true });
    expect(mobileNote("timesfm-2.5", true)).toEqual({ text: "May crash on phones", warn: true });
    expect(mobileNote("timesfm-2.5", false)).toBeNull();
  });
  it("lists Tiny first on mobile and keeps registry order otherwise", () => {
    const ids = ["timesfm-2.5", "chronos-bolt-tiny", "chronos-bolt-mini"] as const;
    expect(menuOrder([...ids], true)).toEqual(["chronos-bolt-tiny", "timesfm-2.5", "chronos-bolt-mini"]);
    expect(menuOrder([...ids], false)).toEqual([...ids]);
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
