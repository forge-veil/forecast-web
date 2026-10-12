export {};

vi.mock("src/core/session", () => ({ createSession: vi.fn() }));
const { loadForecaster, MODELS } = await import("src/index");

describe("loadForecaster", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("rejects an unknown model before fetching", async () => {
    const f = vi.fn();
    vi.stubGlobal("fetch", f);
    await expect(loadForecaster({ model: "nope" as never, worker: false })).rejects.toMatchObject({
      code: "UNKNOWN_MODEL",
    });
    expect(f).not.toHaveBeenCalled();
  });

  it("loads from the weights option when given", async () => {
    const f = vi.fn(async (_url: string) => new Response("", { status: 404 }));
    vi.stubGlobal("fetch", f);
    await expect(
      loadForecaster({ weights: "https://w.example/tfm", worker: false, device: "wasm", cache: false }),
    ).rejects.toMatchObject({ code: "LOAD_FAILED" });
    expect(f.mock.calls[0][0]).toBe("https://w.example/tfm/config.json");
  });

  it("defaults to the model's published weights", async () => {
    const f = vi.fn(async (_url: string) => new Response("", { status: 404 }));
    vi.stubGlobal("fetch", f);
    await loadForecaster({ worker: false, device: "wasm", cache: false }).catch(() => {});
    expect(f.mock.calls[0][0]).toBe(
      "https://huggingface.co/bakshinghampalace/timesfm-2.5-200m-onnx/resolve/main/config.json",
    );
  });

  it("exports MODELS", () => {
    expect(Object.keys(MODELS)).toEqual(["timesfm-2.5"]);
  });
});
