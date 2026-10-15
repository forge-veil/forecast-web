export {};
vi.mock("src/core/session", () => ({ createSession: vi.fn() }));
const { loadTabular } = await import("src/tabular/index");

describe("loadTabular", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("downloads each task's weights at a pinned revision", async () => {
    const f = vi.fn(async (_url: string) => new Response("", { status: 404 }));
    vi.stubGlobal("fetch", f);
    await loadTabular({ task: "classification", worker: false, device: "wasm", cache: false }).catch(() => {});
    await loadTabular({ task: "regression", worker: false, device: "wasm", cache: false }).catch(() => {});
    expect(f.mock.calls[0][0]).toBe("https://huggingface.co/bakshinghampalace/tabicl-classifier-v2-onnx/resolve/098fb7be8df8a853a9d7f54a344ee9c6734a53d1/config.json");
    expect(f.mock.calls[1][0]).toBe("https://huggingface.co/bakshinghampalace/tabicl-regressor-v2-onnx/resolve/f7d19314f03770294c4fa3e019b28335dfa86986/config.json");
  });

  it("keeps the pinned revision when weights names the published repo", async () => {
    const f = vi.fn(async (_url: string) => new Response("", { status: 404 }));
    vi.stubGlobal("fetch", f);
    await loadTabular({ task: "classification", weights: "bakshinghampalace/tabicl-classifier-v2-onnx", worker: false, device: "wasm", cache: false }).catch(() => {});
    expect(f.mock.calls[0][0]).toContain("/resolve/098fb7be8df8a853a9d7f54a344ee9c6734a53d1/");
  });
});
