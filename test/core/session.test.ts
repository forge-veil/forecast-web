export {};

const { create } = vi.hoisted(() => ({ create: vi.fn() }));

vi.mock("onnxruntime-web", () => ({
  env: { wasm: {}, versions: { web: "1.30.0" } },
  Tensor: class {
    constructor(
      public type: string,
      public data: Float32Array,
      public dims: number[],
    ) {}
  },
  InferenceSession: { create: (...a: unknown[]) => create(...a) },
}));

const { createSession } = await import("src/core/session");

describe("createSession", () => {
  beforeEach(() => create.mockReset());

  it("passes named feeds as tensors and returns every output by name", async () => {
    const run = vi.fn(async (feeds: Record<string, { data: Float32Array; dims: number[] }>) => ({
      a: { getData: async () => feeds.x.data.map((v) => v * 2) },
      b: { getData: async () => new Float32Array([7]) },
    }));
    create.mockResolvedValue({ run, release: vi.fn(async () => {}) });
    const s = await createSession(new Uint8Array(1), { device: "wasm", fallback: false });
    const out = await s.run({ x: { data: new Float32Array([1, 2]), dims: [1, 2] } });
    expect(run.mock.calls[0][0].x.dims).toEqual([1, 2]);
    expect(Array.from(out.a)).toEqual([2, 4]);
    expect(Array.from(out.b)).toEqual([7]);
  });

  it("runs calls one at a time and releases after the queue drains", async () => {
    const order: string[] = [];
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const run = vi.fn(async (feeds: Record<string, { data: Float32Array }>) => {
      const tag = String(feeds.x.data[0]);
      order.push(`start ${tag}`);
      if (tag === "1") await gate;
      order.push(`end ${tag}`);
      return {};
    });
    const sessionRelease = vi.fn(async () => void order.push("release"));
    create.mockResolvedValue({ run, release: sessionRelease });
    const s = await createSession(new Uint8Array(1), { device: "wasm", fallback: false });
    const feed = (v: number) => ({ x: { data: new Float32Array([v]), dims: [1] } });
    const p1 = s.run(feed(1));
    const p2 = s.run(feed(2));
    const done = s.release();
    release();
    await Promise.all([p1, p2, done]);
    expect(order).toEqual(["start 1", "end 1", "start 2", "end 2", "release"]);
  });
});
