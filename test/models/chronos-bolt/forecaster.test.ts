import { BLOCK, CONTEXT, NQ } from "src/models/chronos-bolt/constants";
import { createForecaster, type ChronosRun } from "src/models/chronos-bolt/forecaster";

/**
 * Fake graph: quantile q at step t = normalized last observed value + (q - 4) + t / 100.
 * Lets tests check de-normalization, the rollout and the 81 -> 9 reduction by hand.
 */
function fakeRun() {
  const batches: number[] = [];
  const run: ChronosRun = async (context, mask, batch) => {
    batches.push(batch);
    const out = new Float32Array(batch * NQ * BLOCK);
    for (let b = 0; b < batch; b++) {
      let last = 0;
      for (let i = CONTEXT - 1; i >= 0; i--)
        if (mask[b * CONTEXT + i]) {
          last = context[b * CONTEXT + i];
          break;
        }
      for (let q = 0; q < NQ; q++) for (let t = 0; t < BLOCK; t++) out[(b * NQ + q) * BLOCK + t] = last + (q - 4) + t / 100;
    }
    return out;
  };
  return { run, batches };
}

describe("Chronos-Bolt forecaster", () => {
  it("defaults to one 64-step pass and returns the median and nine quantiles", async () => {
    const { run, batches } = fakeRun();
    const [f] = await createForecaster(run)([[1, 2, 3]]);
    expect(batches).toEqual([1]);
    expect(f.horizon).toBe(BLOCK);
    expect(f.median).toHaveLength(BLOCK);
    expect(f.mean).toBeUndefined();
    expect(f.median).toEqual(f.q50);
    // loc 2, scale sqrt(2/3); the last normalized value maps back to 3 at q50, t = 0
    expect(f.q50[0]).toBeCloseTo(3, 5);
    expect(f.q10[0]).toBeCloseTo(3 - 4 * Math.sqrt(2 / 3), 5);
  });

  it.each([
    [1, [1]],
    [64, [1]],
    [65, [1, 9]],
    [200, [1, 9, 9, 9]],
  ])("horizon %i runs passes with batches %j", async (h, expected) => {
    const { run, batches } = fakeRun();
    const [f] = await createForecaster(run)([[1, 2, 3, 4]], { horizon: h });
    expect(batches).toEqual(expected);
    expect(f.median).toHaveLength(h);
  });

  it("batches every series and path together", async () => {
    const { run, batches } = fakeRun();
    const out = await createForecaster(run)([[1, 2], [5, 6, 7]], { horizon: 100 });
    expect(batches).toEqual([2, 18]);
    expect(out).toHaveLength(2);
  });

  it("reduces 81 rollout values per step to quantiles of their empirical distribution", async () => {
    const { run } = fakeRun();
    const [f] = await createForecaster(run)([[0, 0, 0, 10]], { horizon: 65 });
    expect(f.q10[64]).toBeLessThan(f.q50[64]);
    expect(f.q50[64]).toBeLessThan(f.q90[64]);
    for (let t = 0; t < 65; t++) {
      expect(f.q10[t]).toBeLessThanOrEqual(f.q20[t]);
      expect(f.q80[t]).toBeLessThanOrEqual(f.q90[t]);
    }
  });

  it("keeps an all-missing series finite", async () => {
    const { run } = fakeRun();
    const [f] = await createForecaster(run)([[null, NaN]]);
    expect(f.median.every(Number.isFinite)).toBe(true);
  });

  it.each([[0], [513], [1.5]])("rejects horizon %s", async (h) => {
    const { run } = fakeRun();
    await expect(createForecaster(run)([[1, 2]], { horizon: h })).rejects.toMatchObject({
      code: h === 513 ? "HORIZON_TOO_LONG" : "INVALID_HORIZON",
    });
  });

  it("rejects an empty series list", async () => {
    const { run } = fakeRun();
    await expect(createForecaster(run)([])).rejects.toMatchObject({ code: "EMPTY_INPUT" });
  });
});
