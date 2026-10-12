import { expandHalfToFloat, halfToFloat } from "src/core/half";

describe("halfToFloat", () => {
  it.each([
    [0x0000, 0],
    [0x3c00, 1],
    [0xc000, -2],
    [0x3555, 0.333251953125],
    [0x7bff, 65504],
    [0x0001, 2 ** -24], // smallest subnormal
    [0x03ff, 1023 * 2 ** -24], // largest subnormal
    [0x7c00, Infinity],
    [0xfc00, -Infinity],
  ])("decodes 0x%s", (bits, expected) => {
    expect(halfToFloat(bits)).toBe(expected);
  });

  it("decodes negative zero and NaN", () => {
    expect(Object.is(halfToFloat(0x8000), -0)).toBe(true);
    expect(halfToFloat(0x7e00)).toBeNaN();
  });
});

describe("expandHalfToFloat", () => {
  it("doubles the byte length and decodes little-endian halves in order", () => {
    const half = new Uint8Array([0x00, 0x3c, 0x00, 0xc0, 0x55, 0x35]);
    const out = expandHalfToFloat(half);
    expect(out.byteLength).toBe(12);
    expect(Array.from(new Float32Array(out.buffer, out.byteOffset, 3))).toEqual([1, -2, 0.333251953125]);
  });

  it("handles an unaligned input view", () => {
    const backing = new Uint8Array(5);
    backing.set([0x00, 0x3c, 0x00, 0x40], 1);
    const out = expandHalfToFloat(backing.subarray(1));
    expect(Array.from(new Float32Array(out.buffer, out.byteOffset, 2))).toEqual([1, 2]);
  });

  it("gives identical results through the lookup-table fallback (no Float16Array)", () => {
    const all = new Uint16Array(65536).map((_, i) => i);
    const bytes = new Uint8Array(all.buffer);
    const native = new Float32Array(expandHalfToFloat(bytes).buffer);
    vi.stubGlobal("Float16Array", undefined);
    try {
      const fallback = new Float32Array(expandHalfToFloat(bytes).buffer);
      for (let i = 0; i < 65536; i++) expect(Object.is(fallback[i], native[i]) || (isNaN(fallback[i]) && isNaN(native[i]))).toBe(true);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("rejects an odd byte length", () => {
    expect(() => expandHalfToFloat(new Uint8Array(3))).toThrow(/even/);
  });
});
