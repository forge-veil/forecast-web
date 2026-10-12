/** Decodes one IEEE 754 half-precision value from its 16-bit pattern. */
export function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

let table: Float32Array | null = null;

/**
 * Expands little-endian fp16 bytes into fp32 bytes (twice the length). Used to
 * turn the compact fp16 weight file into the fp32 external data the graph expects.
 */
export function expandHalfToFloat(bytes: Uint8Array): Uint8Array {
  if (bytes.byteLength % 2) throw new Error("fp16 data must have an even byte length");
  const n = bytes.byteLength / 2;
  const out = new Float32Array(n);
  // Copy if unaligned so the Uint16Array view is legal.
  const src = bytes.byteOffset % 2 ? bytes.slice() : bytes;
  const halves = new Uint16Array(src.buffer, src.byteOffset, n);

  const F16 = (globalThis as { Float16Array?: new (b: ArrayBufferLike, o: number, l: number) => ArrayLike<number> })
    .Float16Array;
  if (F16) {
    out.set(new F16(src.buffer, src.byteOffset, n));
  } else {
    if (!table) {
      table = new Float32Array(65536);
      for (let i = 0; i < 65536; i++) table[i] = halfToFloat(i);
    }
    for (let i = 0; i < n; i++) out[i] = table[halves[i]];
  }
  return new Uint8Array(out.buffer);
}
