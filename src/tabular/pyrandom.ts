/**
 * CPython's random.Random (MT19937) for non-negative integer seeds:
 * init_by_array seeding, random(), getrandbits(k <= 32), _randbelow,
 * shuffle and choice, bit-exact with CPython 3.12. TabICL builds its
 * ensemble from random.Random(42), so matching it reproduces upstream exactly.
 */
export class PyRandom {
  private mt = new Uint32Array(624);
  private i = 625;

  constructor(seed: number) {
    // CPython seeds with init_by_array over the 32-bit words of |seed|, least significant first.
    let s = Math.abs(Math.trunc(seed));
    const key: number[] = [];
    do {
      key.push(s % 2 ** 32);
      s = Math.floor(s / 2 ** 32);
    } while (s > 0);
    this.initByArray(key);
  }

  private initGenrand(s: number) {
    const mt = this.mt;
    mt[0] = s >>> 0;
    for (let i = 1; i < 624; i++) mt[i] = (Math.imul(1812433253, mt[i - 1] ^ (mt[i - 1] >>> 30)) + i) >>> 0;
    this.i = 624;
  }

  private initByArray(key: number[]) {
    const mt = this.mt;
    this.initGenrand(19650218);
    let i = 1;
    let j = 0;
    for (let k = Math.max(624, key.length); k > 0; k--) {
      mt[i] = ((mt[i] ^ Math.imul(mt[i - 1] ^ (mt[i - 1] >>> 30), 1664525)) + key[j] + j) >>> 0;
      i++;
      j++;
      if (i >= 624) {
        mt[0] = mt[623];
        i = 1;
      }
      if (j >= key.length) j = 0;
    }
    for (let k = 623; k > 0; k--) {
      mt[i] = ((mt[i] ^ Math.imul(mt[i - 1] ^ (mt[i - 1] >>> 30), 1566083941)) - i) >>> 0;
      i++;
      if (i >= 624) {
        mt[0] = mt[623];
        i = 1;
      }
    }
    mt[0] = 0x80000000;
  }

  private next(): number {
    const mt = this.mt;
    if (this.i >= 624) {
      for (let k = 0; k < 624; k++) {
        const y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff);
        mt[k] = mt[(k + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      this.i = 0;
    }
    let y = mt[this.i++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** A float in [0, 1) with 53 random bits. */
  random(): number {
    const a = this.next() >>> 5;
    const b = this.next() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }

  getrandbits(k: number): number {
    if (k <= 0 || k > 32) throw new RangeError("getrandbits supports 1 to 32 bits");
    return this.next() >>> (32 - k);
  }

  /** random._randbelow_with_getrandbits: rejection sampling with n.bit_length() bits. */
  randbelow(n: number): number {
    if (n <= 0) return 0;
    const k = 32 - Math.clz32(n);
    let r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  }

  shuffle<T>(a: T[]): void {
    for (let i = a.length - 1; i > 0; i--) {
      const j = this.randbelow(i + 1);
      [a[i], a[j]] = [a[j], a[i]];
    }
  }

  choice<T>(a: T[]): T {
    return a[this.randbelow(a.length)];
  }
}
