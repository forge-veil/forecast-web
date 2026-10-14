import { PyRandom } from "src/tabular/pyrandom";

export type Norm = "none" | "power";

export interface MemberConfig {
  norm: Norm;
  /** Column order for this member. */
  feat: number[];
  /** Class remapping (circular shift) for this member; null for regression. */
  cls: number[] | null;
}

/**
 * Upstream groups members by norm method through `list(set(...))`, whose order
 * depends on the string hash seed. The reference fixtures run with
 * PYTHONHASHSEED=0, where "power" comes before "none"; we use that order.
 */
const GROUP_ORDER: Norm[] = ["power", "none"];

/** Shuffler._latin_squares: recursive construction, then shuffle rows, transpose, shuffle. */
export function latinSquares(n: number, rng: PyRandom): number[][] {
  const rls = (symbols: number[]): number[][] => {
    if (symbols.length === 1) return [[symbols[0]]];
    const sym = rng.choice(symbols);
    symbols.splice(symbols.indexOf(sym), 1);
    const square = rls(symbols);
    square.push([...square[0]]);
    for (let i = 0; i < square.length; i++) square[i].splice(i, 0, sym);
    return square;
  };
  const square = rls(Array.from({ length: n }, (_, i) => i));
  rng.shuffle(square);
  const trans = square[0].map((_, c) => square.map((row) => row[c]));
  rng.shuffle(trans);
  return trans;
}

/** Shuffler(n, "latin").shuffle(nEstimators). */
export function featureShuffles(n: number, nEstimators: number, seed: number): number[][] {
  if (nEstimators === 1) return [Array.from({ length: n }, (_, i) => i)];
  if (n > 4000) throw new RangeError("more than 4000 features is not supported");
  return latinSquares(n, new PyRandom(seed));
}

/** Shuffler(c, "shift").shuffle(nEstimators): every circular shift of the class indices. */
export function classShuffles(c: number, nEstimators: number): number[][] {
  const idx = Array.from({ length: c }, (_, i) => i);
  if (nEstimators === 1) return [idx];
  return idx.map((i) => (i === 0 ? [...idx] : [...idx.slice(-i), ...idx.slice(0, -i)]));
}

/** EnsembleGenerator._generate_ensemble, flattened in upstream's member order. */
export function ensembleConfigs(opts: {
  nFeatures: number;
  nClasses?: number;
  nEstimators: number;
  seed: number;
  normMethods: Norm[];
}): MemberConfig[] {
  const feats = featureShuffles(opts.nFeatures, opts.nEstimators, opts.seed);
  const classes = opts.nClasses !== undefined ? classShuffles(opts.nClasses, opts.nEstimators) : [null];
  const pairs: [number[], number[] | null][] = [];
  for (const f of feats) for (const c of classes) pairs.push([f, c]);
  new PyRandom(opts.seed).shuffle(pairs);
  const members: MemberConfig[] = [];
  for (const [feat, cls] of pairs) for (const norm of opts.normMethods) members.push({ norm, feat, cls });
  const taken = members.slice(0, opts.nEstimators);
  return GROUP_ORDER.flatMap((norm) => taken.filter((m) => m.norm === norm));
}
