/** torch.quantile with linear interpolation, on values already sorted ascending. */
export function quantileLinear(sorted: ArrayLike<number>, q: number): number {
  const pos = q * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
