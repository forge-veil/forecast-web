/** One ensemble member's raw logits: rows of `stride` values (10 from the graph), first C used. */
export interface MemberLogits {
  logits: Float32Array;
  stride: number;
  /** The member's class shift: position k holds the class index the member used for class k. */
  cls: number[];
}

/**
 * TabICLClassifier.predict_proba with average_logits=True: undo each member's
 * class shift (`out[..., shuffle]`), average logits, softmax at the given
 * temperature, renormalize.
 */
export function combineClassification(members: MemberLogits[], nTest: number, nClasses: number, temperature = 0.9): number[][] {
  const out: number[][] = [];
  for (let t = 0; t < nTest; t++) {
    const avg = new Float64Array(nClasses);
    for (const m of members) for (let k = 0; k < nClasses; k++) avg[k] += m.logits[t * m.stride + m.cls[k]];
    let max = -Infinity;
    for (let k = 0; k < nClasses; k++) {
      avg[k] /= members.length;
      max = Math.max(max, avg[k] / temperature);
    }
    const e = Array.from(avg, (v) => Math.exp(v / temperature - max));
    const sum = e.reduce((a, b) => a + b, 0);
    const p = e.map((v) => v / sum);
    const total = p.reduce((a, b) => a + b, 0);
    out.push(p.map((v) => v / total));
  }
  return out;
}
