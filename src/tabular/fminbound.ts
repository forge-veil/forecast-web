/**
 * scipy.optimize.fminbound (scipy 1.18 _minimize_scalar_bounded): bounded
 * Brent minimization, ported step for step so the optimum matches scipy.
 */
export function fminbound(func: (x: number) => number, x1: number, x2: number, xatol = 1e-5, maxfun = 500): number {
  if (!(Number.isFinite(x1) && Number.isFinite(x2))) throw new RangeError("Optimization bounds must be finite scalars.");
  if (x1 > x2) throw new RangeError("The lower bound exceeds the upper bound.");
  const sign = (v: number) => (v > 0 ? 1 : v < 0 ? -1 : 0);
  const sqrtEps = Math.sqrt(2.2e-16);
  const goldenMean = 0.5 * (3.0 - Math.sqrt(5.0));
  let a = x1;
  let b = x2;
  let fulc = a + goldenMean * (b - a);
  let nfc = fulc;
  let xf = fulc;
  let rat = 0;
  let e = 0;
  let x = xf;
  let fx = func(x);
  let num = 1;
  let fu = Infinity;
  let ffulc = fx;
  let fnfc = fx;
  let xm = 0.5 * (a + b);
  let tol1 = sqrtEps * Math.abs(xf) + xatol / 3.0;
  let tol2 = 2.0 * tol1;

  while (Math.abs(xf - xm) > tol2 - 0.5 * (b - a)) {
    let golden = true;
    if (Math.abs(e) > tol1) {
      golden = false;
      let r = (xf - nfc) * (fx - ffulc);
      let q = (xf - fulc) * (fx - fnfc);
      let p = (xf - fulc) * q - (xf - nfc) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) p = -p;
      q = Math.abs(q);
      r = e;
      e = rat;
      if (Math.abs(p) < Math.abs(0.5 * q * r) && p > q * (a - xf) && p < q * (b - xf)) {
        rat = (p + 0.0) / q;
        x = xf + rat;
        if (x - a < tol2 || b - x < tol2) {
          const si = sign(xm - xf) + (xm - xf === 0 ? 1 : 0);
          rat = tol1 * si;
        }
      } else {
        golden = true;
      }
    }
    if (golden) {
      e = xf >= xm ? a - xf : b - xf;
      rat = goldenMean * e;
    }
    const si = sign(rat) + (rat === 0 ? 1 : 0);
    x = xf + si * Math.max(Math.abs(rat), tol1);
    fu = func(x);
    num += 1;
    if (fu <= fx) {
      if (x >= xf) a = xf;
      else b = xf;
      fulc = nfc;
      ffulc = fnfc;
      nfc = xf;
      fnfc = fx;
      xf = x;
      fx = fu;
    } else {
      if (x < xf) a = x;
      else b = x;
      if (fu <= fnfc || nfc === xf) {
        fulc = nfc;
        ffulc = fnfc;
        nfc = x;
        fnfc = fu;
      } else if (fu <= ffulc || fulc === xf || fulc === nfc) {
        fulc = x;
        ffulc = fu;
      }
    }
    xm = 0.5 * (a + b);
    tol1 = sqrtEps * Math.abs(xf) + xatol / 3.0;
    tol2 = 2.0 * tol1;
    if (num >= maxfun) break;
  }
  return xf;
}
