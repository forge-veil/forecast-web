---
title: Tabular prediction
group: Guides
---

# Tabular prediction

`forecast-web/tabular` runs [TabICLv2](https://github.com/soda-inria/tabicl) (Soda team, Inria): classification and regression on a table with no training. The model reads the training rows in context and predicts the test rows in one forward pass.

```ts
import { loadTabular } from "forecast-web/tabular";

const clf = await loadTabular({ task: "classification" });
const fitted = clf.fit(trainRows, trainLabels); // rows of numbers, strings, booleans or null
const proba = await fitted.predictProba(testRows);
const labels = await fitted.predict(testRows);

const reg = await loadTabular({ task: "regression" });
const values = await reg.fit(trainRows, trainTargets).predict(testRows, { output: "quantiles", alphas: [0.1, 0.9] });
```

## From a CSV to predictions

A common case: a spreadsheet export with a mix of text and number columns, where some rows are missing the value you want to predict. Fit on the rows that have it, predict the rest.

```ts
// species,island,bill_length_mm,body_mass_g,sex
// Adelie,Torgersen,39.1,3750,male
// Gentoo,Biscoe,46.1,,female
// Adelie,Dream,37.8,3400,
const [header, ...lines] = (await (await fetch("/data/penguins.csv")).text()).trim().split("\n");
const columns = header.split(",");
const target = columns.indexOf("sex");

// Numbers become numbers, blanks become null, everything else stays text (encoded as a category).
const cell = (s: string) => (s === "" ? null : Number.isFinite(Number(s)) ? Number(s) : s);
const rows = lines.map((line) => line.split(",").map(cell));
const features = (r: (string | number | null)[]) => r.filter((_, i) => i !== target);

const known = rows.filter((r) => r[target] !== null);
const unknown = rows.filter((r) => r[target] === null);

const clf = await loadTabular({ task: "classification" });
const fitted = clf.fit(known.map(features), known.map((r) => r[target] as string));
const guesses = await fitted.predict(unknown.map(features));
```

The feature rows passed to `predict` must have the same columns, in the same order, as the ones passed to `fit`.

### Reading probabilities

`predictProba` returns one row per input row, with probabilities in the order of `fitted.classes`.

```ts
const proba = await fitted.predictProba(unknown.map(features));
proba.forEach((p, i) => {
  const best = p.indexOf(Math.max(...p));
  console.log(`row ${i}: ${fitted.classes[best]} (${Math.round(p[best] * 100)}% sure)`);
});
```

### Ranges for numbers

For regression, ask for quantiles to get a likely range along with the estimate.

```ts
const reg = await loadTabular({ task: "regression" });
const fittedPrice = reg.fit(trainRows, trainPrices);
const [low, mid, high] = (await fittedPrice.predict([newListing], { output: "quantiles", alphas: [0.1, 0.5, 0.9] }))[0];
// "Likely $410k, between $360k and $480k"
```

### Checking accuracy before you rely on it

Hold back some known rows, predict them, and compare. The demo's Tables tab does exactly this.

```ts
const shuffled = [...known].sort(() => Math.random() - 0.5);
const test = shuffled.slice(0, Math.floor(shuffled.length / 5));
const train = shuffled.slice(test.length);
const check = clf.fit(train.map(features), train.map((r) => r[target] as string));
const predicted = await check.predict(test.map(features));
const accuracy = predicted.filter((p, i) => p === test[i][target]).length / test.length;
```

## Details

Columns holding strings or booleans are encoded as categories; missing numbers are imputed with the training mean, as upstream. Predictions match `tabicl` 2.2.0 end to end: probabilities within 2e-5 (measured worst 9.7e-6) with identical labels, and regression outputs within 2e-5 of the target's standard deviation, in fp32. The fp16-weight default (56 to 59 MB) stays within 0.2%.

Limits: at most 10 classes; 5,000 training rows, 100 features and 20,000 cells (rows x features) in the browser. Cost grows with rows x features: about 5 s on a laptop CPU for 1,000 x 10 with 100 test rows and the default 8 ensemble members; WebGPU is faster, and `nEstimators: 1` is about 8 times faster. Errors use the codes `TOO_MANY_CLASSES`, `TOO_LARGE` and `INVALID_INPUT`. With a literal `task`, results are typed precisely (`predict` returns `number[]`, or `number[][]` for quantiles); `tabular.backend` and `tabular.dtype` show where it runs, and `LIMITS` exposes the caps for pre-checking a table before downloading.
