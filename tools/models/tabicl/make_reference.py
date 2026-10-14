"""Generate TabICL parity fixtures from the official tabicl package.

Writes, for a model (tabicl-classifier-v2 or tabicl-regressor-v2):
- test/fixtures/<model>/reference.json: a suite of tables with the official
  predict_proba / predict outputs (fit on pandas DataFrames, so categorical
  handling follows the DataFrame path), plus raw per-member network outputs
  for one table.
- test/fixtures/tabicl/*.json: intermediate values for unit tests (random
  streams, Latin squares, ensemble configs, preprocessing; quantile
  post-processing from the regressor run).

Runs on CPU, fp32, float64 input, one thread, PYTHONHASHSEED=0 (set by
re-executing itself), so the ensemble grouping order is reproducible.

Usage: uv run python models/tabicl/make_reference.py <model>
"""

import json
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
  os.environ["PYTHONHASHSEED"] = "0"
  os.execv(sys.executable, [sys.executable] + sys.argv)

import random

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..")
FIX = os.path.join(TOOLS, "..", "test", "fixtures")


def load_meta(name):
  path = os.path.join(TOOLS, "models", name, "model.json")
  if not os.path.isfile(path):
    sys.exit(f"unknown model {name!r}: no tools/models/{name}/model.json")
  with open(path) as f:
    meta = json.load(f)
  if meta.get("scripts") != "tabicl":
    sys.exit(f"{name} is not a TabICL model")
  return meta


def checkpoint(meta):
  return hf_hub_download(meta["source"], meta["checkpoint"], revision=meta["revision"])


def clean(v):
  """JSON-safe: NaN -> None, numpy scalars -> Python."""
  if isinstance(v, dict):
    return {k: clean(x) for k, x in v.items()}
  if isinstance(v, (list, tuple)):
    return [clean(x) for x in v]
  if isinstance(v, np.ndarray):
    return clean(v.tolist())
  if isinstance(v, (np.floating, float)):
    return None if np.isnan(v) else float(v)
  if isinstance(v, (np.integer,)):
    return int(v)
  if isinstance(v, (np.bool_,)):
    return bool(v)
  return v


def dump(path, obj):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    json.dump(clean(obj), f)
  print("wrote", os.path.abspath(path))


# ---------- shared unit fixtures ----------


def rng_fixture():
  out = {"seed": 42}
  r = random.Random(42)
  out["random"] = [r.random() for _ in range(20)]
  r = random.Random(42)
  out["getrandbits32"] = [r.getrandbits(32) for _ in range(20)]
  out["randbelow"] = []
  for n in (1, 2, 3, 7, 10, 100, 1000, 65537):
    r = random.Random(42)
    out["randbelow"].append({"n": n, "values": [r._randbelow(n) for _ in range(30)]})
  out["shuffle"] = []
  for n in (2, 5, 12, 40):
    r = random.Random(42)
    a = list(range(n))
    r.shuffle(a)
    out["shuffle"].append({"input": list(range(n)), "output": a})
  r = random.Random(42)
  inp = [3, 9, 27, 81, 243]
  out["choice"] = {"input": inp, "outputs": [r.choice(inp) for _ in range(20)]}
  return out


def latin_fixture():
  from tabicl._sklearn.preprocessing import Shuffler

  cases = []
  for n in (1, 2, 3, 5, 8, 17):
    cases.append({"n": n, "squares": [list(map(int, p)) for p in Shuffler(n, "latin", random_state=42).shuffle(8)]})
  return {"cases": cases}


def ensemble_fixture():
  from tabicl._sklearn.preprocessing import EnsembleGenerator

  cases = []
  combos = [
    (4, 3, 8, True),
    (1, 2, 8, True),
    (6, 10, 8, True),
    (3, 2, 1, True),
    (5, None, 8, False),
    (2, None, 8, False),
    (20, 5, 8, True),
  ]
  rng = np.random.default_rng(0)
  for f, c, k, cls in combos:
    n = 30
    X = rng.normal(size=(n, f))
    y = (np.arange(n) % c) if cls else rng.normal(size=n)
    g = EnsembleGenerator(classification=cls, n_estimators=k, random_state=42)
    g.fit(X, y)
    configs = []
    for norm, shuffle_configs in g.ensemble_configs_.items():
      for feat, cl in shuffle_configs:
        configs.append({"norm": norm, "feat": list(map(int, feat)), "cls": None if cl is None else list(map(int, cl))})
    cases.append({"n_features": f, "n_classes": c, "n_estimators": k, "classification": cls, "configs": configs})
  return {"cases": cases}


def preprocess_fixture():
  from tabicl._sklearn.preprocessing import PreprocessingPipeline, UniqueFeatureFilter

  rng = np.random.default_rng(3)
  tables = {
    "gaussian": rng.normal(size=(60, 4)),
    "skewed": np.exp(rng.normal(size=(80, 3))),
    "huge_values": rng.normal(size=(50, 2)) * 1e6 + 3e6,
    "all_negative": -np.exp(rng.normal(size=(50, 2))),
    "with_outliers": np.concatenate([rng.normal(size=(59, 2)), [[40.0, -35.0]]]),
    "constant_and_two_rows": np.array([[1.0, 5.0, 2.0], [1.0, 6.0, 2.0]]),
    "all_constant": np.ones((10, 3)),
  }
  out = {"tables": []}
  for name, X in tables.items():
    n = X.shape[0]
    n_train = max(2, int(n * 0.8))
    Xtr, Xte = X[:n_train], X[n_train:] if n > n_train else X[:1]
    uf = UniqueFeatureFilter().fit(Xtr)
    keep = [int(i) for i in np.nonzero(uf.features_to_keep_)[0]]
    Xtr_k, Xte_k = Xtr[:, keep], Xte[:, keep]
    entry = {"name": name, "X_train": Xtr, "X_test": Xte, "keep": keep, "pipelines": {}}
    for norm in ("none", "power"):
      p = PreprocessingPipeline(normalization_method=norm, outlier_threshold=4.0, random_state=42).fit(Xtr_k)
      pe = {
        "scaler_mean": p.standard_scaler_.mean_,
        "scaler_scale": p.standard_scaler_.scale_,
        "lower": p.outlier_remover_.lower_bounds_,
        "upper": p.outlier_remover_.upper_bounds_,
        "train": p.X_transformed_,
        "test": p.transform(Xte_k),
      }
      if norm == "power":
        pe["lambdas"] = p.normalizer_.lambdas_
        pe["power_mean"] = p.normalizer_._scaler.mean_
        pe["power_scale"] = p.normalizer_._scaler.scale_
      entry["pipelines"][norm] = pe
    out["tables"].append(entry)
  return out


def quantile_fixture(model):
  from tabicl._model.quantile_dist import QuantileDistribution  # noqa: F401  (type used via model.quantile_dist)

  rng = np.random.default_rng(11)
  vectors = []
  base = np.sort(rng.normal(size=999)).astype(np.float32)
  vectors.append(base)
  vectors.append((rng.standard_t(3, size=999) * 2 + 5).astype(np.float32))  # heavy tails, unsorted
  vectors.append(np.linspace(-1, 1, 999, dtype=np.float32))
  alphas = [0.0005, 0.1, 0.5, 0.9, 0.9995]
  raw = torch.tensor(np.stack(vectors))[None]  # (1, 3, 999)
  dist = model.quantile_dist(raw)
  return {
    "alphas999": model.quantile_dist.alphas.cpu().numpy() if hasattr(model.quantile_dist, "alphas") else torch.linspace(0, 1, 1001)[1:-1].numpy(),
    "alphas": alphas,
    "cases": [
      {
        "raw": vectors[i],
        "mean": float(raw[0, i].mean()),
        "median": float(dist.icdf(torch.tensor(0.5))[0, i]),
        "icdf": dist.icdf(torch.tensor(alphas, dtype=torch.float32))[0, i].numpy(),
      }
      for i in range(len(vectors))
    ],
  }


# ---------- task tables ----------


def tables(task):
  rng = np.random.default_rng(7)
  T = []

  def num(n, f):
    return rng.normal(size=(n, f))

  def frame(X, names=None):
    return pd.DataFrame(X, columns=names or [f"x{i}" for i in range(X.shape[1])])

  def split(df, y, n_test):
    return df.iloc[:-n_test].reset_index(drop=True), y[:-n_test], df.iloc[-n_test:].reset_index(drop=True)

  if task == "classification":
    X = num(150, 4)
    y = np.digitize(X[:, 0] + 0.5 * X[:, 1], [-0.5, 0.5])
    T.append(("numeric_3class",) + split(frame(X), y, 30))

    n = 160
    df = frame(num(n, 3))
    df.insert(0, "color", pd.array(rng.choice(["red", "green", "blue", "amber"], size=n), dtype="string"))
    df.insert(2, "flag", pd.array(rng.random(n) > 0.4, dtype="boolean"))
    score = df["x0"] + (df["color"] == "red").astype(float) + df["flag"].astype(float)
    y = np.where(score > 0.8, "yes", "no").astype(object)
    T.append(("mixed_categorical",) + split(df, y, 30))

    X = num(120, 5)
    y = (X[:, 0] - X[:, 2] > 0).astype(int)
    Xm = X.copy()
    Xm[rng.random(Xm.shape) < 0.1] = np.nan
    Xm[-1, :] = np.nan
    T.append(("missing_values",) + split(frame(Xm), y, 20))

    X = num(100, 3)
    X[:, 1] = 7.0
    y = (X[:, 0] > 0).astype(int)
    T.append(("constant_column",) + split(frame(X), y, 20))

    X = num(400, 6)
    y = np.digitize(X[:, 0] * 2 + X[:, 1], np.quantile(X[:, 0] * 2 + X[:, 1], np.linspace(0.1, 0.9, 9)))
    T.append(("ten_classes",) + split(frame(X), y, 50))

    X = num(20, 3)
    y = (X[:, 0] > 0).astype(int)
    T.append(("tiny",) + split(frame(X), y, 5))

    X = num(1000, 20)
    y = (X[:, :3].sum(1) > 0).astype(int)
    T.append(("medium",) + split(frame(X), y, 100))

    n = 80
    df = frame(num(n, 2))
    cats = rng.choice(["a", "b", "c"], size=n).astype(object)
    cats[-3:] = "zzz"  # unseen in training
    df.insert(0, "kind", pd.array(cats, dtype="string"))
    y = ((df["x0"] > 0) ^ (df["kind"] == "a")).astype(int).to_numpy()
    T.append(("unseen_category",) + split(df, y, 10))
  else:
    X = num(200, 4)
    y = X @ np.array([1.5, -2.0, 0.5, 0.0]) + rng.normal(scale=0.3, size=200) + 10
    T.append(("linear_noise",) + split(frame(X), y, 40))

    X = num(200, 3)
    y = X[:, 0] * 3 + rng.standard_t(2, size=200)
    T.append(("heavy_tail_outliers",) + split(frame(X), y, 40))

    n = 160
    df = frame(num(n, 2))
    df.insert(0, "region", pd.array(rng.choice(["north", "south", "east"], size=n), dtype="string"))
    y = df["x0"].to_numpy() * 2 + (df["region"] == "south").to_numpy() * 5.0 + rng.normal(scale=0.5, size=n)
    T.append(("mixed_categorical",) + split(df, y, 30))

    X = num(20, 2)
    y = X[:, 0] - X[:, 1]
    T.append(("tiny",) + split(frame(X), y, 5))

    X = num(1000, 20)
    y = X[:, :5].sum(1) + rng.normal(scale=0.2, size=1000)
    T.append(("medium",) + split(frame(X), y, 100))
  return T


def to_rows(df):
  rows = []
  for rec in df.itertuples(index=False):
    row = []
    for v in rec:
      if v is pd.NA or v is None or (isinstance(v, float) and np.isnan(v)):
        row.append(None)
      elif isinstance(v, (bool, np.bool_)):
        row.append(bool(v))
      elif isinstance(v, (int, float, np.integer, np.floating)):
        row.append(float(v))
      else:
        row.append(str(v))
    rows.append(row)
  return rows


def capture_members(est):
  """Record raw network outputs per ensemble member during predict."""
  outs = []
  orig = est.model_._inference_forward

  def wrapped(*a, **k):
    r = orig(*a, **k)
    outs.append(r.detach().float().cpu().numpy())
    return r

  est.model_._inference_forward = wrapped
  return outs, lambda: setattr(est.model_, "_inference_forward", orig)


def main():
  if len(sys.argv) < 2:
    sys.exit("usage: make_reference.py <model>")
  name = sys.argv[1]
  meta = load_meta(name)
  ckpt = checkpoint(meta)
  task = meta["task"]
  from tabicl import TabICLClassifier, TabICLRegressor

  common = dict(random_state=42, use_amp=False, use_fa3=False, kv_cache=False, device="cpu", model_path=ckpt, allow_auto_download=False)
  out = {"source": f"tabicl 2.2.0 {meta['source']}@{meta['revision'][:7]} {meta['checkpoint']}", "task": task, "tables": []}
  for tname, Xtr, ytr, Xte in tables(task):
    est = (TabICLClassifier(checkpoint_version=meta["checkpoint"], **common) if task == "classification" else TabICLRegressor(checkpoint_version=meta["checkpoint"], **common))
    est.fit(Xtr, ytr)
    cat_cols = []
    for tname_, tfm, pos in est.X_encoder_.tfm_.transformers_ if hasattr(est.X_encoder_.tfm_, "transformers_") else []:
      if tname_ == "categorical":
        cat_cols = [int(p) for p in pos]
    record = {
      "name": tname,
      "columns": list(map(str, Xtr.columns)),
      "categorical_columns": cat_cols,
      "X_train": to_rows(Xtr),
      "y_train": [v if isinstance(v, str) else float(v) for v in ytr],
      "X_test": to_rows(Xte),
      "expected": {},
    }
    want_members = tname in ("numeric_3class", "linear_noise")
    if want_members:
      members, restore = capture_members(est)
    if task == "classification":
      record["expected"]["proba"] = est.predict_proba(Xte)
      record["expected"]["labels"] = [v if isinstance(v, str) else float(v) for v in est.predict(Xte)]
      record["classes"] = [v if isinstance(v, str) else float(v) for v in est.classes_]
    else:
      res = est.predict(Xte, output_type=["mean", "median", "quantiles"])
      record["expected"].update({k: res[k] for k in ("mean", "median", "quantiles")})
      record["y_mean"] = est.y_scaler_.mean_
      record["y_scale"] = est.y_scaler_.scale_
    if want_members:
      restore()
      # First 5 test rows are enough to check the combine step and keep the fixture small.
      record["member_outputs"] = np.concatenate(members, axis=0)[:, :5]
    out["tables"].append(record)
    print(tname, "done")
  dump(os.path.join(FIX, name, "reference.json"), out)

  if task == "classification":
    dump(os.path.join(FIX, "tabicl", "rng.json"), rng_fixture())
    dump(os.path.join(FIX, "tabicl", "latin.json"), latin_fixture())
    dump(os.path.join(FIX, "tabicl", "ensemble.json"), ensemble_fixture())
    dump(os.path.join(FIX, "tabicl", "preprocess.json"), preprocess_fixture())
  else:
    est = TabICLRegressor(checkpoint_version=meta["checkpoint"], **common)
    rng = np.random.default_rng(1)
    est.fit(pd.DataFrame(rng.normal(size=(30, 2))), rng.normal(size=30))
    dump(os.path.join(FIX, "tabicl", "quantile.json"), quantile_fixture(est.model_))


if __name__ == "__main__":
  main()
