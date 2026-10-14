import { parseTable } from "src/lib/csv";
import { accuracy, columnKind, coverage, holdoutSplit, mae, majorityBaseline, meanBaselineError, pageOf, perClass, rankVariants, suggestTask } from "src/lib/tables";

describe("parseTable", () => {
  it("reads a header, numbers, text and missing cells", () => {
    const t = parseTable("species,mass,sex\nAdelie,3750,male\nGentoo,NA,\n");
    expect(t.columns).toEqual(["species", "mass", "sex"]);
    expect(t.rows).toEqual([
      ["Adelie", 3750, "male"],
      ["Gentoo", null, null],
    ]);
  });

  it("handles quoted fields with commas, quotes and other delimiters", () => {
    expect(parseTable('a,b\n"x, y","say ""hi"""\n').rows).toEqual([["x, y", 'say "hi"']]);
    expect(parseTable("a;b\n1;2\n").rows).toEqual([[1, 2]]);
    expect(parseTable("a\tb\n1\t2\n").rows).toEqual([[1, 2]]);
  });

  it("keeps numeric-looking text numeric so it is not treated as categories", () => {
    expect(parseTable("v\n10\n9\n").rows).toEqual([[10], [9]]);
  });

  it("pads short rows with missing cells", () => {
    expect(parseTable("a,b,c\n1,2\n").rows).toEqual([[1, 2, null]]);
  });
});

describe("suggestTask", () => {
  it("suggests classification for text or few distinct integers, regression otherwise", () => {
    expect(suggestTask(["a", "b", "a"])).toBe("classification");
    expect(suggestTask([0, 1, 1, 0, 2])).toBe("classification");
    expect(suggestTask([3750, 3800, 4200.5, 5000, 3100, 2900, 4100, 4300, 3600, 3700, 3900, 3950])).toBe("regression");
  });
});

describe("holdoutSplit", () => {
  it("is deterministic and keeps every labeled row exactly once", () => {
    const a = holdoutSplit(100, 0.2, 7);
    expect(holdoutSplit(100, 0.2, 7)).toEqual(a);
    expect(a.test).toHaveLength(20);
    expect([...a.train, ...a.test].sort((x, y) => x - y)).toEqual(Array.from({ length: 100 }, (_, i) => i));
  });
});

describe("metrics", () => {
  it("computes accuracy, mean absolute error and interval coverage", () => {
    expect(accuracy(["a", "b", "a"], ["a", "a", "a"])).toBeCloseTo(2 / 3);
    expect(mae([1, 2, 3], [2, 2, 5])).toBeCloseTo(1);
    expect(coverage([1, 5, 10], [[0, 2], [6, 8], [9, 11]])).toBeCloseTo(2 / 3);
  });
});

describe("rankVariants", () => {
  it("ranks classification by accuracy (higher first) and regression by error (lower first), unscored last", () => {
    expect(rankVariants([{ id: "a", score: 0.8 }, { id: "b" }, { id: "c", score: 0.9 }], "classification").map((r) => r.id)).toEqual(["c", "a", "b"]);
    expect(rankVariants([{ id: "a", score: 5 }, { id: "b", score: 3 }], "regression").map((r) => r.id)).toEqual(["b", "a"]);
  });
});

describe("columnKind", () => {
  it("is number when every present value is numeric, category otherwise", () => {
    expect(columnKind([1, null, 2.5])).toBe("number");
    expect(columnKind(["a", 1, null])).toBe("category");
    expect(columnKind([null, null])).toBe("number");
  });
});

describe("pageOf", () => {
  it("returns the row range, page count and a clamped page", () => {
    expect(pageOf(344, 0, 10)).toEqual({ page: 0, pages: 35, start: 0, end: 10 });
    expect(pageOf(344, 34, 10)).toEqual({ page: 34, pages: 35, start: 340, end: 344 });
    expect(pageOf(344, 99, 10)).toEqual({ page: 34, pages: 35, start: 340, end: 344 });
    expect(pageOf(0, 3, 10)).toEqual({ page: 0, pages: 1, start: 0, end: 0 });
  });
});

describe("backtest baselines and breakdown", () => {
  it("scores always guessing the most common training label", () => {
    expect(majorityBaseline(["a", "a", "b"], ["a", "b", "b", "a"])).toEqual({ label: "a", accuracy: 0.5 });
  });

  it("scores always guessing the training average", () => {
    expect(meanBaselineError([1, 3], [2, 4])).toBeCloseTo(1);
  });

  it("counts right and total per actual class, most frequent first", () => {
    expect(perClass(["a", "b", "a", "a"], ["a", "a", "a", "b"])).toEqual([
      { label: "a", right: 2, total: 3 },
      { label: "b", right: 0, total: 1 },
    ]);
  });
});
