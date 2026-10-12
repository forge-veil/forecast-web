import { holdout, parseCsv, parseNumbers, pickColumn, resample } from "src/lib/series";

describe("parseNumbers", () => {
  it("splits on commas, whitespace, semicolons and newlines", () => {
    expect(parseNumbers("1, 2 3;4\n5\t6")).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("ignores junk tokens and blank lines", () => {
    expect(parseNumbers("value\n\n1.5\nabc\n-2e3\n")).toEqual([1.5, -2000]);
  });

  it("returns an empty array for no numbers", () => {
    expect(parseNumbers("hello world")).toEqual([]);
  });
});

describe("parseCsv", () => {
  it("detects a header and the delimiter", () => {
    const p = parseCsv("date;sales\n2024-01;10\n2024-02;12\n");
    expect(p.columns).toEqual(["date", "sales"]);
    expect(p.rows).toEqual([
      [null, 10],
      [null, 12],
    ]);
  });

  it("names columns when there is no header", () => {
    const p = parseCsv("1,2\n3,4");
    expect(p.columns).toEqual(["Column 1", "Column 2"]);
    expect(p.rows).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it("handles tabs and quoted cells", () => {
    const p = parseCsv('"Month"\t"Passengers"\n"1949-01"\t112');
    expect(p.columns).toEqual(["Month", "Passengers"]);
    expect(p.rows[0][1]).toBe(112);
  });

  it("keeps the raw text of data rows for label columns", () => {
    expect(parseCsv("date,v\n2024-01,3\n2024-02,4").raw).toEqual([
      ["2024-01", "3"],
      ["2024-02", "4"],
    ]);
  });

  it("keeps empty cells as missing", () => {
    expect(parseCsv("a,b\n1,\n2,5").rows).toEqual([
      [1, null],
      [2, 5],
    ]);
  });
});

describe("pickColumn", () => {
  it("defaults to the last column that has numbers", () => {
    const p = parseCsv("date,value,note\n2024-01,3,x\n2024-02,4,y");
    expect(pickColumn(p)).toEqual({ index: 1, values: [3, 4] });
  });

  it("returns the requested column", () => {
    const p = parseCsv("a,b\n1,2\n3,4");
    expect(pickColumn(p, 0).values).toEqual([1, 3]);
  });
});

describe("holdout", () => {
  it("splits off the last n points", () => {
    expect(holdout([1, 2, 3, 4, 5], 2)).toEqual({ train: [1, 2, 3], test: [4, 5] });
  });

  it("always leaves at least two training points", () => {
    expect(holdout([1, 2, 3], 10)).toEqual({ train: [1, 2], test: [3] });
  });

  it("treats zero or negative as no holdout", () => {
    expect(holdout([1, 2, 3], 0)).toEqual({ train: [1, 2, 3], test: [] });
  });
});

describe("resample", () => {
  it("interpolates pointer samples onto an even grid", () => {
    const out = resample(
      [
        { x: 0, y: 0 },
        { x: 10, y: 10 },
      ],
      6,
    );
    expect(out).toEqual([0, 2, 4, 6, 8, 10]);
  });

  it("sorts out-of-order samples and keeps the last y for duplicate x", () => {
    const out = resample(
      [
        { x: 10, y: 0 },
        { x: 0, y: 4 },
        { x: 10, y: 8 },
      ],
      3,
    );
    expect(out).toEqual([4, 6, 8]);
  });

  it("returns an empty array for fewer than two samples", () => {
    expect(resample([{ x: 1, y: 1 }], 5)).toEqual([]);
  });
});

describe("extendLabels", () => {
  it("continues monthly labels across a year boundary", async () => {
    const { extendLabels } = await import("src/lib/series");
    expect(extendLabels(["2026-11"], 3, "monthly")).toEqual(["2026-12", "2027-01", "2027-02"]);
  });
  it("continues daily labels by calendar day", async () => {
    const { extendLabels } = await import("src/lib/series");
    expect(extendLabels(["2026-02-27"], 3, "daily")).toEqual(["2026-02-28", "2026-03-01", "2026-03-02"]);
  });
  it("skips weekends for trading days", async () => {
    const { extendLabels } = await import("src/lib/series");
    // 2026-10-09 is a Friday
    expect(extendLabels(["2026-10-09"], 3, "weekdays")).toEqual(["2026-10-12", "2026-10-13", "2026-10-14"]);
  });
  it("returns nothing for labels it cannot parse", async () => {
    const { extendLabels } = await import("src/lib/series");
    expect(extendLabels(["step 4"], 2, "daily")).toEqual([]);
  });
});
