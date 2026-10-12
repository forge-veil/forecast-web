import { articleItem, changeText, cityItem, datasetItem, inSentence } from "src/lib/items";

it.each([
  ["Sunspot number", "sunspot number"],
  ["US retail sales", "US retail sales"],
  ["CO2 at Mauna Loa", "CO2 at Mauna Loa"],
])("inSentence %s", (a, b) => expect(inSentence(a)).toBe(b));

describe("changeText", () => {
  it("uses percent for ratio units", () => {
    expect(changeText(100, 105, { ratio: true, unit: "USD" })).toBe("+5.0%");
    expect(changeText(100, 50, { ratio: true, unit: "USD" })).toBe("−50%");
  });
  it("uses absolute change for temperatures", () => {
    expect(changeText(10, 12.34, { ratio: false, unit: "°C" })).toBe("+2.3°");
    expect(changeText(-2, -5, { ratio: false, unit: "°C" })).toBe("−3.0°");
  });
  it("handles a zero start", () => {
    expect(changeText(0, 3, { ratio: true, unit: "" })).toBe("+3.0");
  });
});

describe("item builders", () => {
  it("splits a stock title into symbol and company", () => {
    const i = datasetItem({ id: "stock-aapl", group: "stocks", title: "Apple (AAPL)", unit: "USD", freq: "daily", source: "Yahoo Finance", sourceUrl: "", file: "x", available: true });
    expect(i.name).toBe("AAPL");
    expect(i.subtitle).toBe("Apple");
    expect(i.phrase).toBe("the Apple closing price");
    expect(i.ranges[i.defaultRange].label).toBe("1M");
  });
  it("gives cities and articles stable ids", () => {
    expect(cityItem({ name: "Tokyo", detail: "Japan", latitude: 35.6895, longitude: 139.6917 }).id).toBe("city-35.69,139.69");
    expect(articleItem("Tour de France").id).toBe("wiki-tour_de_france");
  });
});

describe("group pill descriptions", () => {
  it("match each group's default range", async () => {
    const { GROUPS, YOURS } = await import("src/lib/items");
    const words: Record<string, string> = { "1M": "next month", "2D": "next 2 days", "2Y": "next 2 years", "20": "next 20 steps" };
    const samples = {
      stocks: datasetItem({ id: "s", group: "stocks", title: "X (X)", unit: "USD", freq: "daily", source: "", sourceUrl: "", file: "", available: true }),
      weather: cityItem({ name: "A", detail: "", latitude: 0, longitude: 0 }),
      traffic: articleItem("A"),
      seasonal: datasetItem({ id: "c", group: "classic", title: "C", unit: "u", freq: "monthly", source: "", sourceUrl: "", file: "", available: true }),
      yours: YOURS,
    };
    for (const g of GROUPS) {
      const item = samples[g.id];
      expect(g.pill).toContain(words[item.ranges[item.defaultRange].label]);
    }
  });
});
