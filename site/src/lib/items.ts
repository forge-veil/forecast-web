import { loadDataset, loadPageviews, loadWeather, type City, type DatasetEntry, type SeriesData } from "src/lib/sources";

export type GroupId = "stocks" | "weather" | "traffic" | "seasonal" | "yours";

export interface Group {
  id: GroupId;
  label: string;
  /** What the series in this group are. */
  about: string;
  /** What the change pill means for this group. */
  pill: string;
}

export const GROUPS: Group[] = [
  { id: "stocks", label: "Stocks", about: "Daily closing prices", pill: "Forecast change · next month" },
  { id: "weather", label: "Weather", about: "Hourly temperature, live", pill: "Forecast change · next 2 days" },
  { id: "traffic", label: "Web traffic", about: "Daily Wikipedia views, live", pill: "Forecast change · next month" },
  { id: "seasonal", label: "Seasonal", about: "Monthly, seasonal", pill: "Forecast change · next 2 years" },
  { id: "yours", label: "Your data", about: "Paste or draw", pill: "Forecast change · next 20 steps" },
];

export interface Range {
  label: string;
  steps: number;
}

export interface Item {
  id: string;
  group: GroupId;
  /** Short name for the watchlist row, e.g. "AAPL" or "Tokyo". */
  name: string;
  /** Gray line under the name. */
  subtitle: string;
  /** Phrase that reads inside a sentence: "the Apple closing price". */
  phrase: string;
  unit: string;
  /** Whether a percent change is meaningful (false for temperatures). */
  ratio: boolean;
  stepName: "days" | "hours" | "months" | "steps";
  season?: number;
  ranges: Range[];
  defaultRange: number;
  source?: { text: string; url: string; license?: string };
  note?: string;
  load?: () => Promise<SeriesData>;
}

const DAILY_MARKET: Range[] = [
  { label: "1W", steps: 5 },
  { label: "2W", steps: 10 },
  { label: "1M", steps: 21 },
  { label: "3M", steps: 63 },
  { label: "6M", steps: 126 },
];
const DAILY: Range[] = [
  { label: "1W", steps: 7 },
  { label: "2W", steps: 14 },
  { label: "1M", steps: 30 },
  { label: "2M", steps: 60 },
  { label: "4M", steps: 120 },
];
const HOURLY: Range[] = [
  { label: "12H", steps: 12 },
  { label: "1D", steps: 24 },
  { label: "2D", steps: 48 },
  { label: "3D", steps: 72 },
  { label: "5D", steps: 120 },
];
const MONTHLY: Range[] = [
  { label: "6M", steps: 6 },
  { label: "1Y", steps: 12 },
  { label: "2Y", steps: 24 },
  { label: "5Y", steps: 60 },
  { label: "10Y", steps: 120 },
];
const STEPS: Range[] = [
  { label: "5", steps: 5 },
  { label: "10", steps: 10 },
  { label: "20", steps: 20 },
  { label: "50", steps: 50 },
  { label: "100", steps: 100 },
];

/** Lowercases a leading capitalized word so the title reads mid-sentence; acronyms (US, CO2) stay. */
export const inSentence = (s: string) => (/^[A-Z][a-z]/.test(s) ? s[0].toLowerCase() + s.slice(1) : s);

export function datasetItem(e: DatasetEntry): Item {
  const stock = e.group === "stocks";
  const symbol = stock ? /\(([^)]+)\)$/.exec(e.title)?.[1] ?? e.title : undefined;
  const company = stock ? e.title.replace(/\s*\([^)]+\)$/, "") : undefined;
  return {
    id: e.id,
    group: stock ? "stocks" : "seasonal",
    name: symbol ?? e.title,
    subtitle: stock ? company! : `${e.unit} · monthly`,
    phrase: stock ? `the ${company} closing price` : inSentence(e.title),
    unit: stock ? "USD" : e.unit,
    ratio: true,
    stepName: e.freq === "daily" ? "days" : "months",
    season: e.season,
    ranges: stock ? DAILY_MARKET : MONTHLY,
    defaultRange: 2,
    source: { text: e.source, url: e.sourceUrl, license: e.license },
    note: stock
      ? "Not investment advice. Prices are close to a random walk; try Backtest."
      : undefined,
    load: () => loadDataset(e),
  };
}

export function cityItem(c: City): Item {
  return {
    id: `city-${c.latitude.toFixed(2)},${c.longitude.toFixed(2)}`,
    group: "weather",
    name: c.name,
    subtitle: c.detail ? `${c.detail.split(", ").pop()} · hourly °C` : "hourly °C",
    phrase: `temperature in ${c.name}`,
    unit: "°C",
    ratio: false,
    stepName: "hours",
    season: 24,
    ranges: HOURLY,
    defaultRange: 2,
    source: { text: "Open-Meteo", url: "https://open-meteo.com/", license: "CC BY 4.0" },
    note: "The model sees only past temperatures. Dashed green: Open-Meteo's physics forecast.",
    load: () => loadWeather(c),
  };
}

export function articleItem(title: string): Item {
  const t = title.trim();
  return {
    id: `wiki-${t.toLowerCase().replaceAll(" ", "_")}`,
    group: "traffic",
    name: t.length > 24 ? `${t.slice(0, 23)}…` : t,
    subtitle: "Wikipedia · daily views",
    phrase: `daily views of the "${t}" article`,
    unit: "views",
    ratio: true,
    stepName: "days",
    season: 7,
    ranges: DAILY,
    defaultRange: 2,
    source: { text: "Wikimedia pageviews API", url: "https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews" },
    note: "Weekly and yearly patterns forecast well; news spikes do not.",
    load: () => loadPageviews(t),
  };
}

export const YOURS: Item = {
  id: "yours",
  group: "yours",
  name: "Your series",
  subtitle: "Paste or draw",
  phrase: "your series",
  unit: "",
  ratio: true,
  stepName: "steps",
  ranges: STEPS,
  defaultRange: 2,
};

export const PRESET_CITIES: City[] = [
  { name: "San Francisco", detail: "California, United States", latitude: 37.7749, longitude: -122.4194 },
  { name: "London", detail: "England, United Kingdom", latitude: 51.5085, longitude: -0.1257 },
  { name: "Tokyo", detail: "Japan", latitude: 35.6895, longitude: 139.6917 },
  { name: "Mumbai", detail: "Maharashtra, India", latitude: 19.0728, longitude: 72.8826 },
];

export const PRESET_ARTICLES = ["Christmas", "Python (programming language)", "Tour de France", "Halloween"];

/** "+2.3%" for ratio units, "+2.1 °C" otherwise. */
export function changeText(from: number, to: number, item: Pick<Item, "ratio" | "unit">): string {
  if (!Number.isFinite(from) || !Number.isFinite(to)) return "–";
  const d = to - from;
  const sign = d > 0 ? "+" : d < 0 ? "−" : "";
  if (item.ratio && from !== 0) return `${sign}${Math.abs((d / Math.abs(from)) * 100).toFixed(Math.abs(d / from) < 0.1 ? 1 : 0)}%`;
  const a = Math.abs(d);
  return `${sign}${a >= 100 ? a.toFixed(0) : a.toFixed(1)}${item.unit === "°C" ? "°" : ""}`;
}
