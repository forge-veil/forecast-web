import { extendLabels } from "src/lib/series";

export interface DatasetEntry {
  id: string;
  group: "stocks" | "classic";
  title: string;
  unit: string;
  freq: "daily" | "monthly";
  season?: number;
  source: string;
  sourceUrl: string;
  license?: string;
  file: string;
  available: boolean;
  error?: string;
  points?: number;
  last?: string;
}

export interface Manifest {
  generatedAt: string;
  datasets: DatasetEntry[];
}

/** A series ready to forecast: values with matching labels, plus optional known future labels. */
export interface SeriesData {
  values: number[];
  labels: string[];
  futureLabels?: string[];
  /** A forecast from another source, aligned with futureLabels, for comparison. */
  comparison?: { label: string; values: number[] };
}

const DATA = `${import.meta.env.BASE_URL}data/`;

async function json<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(res.status === 404 ? "not found" : `request failed (HTTP ${res.status})`);
  return res.json() as Promise<T>;
}

export const loadManifest = () => json<Manifest>(DATA + "manifest.json");

export async function loadDataset(entry: DatasetEntry): Promise<SeriesData> {
  const d = await json<{ dates: string[]; values: number[] }>(DATA + entry.file);
  // Stocks trade on weekdays; holidays are not modeled, so future dates are approximate.
  const kind = entry.freq === "monthly" ? "monthly" : entry.group === "stocks" ? "weekdays" : "daily";
  return { values: d.values, labels: d.dates, futureLabels: extendLabels(d.dates, 128, kind) };
}

export interface City {
  name: string;
  detail: string;
  latitude: number;
  longitude: number;
}

export async function searchCities(query: string): Promise<City[]> {
  const q = query.trim();
  if (q.length < 2) return [];
  const r = await json<{ results?: { name: string; country?: string; admin1?: string; latitude: number; longitude: number }[] }>(
    `https://geocoding-api.open-meteo.com/v1/search?count=6&language=en&name=${encodeURIComponent(q)}`,
  );
  return (r.results ?? []).map((c) => ({
    name: c.name,
    detail: [c.admin1, c.country].filter(Boolean).join(", "),
    latitude: c.latitude,
    longitude: c.longitude,
  }));
}

/** Last 21 days of hourly temperature plus Open-Meteo's own 7-day forecast, split at the current hour. */
export async function loadWeather(city: City): Promise<SeriesData> {
  const url =
    `https://api.open-meteo.com/v1/forecast?latitude=${city.latitude}&longitude=${city.longitude}` +
    `&hourly=temperature_2m&current=temperature_2m&past_days=21&forecast_days=7&timezone=auto`;
  const r = await json<{ current: { time: string }; hourly: { time: string[]; temperature_2m: (number | null)[] } }>(url);
  const now = r.current.time.slice(0, 13); // "YYYY-MM-DDTHH"
  let split = r.hourly.time.findIndex((t) => t.slice(0, 13) > now);
  if (split < 0) split = r.hourly.time.length;
  const fmt = (t: string) => t.replace("T", " ");
  const past = r.hourly.time.slice(0, split).map((t, i) => [fmt(t), r.hourly.temperature_2m[i]] as const);
  const kept = past.filter(([, v]) => v !== null) as [string, number][];
  if (kept.length < 48) throw new Error("not enough recent hours for this location");
  return {
    values: kept.map(([, v]) => v),
    labels: kept.map(([t]) => t),
    futureLabels: r.hourly.time.slice(split).map(fmt),
    comparison: {
      label: "Open-Meteo forecast",
      values: r.hourly.temperature_2m.slice(split).map((v) => v ?? NaN),
    },
  };
}

const ymd = (d: Date) => d.toISOString().slice(0, 10).replaceAll("-", "");

/** Daily user pageviews for an English Wikipedia article over the last ~2 years. */
export async function loadPageviews(article: string): Promise<SeriesData> {
  const title = article.trim().replaceAll(" ", "_");
  if (!title) throw new Error("enter an article title");
  const end = new Date(Date.now() - 86_400_000);
  const start = new Date(end.getTime() - 730 * 86_400_000);
  const url =
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/" +
    `${encodeURIComponent(title)}/daily/${ymd(start)}/${ymd(end)}`;
  let r: { items: { timestamp: string; views: number }[] };
  try {
    r = await json(url);
  } catch (e) {
    if ((e as Error).message === "not found") throw new Error(`no pageview data for "${article.trim()}" on English Wikipedia`);
    throw e;
  }
  if (r.items.length < 32) throw new Error("this article has too little history to forecast");
  const labels = r.items.map((i) => `${i.timestamp.slice(0, 4)}-${i.timestamp.slice(4, 6)}-${i.timestamp.slice(6, 8)}`);
  return { values: r.items.map((i) => i.views), labels, futureLabels: extendLabels(labels, 128, "daily") };
}
