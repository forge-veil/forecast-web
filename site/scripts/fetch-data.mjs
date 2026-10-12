// Fetches the bundled datasets into public/data/ at build time. Each source is
// independent: a failure is recorded in manifest.json and the build carries on.
// Stock prices are fetched here only and are never committed to git.
import { mkdir, writeFile } from "node:fs/promises";

const OUT = new URL("../public/data/", import.meta.url);
const UA = { "user-agent": "Mozilla/5.0 (timesfm-web demo build; +https://github.com/forge-veil/timesfm-web)" };

async function get(url) {
  const res = await fetch(url, { headers: UA, signal: AbortSignal.timeout(30_000) });
  if (!res.ok) throw new Error(`${url} returned HTTP ${res.status}`);
  return res.text();
}

const STOCKS = [
  ["SPY", "S&P 500 ETF"],
  ["QQQ", "Nasdaq-100 ETF"],
  ["AAPL", "Apple"],
  ["MSFT", "Microsoft"],
  ["NVDA", "NVIDIA"],
  ["TSLA", "Tesla"],
];

function stock(symbol, name) {
  return {
    id: `stock-${symbol.toLowerCase()}`,
    group: "stocks",
    title: `${name} (${symbol})`,
    unit: "USD",
    freq: "daily",
    source: "Yahoo Finance",
    sourceUrl: `https://finance.yahoo.com/quote/${symbol}`,
    async load() {
      const json = JSON.parse(
        await get(`https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?range=2y&interval=1d`),
      );
      const r = json.chart?.result?.[0];
      if (!r) throw new Error(json.chart?.error?.description ?? "no result");
      const closes = r.indicators.adjclose?.[0]?.adjclose ?? r.indicators.quote[0].close;
      const pairs = r.timestamp.map((t, i) => [new Date(t * 1000).toISOString().slice(0, 10), closes[i]]);
      return pairs.filter(([, v]) => v !== null && Number.isFinite(v));
    },
  };
}

const monthly = (y, m) => `${y}-${String(m).padStart(2, "0")}`;

async function fred(series) {
  const rows = (await get(`https://fred.stlouisfed.org/graph/fredgraph.csv?id=${series}`)).trim().split("\n").slice(1);
  return rows
    .map((l) => l.split(","))
    .filter(([, v]) => v && v !== "." && Number.isFinite(Number(v)))
    .map(([d, v]) => [d.slice(0, 7), Number(v)]);
}

const CLASSIC = [
  {
    id: "co2",
    title: "CO2 at Mauna Loa",
    unit: "ppm",
    freq: "monthly",
    season: 12,
    source: "NOAA Global Monitoring Laboratory",
    sourceUrl: "https://gml.noaa.gov/ccgg/trends/",
    license: "Public domain",
    async load() {
      const csv = await get("https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv");
      return csv
        .split("\n")
        .filter((l) => l && !l.startsWith("#") && !l.startsWith("year"))
        .map((l) => l.split(","))
        .filter((c) => Number(c[3]) > 0)
        .map((c) => [monthly(c[0], c[1]), Number(c[3])]);
    },
  },
  {
    id: "electricity",
    title: "US electric and gas utility output",
    unit: "index, 2017 = 100",
    freq: "monthly",
    season: 12,
    source: "Federal Reserve via FRED (IPG2211A2N)",
    sourceUrl: "https://fred.stlouisfed.org/series/IPG2211A2N",
    license: "Public domain",
    load: () => fred("IPG2211A2N"),
  },
  {
    id: "retail",
    title: "US retail sales",
    unit: "million USD",
    freq: "monthly",
    season: 12,
    source: "US Census Bureau via FRED (RSXFSN)",
    sourceUrl: "https://fred.stlouisfed.org/series/RSXFSN",
    license: "Public domain",
    load: () => fred("RSXFSN"),
  },
  {
    id: "sunspots",
    title: "Sunspot number",
    unit: "monthly mean",
    freq: "monthly",
    season: 132,
    source: "SILSO, Royal Observatory of Belgium",
    sourceUrl: "https://www.sidc.be/SILSO/",
    license: "CC BY-NC 4.0",
    async load() {
      const csv = await get("https://www.sidc.be/SILSO/INFO/snmtotcsv.php");
      return csv
        .trim()
        .split("\n")
        .map((l) => l.split(";").map((s) => s.trim()))
        .filter((c) => Number(c[3]) >= 0)
        .map((c) => [monthly(c[0], Number(c[1])), Number(c[3])]);
    },
  },
  {
    id: "airline",
    title: "Airline passengers, 1949 to 1960",
    unit: "thousands",
    freq: "monthly",
    season: 12,
    source: "Box and Jenkins, Time Series Analysis",
    sourceUrl: "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
    license: "Public dataset",
    async load() {
      const csv = await get("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv");
      return csv
        .trim()
        .split("\n")
        .slice(1)
        .map((l) => l.replaceAll('"', "").split(","))
        .map(([d, v]) => [d, Number(v)]);
    },
  },
];

const datasets = [...STOCKS.map(([s, n]) => stock(s, n)), ...CLASSIC.map((d) => ({ group: "classic", ...d }))];

await mkdir(OUT, { recursive: true });
const manifest = { generatedAt: new Date().toISOString(), datasets: [] };

await Promise.all(
  datasets.map(async ({ load, ...meta }) => {
    const entry = { ...meta, file: `${meta.id}.json` };
    try {
      const pairs = await load();
      if (pairs.length < 32) throw new Error(`only ${pairs.length} points`);
      await writeFile(
        new URL(entry.file, OUT),
        JSON.stringify({ dates: pairs.map((p) => p[0]), values: pairs.map((p) => p[1]) }),
      );
      Object.assign(entry, { available: true, points: pairs.length, last: pairs.at(-1)[0] });
      console.log(`ok    ${meta.id.padEnd(12)} ${pairs.length} points, last ${entry.last}`);
    } catch (e) {
      Object.assign(entry, { available: false, error: String(e.message ?? e) });
      console.warn(`FAIL  ${meta.id.padEnd(12)} ${entry.error}`);
    }
    manifest.datasets.push(entry);
  }),
);

manifest.datasets.sort((a, b) => datasets.findIndex((d) => d.id === a.id) - datasets.findIndex((d) => d.id === b.id));
await writeFile(new URL("manifest.json", OUT), JSON.stringify(manifest, null, 1));
const failed = manifest.datasets.filter((d) => !d.available).length;
console.log(`${manifest.datasets.length - failed} of ${manifest.datasets.length} datasets fetched`);
