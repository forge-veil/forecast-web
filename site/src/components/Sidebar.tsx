import { useEffect, useState } from "react";
import { MODELS, type Forecast } from "forecast-web";
import { Sparkline } from "src/components/Sparkline";
import { formatNumber } from "src/lib/format";
import { changeText, GROUPS, type GroupId, type Item } from "src/lib/items";
import { useModels } from "src/lib/models";
import { searchCities, type City } from "src/lib/sources";
import type { Entry } from "src/lib/useWatchlist";

const today = new Date().toLocaleDateString("en-US", { month: "long", day: "numeric" });

export function Sidebar(props: {
  items: Item[];
  data: Record<string, Entry>;
  forecastFor: (id: string, values?: number[]) => Forecast | undefined;
  selected: string;
  onSelect: (id: string) => void;
  onAddCity: (c: City) => void;
  onAddArticle: (title: string) => void;
}) {
  return (
    <aside className="sidebar" aria-label="Watchlist">
      <div className="sb-head">
        <h1 className="sb-title">Forecasts</h1>
        <p className="sb-date">{today}</p>
        <ModelStatus />
        <Search onAddCity={props.onAddCity} onAddArticle={props.onAddArticle} />
      </div>
      <nav className="list">
        {GROUPS.map((g) => {
          const rows = props.items.filter((i) => i.group === g.id);
          if (!rows.length) return null;
          return (
            <section key={g.id} aria-label={g.label} className={`group-sec g-${g.id}`}>
              <header className="group">
                <GroupIcon id={g.id} />
                <div>
                  <div className="group-name">{g.label}</div>
                  <div className="group-about">{g.about}</div>
                </div>
              </header>
              <div className="group-card">
              {rows.map((item) => (
                <Row
                  key={item.id}
                  item={item}
                  entry={props.data[item.id]}
                  forecast={props.forecastFor(item.id, props.data[item.id]?.data?.values)}
                  selected={item.id === props.selected}
                  onSelect={() => props.onSelect(item.id)}
                />
              ))}
              </div>
              <div className="group-pill">{g.pill}</div>
            </section>
          );
        })}
      </nav>
    </aside>
  );
}

function Row({ item, entry, forecast, selected, onSelect }: { item: Item; entry?: Entry; forecast?: Forecast; selected: boolean; onSelect: () => void }) {
  const values = entry?.data?.values;
  const last = values?.[values.length - 1];
  const steps = item.ranges[item.defaultRange].steps;
  const end = forecast?.median[steps - 1];
  const dir = end === undefined || last === undefined ? "flat" : end > last ? "up" : end < last ? "down" : "flat";
  const { choice } = useModels();
  return (
    <button className="row" aria-current={selected} onClick={onSelect}>
      <div style={{ minWidth: 0 }}>
        <div className="name">{item.name}</div>
        <div className="sub">{entry?.error ? "Unavailable" : item.subtitle}</div>
      </div>
      {values ? <Sparkline history={values.slice(-60)} forecast={forecast?.median.slice(0, steps)} dir={dir} /> : <span />}
      <div className="right">
        <div className="price">{last === undefined ? (entry?.error ? "" : "…") : formatNumber(last)}</div>
        <span className={`chg ${dir}`} title={`${MODELS[choice].name} forecast change over ${item.ranges[item.defaultRange].label}`}>
          {end !== undefined && last !== undefined ? changeText(last, end, item) : "N/A"}
        </span>
      </div>
    </button>
  );
}

const ICONS: Record<GroupId, string> = {
  stocks: "M3 13l4-4 3 3 6-7",
  weather: "M9 3a2 2 0 0 0-2 2v6.3a3.5 3.5 0 1 0 4 0V5a2 2 0 0 0-2-2z",
  traffic: "M10 2.5a7.5 7.5 0 1 0 0 15a7.5 7.5 0 1 0 0-15zM2.5 10h15M10 2.5c2.2 2.2 2.2 12.8 0 15M10 2.5c-2.2 2.2-2.2 12.8 0 15",
  seasonal: "M2 10c2.5-6 5.5-6 8 0s5.5 6 8 0",
  yours: "M4 16l1-4 8-8 3 3-8 8-4 1z",
};

function GroupIcon({ id }: { id: GroupId }) {
  return (
    <span className={`gicon g-${id}`} aria-hidden="true">
      <svg viewBox="0 0 20 20"><path d={ICONS[id]} fill="none" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
    </span>
  );
}

/** Which model fills the watchlist; models are loaded from the detail view's model bar. */
function ModelStatus() {
  const { choice, slot } = useModels();
  const { status, backend, progress } = slot(choice);
  const pct = progress?.total ? Math.round((100 * progress.loaded) / progress.total) : 0;
  return (
    <div className="model">
      {status === "ready" ? <i className="dot" /> : null}
      <span>
        {MODELS[choice].name}
        {status === "ready" ? ` · ${backend === "webgpu" ? "WebGPU" : "WebAssembly"}` : status === "loading" ? ` · ${pct ? `${pct}%` : "loading…"}` : " · not loaded"}
      </span>
    </div>
  );
}

function Search({ onAddCity, onAddArticle }: { onAddCity: (c: City) => void; onAddArticle: (t: string) => void }) {
  const [q, setQ] = useState("");
  const [cities, setCities] = useState<City[]>([]);

  useEffect(() => {
    if (q.trim().length < 2) return setCities([]);
    let live = true;
    const t = setTimeout(() => searchCities(q).then((c) => live && setCities(c.slice(0, 4)), () => live && setCities([])), 250);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [q]);

  const done = () => {
    setQ("");
    setCities([]);
  };

  return (
    <div className="search">
      <div style={{ position: "relative" }}>
        <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.8" /><path d="M11 11l3.5 3.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>
        <input
          type="search"
          placeholder="Add city or article"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Escape" && done()}
          aria-label="Add a city or Wikipedia article to the watchlist"
        />
      </div>
      {q.trim().length >= 2 && (
        <div className="suggest">
          {cities.map((c) => (
            <button key={`${c.latitude},${c.longitude}`} onClick={() => (onAddCity(c), done())}>
              {c.name} <small>· {c.detail}</small>
            </button>
          ))}
          <button onClick={() => (onAddArticle(q), done())}>
            “{q.trim()}” <small>· Wikipedia</small>
          </button>
        </div>
      )}
    </div>
  );
}
