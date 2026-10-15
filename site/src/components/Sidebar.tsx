import { useEffect, useState } from "react";
import { MODELS, type Forecast } from "forecast-web";
import { AddModel } from "src/components/ModelBar";
import { Sparkline } from "src/components/Sparkline";
import { formatNumber } from "src/lib/format";
import { changeText, GROUPS, type GroupId, type Item } from "src/lib/items";
import { TABLE_SAMPLES } from "src/components/Tables";
import { useModels } from "src/lib/models";
import { searchCities, type City } from "src/lib/sources";
import type { Entry } from "src/lib/useWatchlist";

const TABLE_GROUPS = [
  { id: "real-estate", label: "Real estate", about: "Home sales, predict the price", icon: "M3 10l7-6 7 6M5 9v7h10V9M8 16v-4h4v4" },
  { id: "classic", label: "Classic", about: "Well-known teaching datasets", icon: "M3 4h14v12H3zM3 8h14M3 12h14M8 4v12" },
  { id: "yours", label: "Your data", about: "Paste a CSV", icon: "M4 16l1-4 8-8 3 3-8 8-4 1z" },
] as const;

const today = new Date().toLocaleDateString("en-US", { month: "long", day: "numeric" });

export function Sidebar(props: {
  tab: "forecasts" | "tables";
  onTab: (tab: "forecasts" | "tables") => void;
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
        <div className="seg sb-tabs" role="tablist" aria-label="Section">
          <button role="tab" aria-selected={props.tab === "forecasts"} aria-pressed={props.tab === "forecasts"} onClick={() => props.onTab("forecasts")}>Forecasts</button>
          <button role="tab" aria-selected={props.tab === "tables"} aria-pressed={props.tab === "tables"} onClick={() => props.onTab("tables")}>Tables</button>
        </div>
        <h1 className="sb-title">{props.tab === "tables" ? "Tables" : "Forecasts"}</h1>
        <p className="sb-date">{props.tab === "tables" ? "Predict a column, no training" : today}</p>
        {props.tab === "forecasts" && (
          <>
            <ModelStatus />
            <Search onAddCity={props.onAddCity} onAddArticle={props.onAddArticle} />
          </>
        )}
      </div>
      <nav className="list">
        {props.tab === "forecasts" && GROUPS.map((g) => {
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
        {props.tab === "tables" &&
          TABLE_GROUPS.map((g) => (
            <section key={g.id} aria-label={g.label} className="group-sec g-tables">
              <header className="group">
                <span className={`gicon g-tables-${g.id}`} aria-hidden="true">
                  <svg viewBox="0 0 20 20"><path d={g.icon} fill="none" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
                </span>
                <div>
                  <div className="group-name">{g.label}</div>
                  <div className="group-about">{g.about}</div>
                </div>
              </header>
              <div className="group-card">
                {TABLE_SAMPLES.filter((s) => s.group === g.id).map((s) => (
                  <button key={s.id} className="row" aria-current={s.id === props.selected} onClick={() => props.onSelect(s.id)}>
                    <div style={{ minWidth: 0 }}>
                      <div className="name">{s.name}</div>
                      <div className="sub">{s.subtitle}</div>
                    </div>
                    <span />
                    <div className="right" />
                  </button>
                ))}
              </div>
            </section>
          ))}
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

/** Which model fills the watchlist, or a menu to load one when none is loaded yet. */
function ModelStatus() {
  const { choice, slot } = useModels();
  const { status, backend, progress } = slot(choice);
  const pct = progress?.total ? Math.round((100 * progress.loaded) / progress.total) : 0;
  if (status === "idle" || status === "error")
    return (
      <div className="model">
        {status === "error" && <span>Could not load {MODELS[choice].name}.</span>}
        <AddModel />
      </div>
    );
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
