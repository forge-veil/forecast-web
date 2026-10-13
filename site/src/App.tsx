import { useCallback, useEffect, useMemo, useState } from "react";
import { Detail } from "src/components/Detail";
import { Sidebar } from "src/components/Sidebar";
import { TopLinks } from "src/components/TopLinks";
import { useAsync } from "src/components/useAsync";
import { YourDataEditor } from "src/components/YourDataEditor";
import { articleItem, cityItem, datasetItem, PRESET_ARTICLES, PRESET_CITIES, YOURS, type Item } from "src/lib/items";
import { useModels } from "src/lib/models";
import { loadManifest, type City, type SeriesData } from "src/lib/sources";
import { useSeriesData, useWatchlistForecasts, type Entry } from "src/lib/useWatchlist";

const initialSelection = () => decodeURIComponent(location.hash.slice(1)) || "stock-spy";

export function App() {
  const manifest = useAsync(loadManifest, []);
  const { choice, slot, forecast } = useModels();
  const [extra, setExtra] = useState<Item[]>([]);
  const [selected, setSelected] = useState(initialSelection);
  const [view, setView] = useState<"list" | "detail">(location.hash ? "detail" : "list");
  const [yours, setYours] = useState<SeriesData | null>(null);
  const [yoursSeason, setYoursSeason] = useState<number | undefined>(12);

  const items = useMemo(() => {
    const datasets = (manifest.data?.datasets ?? []).filter((d) => d.available).map(datasetItem);
    return [
      ...datasets.filter((i) => i.group === "stocks"),
      ...PRESET_CITIES.map(cityItem),
      ...PRESET_ARTICLES.map(articleItem),
      ...datasets.filter((i) => i.group === "seasonal"),
      ...extra,
      YOURS,
    ];
  }, [manifest.data, extra]);

  const loaded = useSeriesData(items);
  const data = useMemo<Record<string, Entry>>(() => ({ ...loaded, yours: yours ? { data: yours } : {} }), [loaded, yours]);
  const seriesById = useMemo(
    () => Object.fromEntries(Object.entries(data).map(([id, e]) => [id, e.data?.values])),
    [data],
  );
  const forecastChosen = useCallback(
    (series: number[][], opts: { horizon: number }) => forecast(choice, series, opts),
    [choice, forecast],
  );
  const forecastFor = useWatchlistForecasts(choice, slot(choice).status === "ready", forecastChosen, seriesById);

  useEffect(() => {
    const on = () => setSelected(initialSelection());
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);

  const select = (id: string) => {
    setSelected(id);
    setView("detail");
    history.replaceState(null, "", `#${encodeURIComponent(id)}`);
  };
  const add = (item: Item) => {
    setExtra((x) => (items.some((i) => i.id === item.id) ? x : [...x, item]));
    select(item.id);
  };

  const item = items.find((i) => i.id === selected) ?? items[0];
  const onYours = useCallback((d: SeriesData | null) => setYours(d), []);
  const onSeason = useCallback((s?: number) => setYoursSeason(s), []);

  if (manifest.error && !manifest.data) {
    return <p className="empty error">Could not load the bundled datasets: {manifest.error}</p>;
  }

  return (
    <div className="app" data-view={view}>
      <TopLinks />
      <Sidebar
        items={items}
        data={data}
        forecastFor={forecastFor}
        selected={item?.id ?? ""}
        onSelect={select}
        onAddCity={(c: City) => add(cityItem(c))}
        onAddArticle={(t: string) => add(articleItem(t))}
      />
      <main className="detail">
        {item && (
          <Detail
            key={item.id}
            item={item}
            entry={data[item.id]}
            full={forecastFor(item.id, data[item.id]?.data?.values)}
            season={item.id === "yours" ? yoursSeason : undefined}
            onBack={() => setView("list")}
          >
            {item.id === "yours" && <YourDataEditor onChange={onYours} onSeason={onSeason} />}
          </Detail>
        )}
      </main>
    </div>
  );
}
