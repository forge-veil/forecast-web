import { useEffect, useRef, useState } from "react";
import type { Forecast } from "timesfm-web";
import type { Item } from "src/lib/items";
import type { SeriesData } from "src/lib/sources";

export interface Entry {
  data?: SeriesData;
  error?: string;
}

/** Loads every item's series once and keeps the results by id. */
export function useSeriesData(items: Item[]) {
  const [map, setMap] = useState<Record<string, Entry>>({});
  const started = useRef(new Set<string>());

  useEffect(() => {
    for (const item of items) {
      if (!item.load || started.current.has(item.id)) continue;
      started.current.add(item.id);
      item.load().then(
        (data) => setMap((m) => ({ ...m, [item.id]: { data } })),
        (e: unknown) => setMap((m) => ({ ...m, [item.id]: { error: e instanceof Error ? e.message : String(e) } })),
      );
    }
  }, [items]);

  return map;
}

export const FULL_HORIZON = 128;
const sig = (v: number[]) => `${v.length}:${v[0]}:${v[v.length - 1]}`;

/**
 * Forecasts every loaded series 128 steps ahead in one batched call, so the
 * watchlist and the forecast view share results. Re-runs only for series
 * that are new or changed.
 */
export function useWatchlistForecasts(
  ready: boolean,
  forecast: (series: number[][], opts: { horizon: number }) => Promise<Forecast[]>,
  series: Record<string, number[] | undefined>,
) {
  const [map, setMap] = useState<Record<string, { sig: string; f: Forecast }>>({});
  const inflight = useRef(new Set<string>());

  useEffect(() => {
    if (!ready) return;
    const todo = Object.entries(series).filter(
      ([id, v]) => v && v.length >= 3 && map[id]?.sig !== sig(v) && !inflight.current.has(`${id}|${sig(v)}`),
    ) as [string, number[]][];
    if (todo.length === 0) return;
    todo.forEach(([id, v]) => inflight.current.add(`${id}|${sig(v)}`));
    forecast(
      todo.map(([, v]) => v),
      { horizon: FULL_HORIZON },
    ).then(
      (out) =>
        setMap((m) => {
          const next = { ...m };
          todo.forEach(([id, v], i) => (next[id] = { sig: sig(v), f: out[i] }));
          return next;
        }),
      () => {},
    ).finally(() => todo.forEach(([id, v]) => inflight.current.delete(`${id}|${sig(v)}`)));
  }, [ready, forecast, series, map]);

  return (id: string, values?: number[]) => (values && map[id]?.sig === sig(values) ? map[id].f : undefined);
}
