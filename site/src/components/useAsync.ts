import { useEffect, useState } from "react";

/** Runs an async loader whenever `deps` change; results from superseded runs are dropped. */
export function useAsync<T>(fn: (() => Promise<T>) | null, deps: unknown[]) {
  const [state, setState] = useState<{ data: T | null; loading: boolean; error: string | null }>({
    data: null,
    loading: !!fn,
    error: null,
  });
  useEffect(() => {
    if (!fn) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    let live = true;
    setState((s) => ({ data: s.data, loading: true, error: null }));
    fn().then(
      (data) => live && setState({ data, loading: false, error: null }),
      (e: unknown) => live && setState({ data: null, loading: false, error: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}
