import { useCallback, useEffect, useState } from "react";
import { getHealth } from "../lib/api";
import type { HealthResponse } from "../lib/types";

export type Health =
  | { state: "checking" }
  | { state: "online"; data: HealthResponse }
  | { state: "offline" };

/** Backend reachability, checked when the workspace opens and on demand. */
export function useHealth(enabled: boolean) {
  const [health, setHealth] = useState<Health>({ state: "checking" });

  const check = useCallback(async (signal?: AbortSignal) => {
    setHealth({ state: "checking" });
    try {
      setHealth({ state: "online", data: await getHealth(signal) });
    } catch {
      if (!signal?.aborted) setHealth({ state: "offline" });
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    const c = new AbortController();
    void check(c.signal);
    return () => c.abort();
  }, [enabled, check]);

  return { health, recheck: () => void check() };
}
