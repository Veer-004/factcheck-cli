import { useCallback, useEffect, useRef, useState } from "react";
import { scrollToTarget } from "../lib/smoothScroll";

export type Route = "home" | "investigate" | "login";

const HASH: Record<Route, string> = { home: "#/", investigate: "#/investigate", login: "#/login" };

function read(): Route {
  const h = window.location.hash;
  if (h.startsWith("#/investigate")) return "investigate";
  if (h.startsWith("#/login")) return "login";
  return "home";
}

/**
 * Views selected by the URL hash, so the browser back button and deep links work.
 * When the view changes, scrolling waits for the page transition (call `onTransitionDone`):
 * a new page starts at the top, an anchor is reached with Lenis' eased scroll.
 */
export function useRoute() {
  const [route, setRoute] = useState<Route>(read);
  const pending = useRef<string | null>(null);

  useEffect(() => {
    const onHash = () => setRoute(read());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const navigate = useCallback((to: Route, anchor?: string) => {
    if (to === read()) {
      requestAnimationFrame(() => scrollToTarget(anchor ?? null));
      return;
    }
    pending.current = anchor ?? null;
    window.location.hash = HASH[to];
    setRoute(to);
  }, []);

  const onTransitionDone = useCallback(() => {
    const anchor = pending.current;
    pending.current = null;
    requestAnimationFrame(() => {
      scrollToTarget(null, true); // new page: start at the top
      if (anchor) requestAnimationFrame(() => scrollToTarget(anchor));
    });
  }, []);

  return { route, navigate, onTransitionDone };
}
