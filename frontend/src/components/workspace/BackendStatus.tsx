import { RefreshCw } from "lucide-react";
import type { Health } from "../../hooks/useHealth";
import { API_BASE_URL } from "../../lib/config";

export function BackendStatus({ health, onRecheck }: { health: Health; onRecheck: () => void }) {
  let dot = "bg-muted";
  let text = "Checking the server…";
  if (health.state === "offline") {
    dot = "bg-refuted";
    text = "Server offline";
  } else if (health.state === "online") {
    dot = health.data.fact_check_configured ? "bg-supported" : "bg-warn";
    text = health.data.fact_check_configured ? "Server online" : "Server online · API keys missing";
  }
  return (
    <div className="chip glass shrink-0 self-start !px-3 !py-1.5 !text-xs sm:self-auto" title={API_BASE_URL}>
      <span className={`h-2 w-2 rounded-full ${dot} ${health.state === "checking" ? "animate-pulse" : ""}`} />
      {text}
      {health.state !== "checking" && (
        <button onClick={onRecheck} className="ml-1 hover:text-ice" aria-label="Check the server again">
          <RefreshCw className="h-3 w-3" />
        </button>
      )}
    </div>
  );
}
