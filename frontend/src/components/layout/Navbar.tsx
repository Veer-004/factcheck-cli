import { GitBranch, LogIn, LogOut, ScanSearch, UserRound } from "lucide-react";
import type { Route } from "../../hooks/useRoute";
import { GITHUB_URL } from "../../lib/config";
import { BrandMark } from "../visual/BrandMark";

interface NavbarProps {
  route: Route;
  navigate: (to: Route, anchor?: string) => void;
  busy: boolean;
  user: string | null;
  onSignOut: () => void;
}

export function Navbar({ route, navigate, busy, user, onSignOut }: NavbarProps) {
  const link = "rounded-full px-3 py-1.5 text-sm text-muted transition-colors hover:text-ice";
  return (
    <header className="fixed inset-x-0 top-0 z-40 px-3 sm:px-5">
      <nav className="glass mx-auto mt-3 flex max-w-6xl items-center justify-between gap-2 rounded-2xl px-3 py-2 sm:mt-4 sm:px-4">
        <button onClick={() => navigate("home")} className="flex items-center gap-2.5 rounded-xl pr-2" aria-label="FactCheck home">
          <BrandMark />
          <span className="font-display text-lg font-semibold tracking-tight">FactCheck</span>
        </button>
        <div className="flex items-center gap-1 sm:gap-2">
          <button onClick={() => navigate("home")} className={`${link} hidden sm:inline-flex ${route === "home" ? "text-ice" : ""}`}>
            Home
          </button>
          <button onClick={() => navigate("home", "how-it-works")} className={`${link} hidden md:inline-flex`}>
            How it works
          </button>
          <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" className={`${link} inline-flex items-center gap-1.5`} aria-label="Source code on GitHub">
            <GitBranch className="h-4 w-4" />
            <span className="hidden lg:inline">GitHub</span>
          </a>
          <button
            onClick={() => navigate("investigate")}
            className={`btn btn-sm !text-sm ${route === "investigate" ? "btn-glass hairline-glow" : "btn-glass !text-teal"}`}
          >
            <ScanSearch className="h-4 w-4" />
            <span className="hidden sm:inline">Investigate</span>
            {busy && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-teal" aria-label="investigation running" />}
          </button>
          {user ? (
            <div className="flex items-center gap-1">
              <span className="chip hidden md:inline-flex"><UserRound className="h-3 w-3 text-teal" />{user}</span>
              <button onClick={onSignOut} className="btn btn-ghost btn-sm !px-2.5" aria-label="Sign out" title="Sign out">
                <LogOut className="h-4 w-4" />
              </button>
            </div>
          ) : (
            route !== "login" && (
              <button onClick={() => navigate("login")} className="btn btn-ghost btn-sm !px-2.5 sm:!px-3" aria-label="Sign in">
                <LogIn className="h-4 w-4" /><span className="hidden lg:inline">Sign in</span>
              </button>
            )
          )}
        </div>
      </nav>
    </header>
  );
}
