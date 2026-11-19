// Demo sign-in. Static credentials (no database, no backend call): VITE_DEMO_USER / VITE_DEMO_PASSWORD,
// default admin / admin123. This is a front-end gate for demonstrations only — it is NOT security:
// the values ship in the JavaScript bundle and the API itself does not require a login.

const USER = import.meta.env.VITE_DEMO_USER || "admin";
const PASSWORD = import.meta.env.VITE_DEMO_PASSWORD || "admin123";
const KEY = "factcheck.session";

export interface Session {
  user: string;
  since: number;
}

export function readSession(): Session | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    const s = raw ? (JSON.parse(raw) as Session) : null;
    return s && s.user === USER ? s : null;
  } catch {
    return null; // storage blocked: just not signed in
  }
}

/** Constant-time-ish comparison so a wrong password doesn't return faster for an early mismatch. */
function same(a: string, b: string): boolean {
  let diff = a.length ^ b.length;
  for (let i = 0; i < Math.max(a.length, b.length); i++) diff |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
  return diff === 0;
}

export function signIn(user: string, password: string): Session | null {
  if (!same(user.trim(), USER) || !same(password, PASSWORD)) return null;
  const s = { user: USER, since: Date.now() };
  try {
    sessionStorage.setItem(KEY, JSON.stringify(s)); // ends when the tab is closed
  } catch {
    /* storage blocked: signed in for this page load only */
  }
  return s;
}

export function signOut() {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}
