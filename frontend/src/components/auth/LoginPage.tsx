import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ArrowRight, Eye, EyeOff, KeyRound, LockKeyhole, ShieldAlert, UserRound } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { signIn, type Session } from "../../lib/auth";
import { MaskRise, ScrambleText } from "../fx/TextFx";
import { BrandMark } from "../visual/BrandMark";
import { Sonar } from "../visual/Sonar";

interface Props {
  onSignedIn: (s: Session) => void;
  onBack: () => void;
}

export function LoginPage({ onSignedIn, onBack }: Props) {
  const reduce = useReducedMotion();
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shake, setShake] = useState(0);
  const [caps, setCaps] = useState(false);
  const [granted, setGranted] = useState(false);
  const userRef = useRef<HTMLInputElement>(null);

  useEffect(() => userRef.current?.focus(), []);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (granted) return;
    if (!user.trim() || !password) {
      setError("Enter your user ID and password.");
      setShake((n) => n + 1);
      return;
    }
    const s = signIn(user, password);
    if (!s) {
      setError("Those credentials don't match. Check the user ID and password and try again.");
      setPassword("");
      setShake((n) => n + 1);
      return;
    }
    setError(null);
    setGranted(true);
    setTimeout(() => onSignedIn(s), reduce ? 0 : 1100);
  };

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => setCaps(e.getModifierState?.("CapsLock") ?? false);

  return (
    <main className="mx-auto grid min-h-[100svh] max-w-6xl items-center gap-10 px-5 pb-16 pt-28 sm:px-8 lg:grid-cols-[1fr_minmax(0,440px)] lg:gap-16">
      {/* left: identity */}
      <div className="relative hidden lg:block">
        <p className="eyebrow flex items-center gap-3"><span className="h-px w-10 bg-teal/60" /><span className="text-shimmer">Restricted access</span></p>
        <h1 className="h-page mt-6">
          <MaskRise delay={0.1}>Enter the</MaskRise>
          <MaskRise delay={0.22} className="italic text-teal"><ScrambleText text="investigation lab." delay={300} /></MaskRise>
        </h1>
        <p className="mt-6 max-w-md leading-relaxed text-muted">
          Sign in to open the workspace and investigate text, images, video and audio.
        </p>
        <div className="relative mt-12 w-[min(360px,80%)] opacity-80">
          <Sonar />
        </div>
      </div>

      {/* right: form */}
      <motion.div
        key={shake}
        initial={shake ? { x: 0 } : { opacity: 0, y: 20 }}
        animate={shake && !reduce ? { x: [0, -10, 9, -6, 4, 0] } : { opacity: 1, y: 0 }}
        transition={shake ? { duration: 0.45 } : { duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        className="panel hud mx-auto w-full max-w-md p-7 sm:p-9"
      >
        <div className="flex items-center gap-3">
          <span className="grid h-11 w-11 place-items-center rounded-2xl border border-teal/30 bg-teal/10"><BrandMark className="h-6 w-6" /></span>
          <div>
            <p className="font-display text-xl">FactCheck</p>
            <p className="eyebrow !text-micro">Secure sign-in</p>
          </div>
        </div>

        <AnimatePresence mode="wait">
          {granted ? (
            <motion.div key="ok" initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}
                        className="mt-10 flex flex-col items-center py-8 text-center" role="status">
              <span className="grid h-16 w-16 place-items-center rounded-full border border-teal/40 bg-teal/15 text-teal hairline-glow">
                <KeyRound className="h-7 w-7" />
              </span>
              <p className="h-card mt-5"><ScrambleText text="Access granted" duration={700} /></p>
              <p className="mt-2 text-sm text-muted">Opening the workspace…</p>
            </motion.div>
          ) : (
            <motion.form key="form" onSubmit={submit} className="mt-8 space-y-5" noValidate exit={{ opacity: 0 }}>
              <div>
                <label htmlFor="login-user" className="eyebrow">User ID</label>
                <div className="relative mt-2">
                  <UserRound className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
                  <input id="login-user" ref={userRef} value={user} onChange={(e) => setUser(e.target.value)}
                         autoComplete="username" spellCheck={false} className="field !pl-10" placeholder="Enter your user ID" />
                </div>
              </div>
              <div>
                <label htmlFor="login-password" className="eyebrow">Password</label>
                <div className="relative mt-2">
                  <LockKeyhole className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
                  <input id="login-password" type={show ? "text" : "password"} value={password}
                         onChange={(e) => setPassword(e.target.value)} onKeyUp={onKey} onKeyDown={onKey}
                         autoComplete="current-password" className="field !pl-10 !pr-11" placeholder="Enter your password" />
                  <button type="button" onClick={() => setShow((s) => !s)} aria-label={show ? "Hide password" : "Show password"}
                          className="absolute right-2 top-1/2 grid h-8 w-8 -translate-y-1/2 place-items-center rounded-xl text-muted hover:text-ice">
                    {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
                {caps && <p className="mt-1.5 text-xs text-warn">Caps Lock is on.</p>}
              </div>

              <AnimatePresence>
                {error && (
                  <motion.p initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
                            className="flex items-start gap-2 overflow-hidden rounded-xl border border-refuted/30 bg-refuted/[0.08] px-3 py-2.5 text-sm text-ice/90" role="alert">
                    <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-refuted" /> {error}
                  </motion.p>
                )}
              </AnimatePresence>

              <button type="submit" className="group btn btn-primary btn-lg w-full">
                Sign in <ArrowRight className="relative h-4 w-4 transition-transform group-hover:translate-x-1" />
              </button>
              <button type="button" onClick={onBack} className="btn btn-ghost w-full">Back to home</button>
            </motion.form>
          )}
        </AnimatePresence>
      </motion.div>
    </main>
  );
}
