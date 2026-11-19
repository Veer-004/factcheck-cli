import "lenis/dist/lenis.css";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useCallback, useEffect, useState } from "react";
import { LoginPage } from "./components/auth/LoginPage";
import { LandingPage } from "./components/landing/LandingPage";
import { Footer } from "./components/layout/Footer";
import { Navbar } from "./components/layout/Navbar";
import { ScrollProgress } from "./components/layout/ScrollProgress";
import { ToastProvider } from "./components/ui/Toast";
import { OceanBackground } from "./components/visual/OceanBackground";
import { InvestigationWorkspace } from "./components/workspace/InvestigationWorkspace";
import { useHealth } from "./hooks/useHealth";
import { useInvestigation, type Inputs } from "./hooks/useInvestigation";
import { useRoute } from "./hooks/useRoute";
import { readSession, signOut, type Session } from "./lib/auth";
import type { Mode } from "./lib/config";
import { startSmoothScroll } from "./lib/smoothScroll";

export default function App() {
  const reduce = useReducedMotion();
  const { route, navigate, onTransitionDone } = useRoute();
  const [session, setSession] = useState<Session | null>(readSession);
  // Workspace state lives here so leaving the workspace never loses input or a running request.
  const [mode, setMode] = useState<Mode>("text");
  const [inputs, setInputs] = useState<Inputs>({
    text: "",
    files: { image: null, video: null, audio: null },
    durations: { image: null, video: null, audio: null },
  });
  const { state, start, stopWaiting, reset } = useInvestigation();
  const { health, recheck } = useHealth(route === "investigate" && !!session);

  useEffect(() => startSmoothScroll(), []);

  // The workspace needs a sign-in; a signed-in user skips the login page.
  useEffect(() => {
    if (route === "investigate" && !session) navigate("login");
    else if (route === "login" && session) navigate("investigate");
  }, [route, session, navigate]);

  // A failed connection makes the status pill stale: check the server again.
  const networkFailed = state.phase === "error" && state.error.kind === "network";
  useEffect(() => {
    if (networkFailed) recheck();
  }, [networkFailed]); // recheck is recreated each render; only the failure matters

  const onSignOut = useCallback(() => {
    stopWaiting();
    reset();
    signOut();
    setSession(null);
    navigate("home");
  }, [navigate, reset, stopWaiting]);

  const view = route === "investigate" && !session ? "login" : route;

  return (
    <ToastProvider>
      <OceanBackground />
      <ScrollProgress />
      <Navbar route={route} navigate={navigate} busy={state.phase === "running"} user={session?.user ?? null} onSignOut={onSignOut} />
      <AnimatePresence mode="wait" onExitComplete={onTransitionDone}>
        <motion.div
          key={view}
          initial={reduce ? { opacity: 0 } : { opacity: 0, y: 18, filter: "blur(8px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          exit={reduce ? { opacity: 0 } : { opacity: 0, y: -12, filter: "blur(6px)" }}
          transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
        >
          {view === "home" && (
            <LandingPage onStart={() => navigate("investigate")} onHowItWorks={() => navigate("home", "how-it-works")} />
          )}
          {view === "login" && (
            <LoginPage onSignedIn={(s) => { setSession(s); navigate("investigate"); }} onBack={() => navigate("home")} />
          )}
          {view === "investigate" && (
            <InvestigationWorkspace
              mode={mode}
              setMode={setMode}
              inputs={inputs}
              setInputs={setInputs}
              investigation={state}
              onVerify={() => void start(state.phase === "error" ? state.mode : mode, inputs)}
              onStopWaiting={stopWaiting}
              onReset={reset}
              health={health}
              onRecheck={recheck}
            />
          )}
        </motion.div>
      </AnimatePresence>
      <Footer />
    </ToastProvider>
  );
}
