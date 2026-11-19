import { ArrowRight } from "lucide-react";
import { HeroSection } from "./HeroSection";
import { HowItWorks } from "./HowItWorks";

interface LandingProps {
  onStart: () => void;
  onHowItWorks: () => void;
}

export function LandingPage({ onStart, onHowItWorks }: LandingProps) {
  return (
    <main>
      <HeroSection onStart={onStart} onHowItWorks={onHowItWorks} />
      <HowItWorks />
      <section className="mx-auto max-w-6xl px-5 pb-24 sm:px-8">
        <div className="panel hud overflow-hidden px-6 py-14 text-center sm:px-12">
          <h2 className="h-section">
            Have something worth checking?
          </h2>
          <p className="mx-auto mt-3 max-w-lg text-muted">
            Submit a claim, a screenshot, a clip or a recording. Nothing is stored — uploads are deleted as soon as the report is ready.
          </p>
          <button
            onClick={onStart}
            className="btn btn-primary btn-lg mt-8"
          >
            Start an Investigation <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </section>
    </main>
  );
}
