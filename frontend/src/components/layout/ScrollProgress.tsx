import { motion, useScroll, useSpring } from "framer-motion";

/** Thin teal line at the very top showing how far down the page you are. */
export function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 140, damping: 30, mass: 0.4 });
  return (
    <motion.div
      aria-hidden
      className="fixed inset-x-0 top-0 z-50 h-[2px] origin-left bg-[linear-gradient(90deg,rgba(0,213,192,0.2),var(--color-teal),var(--color-ice))] shadow-[0_0_12px_rgba(0,213,192,0.7)]"
      style={{ scaleX }}
    />
  );
}
