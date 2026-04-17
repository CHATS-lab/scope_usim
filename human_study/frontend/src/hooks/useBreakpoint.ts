"use client";

import { useEffect, useState } from "react";

/** Returns true once the viewport matches the given min-width media query. */
export function useBreakpoint(minWidthPx: number = 768): boolean {
  const [match, setMatch] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia(`(min-width: ${minWidthPx}px)`);
    const update = () => setMatch(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, [minWidthPx]);
  return match;
}
