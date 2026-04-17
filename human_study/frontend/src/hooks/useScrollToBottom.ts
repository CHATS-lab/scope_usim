"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Auto-sticks the scroll container to the bottom, but only while the user
 * hasn't manually scrolled up. Returns a ref for the container, a state
 * indicating whether the "jump to bottom" pill should be visible, and an
 * imperative `scrollToBottom()` callback for the pill / new-message hooks.
 */
export function useScrollToBottom<T extends HTMLElement>(
  deps: ReadonlyArray<unknown>,
  { threshold = 64 }: { threshold?: number } = {}
) {
  const ref = useRef<T | null>(null);
  const [stuck, setStuck] = useState(true);

  const scrollToBottom = useCallback((smooth = true) => {
    const el = ref.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onScroll = () => {
      const distance = el.scrollHeight - el.clientHeight - el.scrollTop;
      setStuck(distance < threshold);
    };
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, [threshold]);

  useEffect(() => {
    if (stuck) scrollToBottom(true);
  // deps are intentionally spread so callers can pass e.g. [messages, awaiting]
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stuck, scrollToBottom, ...deps]);

  return { ref, stuck, scrollToBottom };
}
