"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const STORAGE_PREFIX = "usim:draft:";

export function useDraftPersistence(sessionId: string | null, debounceMs = 300) {
  const [value, setValue] = useState("");
  const timer = useRef<number | null>(null);

  // Load on mount / when sessionId changes.
  useEffect(() => {
    if (!sessionId) return;
    try {
      const saved = window.localStorage.getItem(STORAGE_PREFIX + sessionId);
      if (saved !== null) setValue(saved);
    } catch {
      /* localStorage may throw in privacy mode; best-effort */
    }
  }, [sessionId]);

  // Debounced save.
  useEffect(() => {
    if (!sessionId) return;
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      try {
        window.localStorage.setItem(STORAGE_PREFIX + sessionId, value);
      } catch {
        /* ignore */
      }
    }, debounceMs);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [value, sessionId, debounceMs]);

  const clear = useCallback(() => {
    setValue("");
    if (sessionId) {
      try {
        window.localStorage.removeItem(STORAGE_PREFIX + sessionId);
      } catch {
        /* ignore */
      }
    }
  }, [sessionId]);

  return { value, setValue, clear };
}
