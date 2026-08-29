"use client";

import { cn } from "@/lib/cn";

export type SessionPhase = "connecting" | "chatting" | "waiting" | "stopped" | "error";

const STYLES: Record<SessionPhase, { label: string; dot: string; wrap: string }> = {
  connecting: {
    label: "Connecting",
    dot: "bg-amber-400 animate-pulse",
    wrap: "text-amber-200 bg-amber-400/10 border-amber-400/30",
  },
  chatting: {
    label: "Ready",
    dot: "bg-emerald-400",
    wrap: "text-emerald-200 bg-emerald-400/10 border-emerald-400/30",
  },
  waiting: {
    label: "Thinking",
    dot: "bg-accent animate-pulse",
    wrap: "text-accent bg-accent/10 border-accent/30",
  },
  stopped: {
    label: "Conversation ended",
    dot: "bg-muted",
    wrap: "text-muted bg-muted/10 border-muted/30",
  },
  error: {
    label: "Error",
    dot: "bg-red-400 animate-pulse",
    wrap: "text-red-200 bg-red-400/10 border-red-400/40",
  },
};

export function StatusPill({
  phase,
  className,
  message,
}: {
  phase: SessionPhase;
  className?: string;
  message?: string;
}) {
  const s = STYLES[phase];
  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-medium",
        s.wrap,
        className
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", s.dot)} aria-hidden="true" />
      <span>{message ?? s.label}</span>
    </div>
  );
}
