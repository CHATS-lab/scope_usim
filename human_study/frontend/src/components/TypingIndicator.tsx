"use client";

import { cn } from "@/lib/cn";

export function TypingIndicator({ className }: { className?: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-label="Agent is typing"
      className={cn(
        "inline-flex items-center gap-2 rounded-full bg-panelAlt px-3 py-1.5 text-xs text-muted",
        className
      )}
    >
      <span className="sr-only">Agent is thinking</span>
      <span className="flex gap-1" aria-hidden="true">
        <span className="inline-block h-1.5 w-1.5 animate-[bounce_1s_ease-in-out_infinite] rounded-full bg-muted" style={{ animationDelay: "0ms" }} />
        <span className="inline-block h-1.5 w-1.5 animate-[bounce_1s_ease-in-out_infinite] rounded-full bg-muted" style={{ animationDelay: "150ms" }} />
        <span className="inline-block h-1.5 w-1.5 animate-[bounce_1s_ease-in-out_infinite] rounded-full bg-muted" style={{ animationDelay: "300ms" }} />
      </span>
      <span>Agent is thinking</span>
    </div>
  );
}
