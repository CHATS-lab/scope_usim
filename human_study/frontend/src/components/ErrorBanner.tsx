"use client";

import { useState } from "react";
import { AlertTriangle, ChevronDown, X } from "lucide-react";
import { cn } from "@/lib/cn";

const COLLAPSE_AT = 220;

export function ErrorBanner({
  message,
  onDismiss,
  className,
}: {
  message: string;
  onDismiss?: () => void;
  className?: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const isLong = message.length > COLLAPSE_AT;
  const shown = !isLong || expanded ? message : message.slice(0, COLLAPSE_AT) + "…";

  return (
    <div
      role="alert"
      className={cn(
        "flex items-start gap-2 rounded-md border border-red-400/40 bg-red-500/10 px-3 py-2 text-sm text-red-200",
        className
      )}
    >
      <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <div className="font-medium">Something went wrong</div>
        <div className="mt-0.5 break-words text-xs">{shown}</div>
        {isLong && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="mt-1 inline-flex items-center gap-1 text-xs underline hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400"
          >
            <ChevronDown
              className={cn("h-3 w-3 transition", expanded && "rotate-180")}
              aria-hidden="true"
            />
            {expanded ? "Show less" : "View more"}
          </button>
        )}
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss error"
          className="rounded p-0.5 text-red-200/80 hover:text-red-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      )}
    </div>
  );
}
