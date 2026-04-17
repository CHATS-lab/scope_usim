"use client";

import { useState } from "react";
import { ChevronDown, Check, Code2, X } from "lucide-react";
import { cn } from "@/lib/cn";

interface Props {
  name: string;
  argsJson: string;
  result: string | null;
}

export function ToolCallCard({ name, argsJson, result }: Props) {
  const [open, setOpen] = useState(false);
  const argSummary = summariseArgs(argsJson);
  const resultState = parseResultState(result);

  return (
    <div
      className={cn(
        "rounded-lg border bg-bg",
        resultState === "error" ? "border-red-400/40" : "border-border"
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={`Tool call ${name}${resultState ? `, ${resultState}` : ", pending"}`}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-muted hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        <Code2 className="h-3.5 w-3.5 flex-shrink-0" aria-hidden="true" />
        <span className="flex-shrink-0 font-mono text-text">{name}</span>
        <span className="truncate font-mono text-muted">{argSummary}</span>
        <span className="ml-auto flex items-center gap-1">
          {resultState === "ok" && (
            <Check className="h-3.5 w-3.5 text-emerald-400" aria-hidden="true" />
          )}
          {resultState === "error" && (
            <X className="h-3.5 w-3.5 text-red-400" aria-hidden="true" />
          )}
          {resultState === null && (
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" aria-hidden="true" />
          )}
          <ChevronDown
            className={cn("h-3.5 w-3.5 transition", open && "rotate-180")}
            aria-hidden="true"
          />
        </span>
      </button>
      {open && (
        <div className="border-t border-border px-3 py-2 font-mono text-xs">
          <div className="text-muted">arguments</div>
          <pre className="mb-2 overflow-x-auto whitespace-pre-wrap break-all">
            {prettyJson(argsJson)}
          </pre>
          {result !== null && (
            <>
              <div className="text-muted">result</div>
              <pre className="overflow-x-auto whitespace-pre-wrap break-all">
                {prettyJson(result)}
              </pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function summariseArgs(raw: string): string {
  try {
    const obj = JSON.parse(raw) as Record<string, unknown>;
    const first = Object.entries(obj)[0];
    if (!first) return "";
    const [k, v] = first;
    return `${k}: ${String(v).slice(0, 40)}`;
  } catch {
    return raw.slice(0, 40);
  }
}

function prettyJson(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}

function parseResultState(raw: string | null): "ok" | "error" | null {
  if (raw === null) return null;
  try {
    const obj = JSON.parse(raw) as { ok?: boolean };
    if (obj && typeof obj === "object" && "ok" in obj) {
      return obj.ok ? "ok" : "error";
    }
  } catch {
    /* fall through */
  }
  return "ok";
}
