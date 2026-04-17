"use client";

import { useState } from "react";
import { ChevronDown, Code2 } from "lucide-react";

interface Props {
  name: string;
  argsJson: string;
  result: string | null;
}

export function ToolCallCard({ name, argsJson, result }: Props) {
  const [open, setOpen] = useState(false);
  const argSummary = summariseArgs(argsJson);
  return (
    <div className="rounded-lg border border-border bg-bg">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-muted hover:text-text"
      >
        <Code2 className="h-3.5 w-3.5" />
        <span className="font-mono text-text">{name}</span>
        <span className="truncate font-mono text-muted">{argSummary}</span>
        <ChevronDown
          className={`ml-auto h-3.5 w-3.5 transition ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open && (
        <div className="border-t border-border px-3 py-2 font-mono text-xs">
          <div className="text-muted">arguments</div>
          <pre className="mb-2 overflow-x-auto whitespace-pre-wrap">
            {prettyJson(argsJson)}
          </pre>
          {result !== null && (
            <>
              <div className="text-muted">result</div>
              <pre className="overflow-x-auto whitespace-pre-wrap">
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
