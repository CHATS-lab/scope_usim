"use client";

import { useState } from "react";
import { Check, Link2, PanelRight, PanelRightClose } from "lucide-react";
import { copyToClipboard } from "@/lib/copy";
import { StatusPill, type SessionPhase } from "./StatusPill";

interface Props {
  phase: SessionPhase;
  turnCount: number;
  maxTurns: number;
  onToggleInstructions?: () => void;
  instructionsOpen?: boolean;
}

export function TopBar({
  phase,
  turnCount,
  maxTurns,
  onToggleInstructions,
  instructionsOpen,
}: Props) {
  const [copied, setCopied] = useState(false);

  async function handleCopyLink() {
    if (typeof window === "undefined") return;
    const ok = await copyToClipboard(window.location.href);
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    }
  }

  return (
    <header className="flex items-center justify-between border-b border-border bg-panel px-4 py-2 md:px-6">
      <div className="flex items-center gap-3">
        <StatusPill phase={phase} />
        <span className="hidden text-xs tabular-nums text-muted md:inline">
          Turn {turnCount} / {maxTurns}
        </span>
      </div>

      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={handleCopyLink}
          aria-label={copied ? "Link copied" : "Copy shareable link"}
          title="Copy shareable link"
          className="flex h-8 w-8 items-center justify-center rounded-md text-muted hover:bg-panelAlt hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          {copied ? (
            <Check className="h-4 w-4 text-emerald-400" aria-hidden="true" />
          ) : (
            <Link2 className="h-4 w-4" aria-hidden="true" />
          )}
        </button>

        {onToggleInstructions && (
          <button
            type="button"
            onClick={onToggleInstructions}
            aria-label={instructionsOpen ? "Hide instructions" : "Show instructions"}
            aria-expanded={instructionsOpen}
            title={instructionsOpen ? "Hide instructions" : "Show instructions"}
            className="flex h-8 w-8 items-center justify-center rounded-md text-muted hover:bg-panelAlt hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            {instructionsOpen ? (
              <PanelRightClose className="h-4 w-4" aria-hidden="true" />
            ) : (
              <PanelRight className="h-4 w-4" aria-hidden="true" />
            )}
          </button>
        )}
      </div>
    </header>
  );
}
