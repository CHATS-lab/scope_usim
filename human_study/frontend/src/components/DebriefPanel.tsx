"use client";

import { useState } from "react";
import {
  CheckCircle2,
  Copy,
  Check,
  XCircle,
  CircleDashed,
  AlertCircle,
} from "lucide-react";
import type { DebriefInfo, TaskOutcome } from "@/lib/api";
import { copyToClipboard } from "@/lib/copy";
import { cn } from "@/lib/cn";

interface Props {
  completionCode: string;
  debrief: DebriefInfo;
}

export function DebriefPanel({ completionCode, debrief }: Props) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    const ok = await copyToClipboard(completionCode);
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-10">
      <article className="w-full max-w-xl space-y-6 rounded-xl border border-border bg-panel p-8 shadow-xl">
        <header className="flex items-center gap-3">
          <CheckCircle2 className="h-7 w-7 text-emerald-400" aria-hidden="true" />
          <div>
            <h1 className="text-xl font-semibold text-text">Thank you for participating!</h1>
            <p className="text-sm text-muted">
              Your session is complete and your survey responses have been recorded.
            </p>
          </div>
        </header>

        <section
          aria-labelledby="completion-code-heading"
          className="rounded-lg border border-accent/30 bg-accent/5 p-4"
        >
          <h2
            id="completion-code-heading"
            className="text-sm font-semibold text-text"
          >
            Your Prolific completion code
          </h2>
          <p className="mt-1 text-xs text-muted">
            Paste this code into the Prolific submission page to mark the task as
            completed. Copy it before closing this tab.
          </p>
          <div className="mt-3 flex items-center gap-2">
            <code className="flex-1 rounded bg-panelAlt px-3 py-2 font-mono text-sm text-text">
              {completionCode}
            </code>
            <button
              type="button"
              onClick={handleCopy}
              aria-label={copied ? "Copied" : "Copy completion code"}
              className="inline-flex h-9 items-center gap-1 rounded-lg bg-accent px-3 text-sm font-medium text-bg hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              {copied ? (
                <>
                  <Check className="h-3.5 w-3.5" aria-hidden="true" /> Copied
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5" aria-hidden="true" /> Copy
                </>
              )}
            </button>
          </div>
        </section>

        <OutcomeCard outcome={debrief.task_outcome} />

        <section aria-labelledby="debrief-heading" className="space-y-3">
          <h2 id="debrief-heading" className="text-sm font-semibold text-text">
            About the agent you spoke with
          </h2>
          <p className="text-sm text-muted">
            During the study we asked you not to try to guess which agent you were
            talking to, so comparisons across participants would be fair. Now that
            your session is finished, here is what you interacted with.
          </p>

          <div className="rounded-lg border border-border bg-bg/60 p-4">
            <div className="text-sm font-semibold text-text">
              {debrief.condition_label}
            </div>
            <p className="mt-1 text-sm leading-relaxed text-muted">
              {debrief.condition_description}
            </p>
          </div>

          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted sm:grid-cols-4">
            <div>
              <dt className="text-text">Task type</dt>
              <dd>{debrief.task_type}</dd>
            </div>
            <div>
              <dt className="text-text">Split</dt>
              <dd>{debrief.task_split}</dd>
            </div>
            <div>
              <dt className="text-text">Task index</dt>
              <dd>{debrief.task_idx}</dd>
            </div>
            <div>
              <dt className="text-text">Turns taken</dt>
              <dd>{debrief.turn_count}</dd>
            </div>
          </dl>
        </section>

        <footer className="border-t border-border pt-4 text-xs leading-relaxed text-muted">
          <p>
            You can close this tab once you have pasted the completion code into
            Prolific. If you have questions or concerns about the study, please
            contact the research team through the Prolific message system.
          </p>
        </footer>
      </article>
    </main>
  );
}

function OutcomeCard({ outcome }: { outcome: TaskOutcome }) {
  const config = OUTCOME_STYLES[outcome.status];
  const Icon = config.icon;
  return (
    <section
      aria-labelledby="outcome-heading"
      className={cn(
        "rounded-xl border-2 p-5 shadow-lg",
        config.wrap
      )}
    >
      <div className="flex items-start gap-4">
        <div
          className={cn(
            "flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full",
            config.icon_bg
          )}
        >
          <Icon
            className={cn("h-7 w-7", config.icon_color)}
            aria-hidden="true"
            strokeWidth={2.5}
          />
        </div>
        <div className="flex-1">
          <div
            className={cn(
              "text-[11px] font-semibold uppercase tracking-wider",
              config.tag_color
            )}
          >
            {config.tag}
          </div>
          <h2
            id="outcome-heading"
            className={cn("mt-0.5 text-lg font-semibold", config.title_color)}
          >
            {outcome.label}
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-muted">{outcome.detail}</p>
        </div>
      </div>
    </section>
  );
}

const OUTCOME_STYLES: Record<
  TaskOutcome["status"],
  {
    icon: typeof CheckCircle2;
    wrap: string;
    icon_color: string;
    icon_bg: string;
    title_color: string;
    tag: string;
    tag_color: string;
  }
> = {
  success: {
    icon: CheckCircle2,
    wrap: "border-emerald-400/60 bg-emerald-400/10",
    icon_color: "text-emerald-300",
    icon_bg: "bg-emerald-400/20",
    title_color: "text-emerald-50",
    tag: "Task outcome · Success",
    tag_color: "text-emerald-300",
  },
  partial: {
    icon: CircleDashed,
    wrap: "border-amber-400/60 bg-amber-400/10",
    icon_color: "text-amber-300",
    icon_bg: "bg-amber-400/20",
    title_color: "text-amber-50",
    tag: "Task outcome · Partial",
    tag_color: "text-amber-300",
  },
  failure: {
    icon: XCircle,
    wrap: "border-red-400/60 bg-red-400/10",
    icon_color: "text-red-300",
    icon_bg: "bg-red-400/20",
    title_color: "text-red-50",
    tag: "Task outcome · Not completed",
    tag_color: "text-red-300",
  },
  not_evaluated: {
    icon: AlertCircle,
    wrap: "border-border bg-panelAlt",
    icon_color: "text-muted",
    icon_bg: "bg-border/60",
    title_color: "text-text",
    tag: "Task outcome · Pending",
    tag_color: "text-muted",
  },
};
