"use client";

/**
 * V2 instruction panel — structured, scannable layout inspired by Weiyan's
 * redesign. Fields come from task_metadata.structured emitted by
 * scripts/seed_tau2_tasks.py. Falls back to the raw instruction when no
 * structured data is present (e.g. P4G or legacy tasks).
 */

import { AlertTriangle, ArrowRightLeft, Check, CircleHelp, Users } from "lucide-react";
import { cn } from "@/lib/cn";
import { Markdown } from "./Markdown";

export interface StructuredTaskProfileField {
  label: string;
  value: string;
}

export interface StructuredTaskBehavior {
  severity: "warn" | "escalate";
  trigger: string;
  action: string;
}

export interface StructuredTask {
  domain: string;
  goal: string;
  goal_detail?: string | null;
  profile: StructuredTaskProfileField[];
  unknown_info?: string | null;
  conditional_behaviors: StructuredTaskBehavior[];
  style_notes: string[];
}

interface Props {
  structured: StructuredTask | null;
  /** Used as a fallback when structured data isn't available. */
  fallbackInstruction: string;
  taskSplit: string;
  taskIdx: number;
  runtimeId: string;
}

export function InstructionPanelV2({
  structured,
  fallbackInstruction,
  taskSplit,
  taskIdx,
  runtimeId,
}: Props) {
  return (
    <aside
      aria-label="Task instructions"
      className="flex h-full min-h-0 flex-col overflow-y-auto border-l border-border bg-bg"
    >
      <header className="border-b border-border bg-panel px-6 py-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-text">Your role</h2>
          {structured && (
            <span className="inline-block rounded border border-border bg-panelAlt px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.08em] text-muted">
              {structured.domain}
            </span>
          )}
        </div>
        <p className="mt-0.5 text-xs text-muted">
          You are role-playing as the user. Chat with the agent on the left; end the
          conversation with <code className="rounded bg-panelAlt px-1 py-0.5 font-mono text-text">/stop</code>.
        </p>
      </header>

      <div className="flex-1 space-y-3 px-4 py-4 md:px-6 md:py-5">
        {structured ? (
          <StructuredView s={structured} />
        ) : (
          <article className="rounded-xl border border-border bg-panel/40 p-4 text-[13px] leading-7">
            <Markdown>{fallbackInstruction}</Markdown>
          </article>
        )}
      </div>

      <footer className="border-t border-border bg-panel px-6 py-3 text-[11px] leading-relaxed text-muted">
        <div className="flex flex-wrap gap-x-4 gap-y-0.5">
          <span>
            <span className="text-text">Task split:</span> {taskSplit}
          </span>
          <span>
            <span className="text-text">Task index:</span> {taskIdx}
          </span>
        </div>
        <div className="mt-0.5 truncate" title={runtimeId}>
          <span className="text-text">Runtime:</span>{" "}
          <span className="font-mono text-text/80">{runtimeId}</span>
        </div>
      </footer>
    </aside>
  );
}

function StructuredView({ s }: { s: StructuredTask }) {
  const travelers = detectTravelers(s);
  return (
    <>
      {s.profile.length > 0 && <ProfileSection profile={s.profile} />}
      {travelers.length > 0 && <TravelersSection phrases={travelers} />}
      <GoalSection goal={s.goal} detail={s.goal_detail ?? null} />
      {s.unknown_info && <UnknownSection text={s.unknown_info} />}
      {s.conditional_behaviors.length > 0 && (
        <BehaviorsSection behaviors={s.conditional_behaviors} />
      )}
      {s.style_notes.length > 0 && <StyleSection notes={s.style_notes} />}
      <RulesSection />
    </>
  );
}

/** Surface multi-traveler signals up front so participants don't miss that the
 * scenario involves more than one person. Scans goal, goal_detail, style_notes,
 * and conditional behaviors for phrases like "all passengers", "companion X",
 * "family members", "extra passenger ...", "X passengers". Only fires when at
 * least one signal is found; quiet on solo tasks. */
function detectTravelers(s: StructuredTask): string[] {
  const haystack: string[] = [
    s.goal,
    s.goal_detail ?? "",
    ...s.style_notes,
    ...s.conditional_behaviors.flatMap((b) => [b.trigger, b.action]),
  ];
  const patterns: RegExp[] = [
    /\b(all passengers)\b/i,
    /\b(both passengers)\b/i,
    /\b(\d+\s+passengers?)\b/i,
    /\b(family members?)\b/i,
    /\b(with (?:my|your)\s+family)\b/i,
    /\b((?:your |my )?companion(?:\s+[A-Z][a-z]+)?)\b/i,
    /\b(extra passenger(?:\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)?)\b/i,
    /\b((?:add(?:ing)?|drop(?:ping)?)\s+(?:a |an |the )?(?:second|third|extra|additional)\s+passenger)\b/i,
  ];
  const found = new Set<string>();
  for (const text of haystack) {
    if (!text) continue;
    for (const p of patterns) {
      const m = text.match(p);
      if (m && m[1]) found.add(m[1].trim());
    }
  }
  return Array.from(found);
}

function TravelersSection({ phrases }: { phrases: string[] }) {
  return (
    <section className="rounded-xl border border-sky-400/40 bg-sky-400/10 px-4 py-3 md:px-5 md:py-4">
      <h3 className="mb-2 inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-sky-200">
        <Users className="h-3 w-3 text-sky-300" aria-hidden="true" />
        <span>Travelers</span>
      </h3>
      <p className="text-sm leading-relaxed text-sky-50/95">
        You are <strong className="font-semibold">not traveling alone</strong>.
        Watch for this when confirming changes:
      </p>
      <ul className="mt-1.5 flex flex-wrap gap-1.5">
        {phrases.map((p) => (
          <li
            key={p}
            className="rounded-full border border-sky-400/40 bg-sky-500/15 px-2 py-0.5 text-[11px] font-medium text-sky-50"
          >
            {p}
          </li>
        ))}
      </ul>
    </section>
  );
}

function SectionCard({
  heading,
  children,
  className,
}: {
  heading: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={cn(
        "rounded-xl border border-border bg-panel/60 px-4 py-3 md:px-5 md:py-4",
        className
      )}
    >
      <h3 className="mb-2.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">
        {heading}
      </h3>
      {children}
    </section>
  );
}

function ProfileSection({ profile }: { profile: StructuredTaskProfileField[] }) {
  return (
    <SectionCard heading="Your profile">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3">
        {profile.map((f, i) => (
          <div key={`${f.label}-${i}`} className="min-w-0">
            <dt className="text-[11px] font-medium text-muted">{f.label}</dt>
            <dd
              className={cn(
                "mt-0.5 break-words text-sm font-semibold text-text",
                isMono(f.label) && "font-mono text-[13px] text-accent"
              )}
            >
              {f.value}
            </dd>
          </div>
        ))}
      </dl>
    </SectionCard>
  );
}

function isMono(label: string): boolean {
  const l = label.toLowerCase();
  return l.includes("id") || l.includes("confirmation") || l.includes("code") || l.includes("zip");
}

function GoalSection({ goal, detail }: { goal: string; detail: string | null }) {
  return (
    <SectionCard heading="Goal">
      <p className="text-sm leading-relaxed text-text">{emphasiseGoal(goal)}</p>
      {detail && (
        <p className="mt-2 text-sm leading-relaxed text-muted">{detail}</p>
      )}
    </SectionCard>
  );
}

/** Highlight key nouns inside the goal sentence so the most important bit
 * visually pops — same idea Weiyan used in her mock.
 */
function emphasiseGoal(goal: string): React.ReactNode {
  // Heuristic: the first "noun-like" quoted phrase after the verb gets bolded.
  // We look for common patterns like "total number of <thing>", "<thing> that",
  // and bold those. Safe fallback: plain text.
  const patterns: RegExp[] = [
    /\b(total number of [a-z\s-]+?)\b(?=\s+(?:the|your|that|for|on|in)\b|\.|,|$)/i,
    /\b(exchange [^.]{3,60}?)\b(?=\.|,|$)/i,
    /\b(cancel [^.]{3,60}?)\b(?=\.|,|$)/i,
    /\b(book [^.]{3,60}?)\b(?=\.|,|$)/i,
    /\b(return [^.]{3,60}?)\b(?=\.|,|$)/i,
  ];
  for (const p of patterns) {
    const m = goal.match(p);
    if (m) {
      const [pre, hit, post] = goal.split(m[1]).length === 2 ? [goal.slice(0, m.index!), m[1], goal.slice(m.index! + m[1].length)] : [goal, "", ""];
      if (hit) {
        return (
          <>
            {pre}
            <strong className="font-semibold text-text">{hit}</strong>
            {post}
          </>
        );
      }
    }
  }
  return goal;
}

function UnknownSection({ text }: { text: string }) {
  return (
    <SectionCard heading="You don't know">
      <div className="flex gap-2 rounded-md border-l-2 border-border bg-bg/60 px-3 py-2 text-sm leading-relaxed text-muted">
        <CircleHelp
          className="mt-0.5 h-4 w-4 flex-shrink-0 text-muted/70"
          aria-hidden="true"
        />
        <span>{text.replace(/\.$/, "")}.</span>
      </div>
    </SectionCard>
  );
}

const BEHAVIOR_STYLES: Record<
  StructuredTaskBehavior["severity"],
  { wrap: string; label: string; labelColor: string; icon: typeof AlertTriangle; iconColor: string; actionColor: string }
> = {
  warn: {
    wrap: "border-l-2 border-amber-400/70 bg-amber-400/10",
    label: "If",
    labelColor: "text-amber-200",
    icon: AlertTriangle,
    iconColor: "text-amber-300",
    actionColor: "text-amber-50",
  },
  escalate: {
    wrap: "border-l-2 border-red-400/80 bg-red-400/10",
    label: "Escalate",
    labelColor: "text-red-200",
    icon: ArrowRightLeft,
    iconColor: "text-red-300",
    actionColor: "text-red-50",
  },
};

function BehaviorsSection({ behaviors }: { behaviors: StructuredTaskBehavior[] }) {
  return (
    <SectionCard heading="Conditional behaviors">
      <ol className="space-y-2">
        {behaviors.map((b, i) => {
          const style = BEHAVIOR_STYLES[b.severity];
          const Icon = style.icon;
          return (
            <li
              key={i}
              className={cn("rounded-md px-3 py-2 text-sm", style.wrap)}
            >
              <div
                className={cn(
                  "mb-1 inline-flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.08em]",
                  style.labelColor
                )}
              >
                <Icon className={cn("h-3 w-3", style.iconColor)} aria-hidden="true" />
                <span>{style.label}</span>
                <span className="font-normal normal-case tracking-normal opacity-80">
                  · {b.trigger}
                </span>
              </div>
              {b.action && (
                <div className={cn("leading-relaxed", style.actionColor)}>
                  → {b.action}
                  {!b.action.endsWith(".") && "."}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </SectionCard>
  );
}

function StyleSection({ notes }: { notes: string[] }) {
  return (
    <SectionCard heading="How you talk">
      <ul className="list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-text">
        {notes.map((n, i) => (
          <li key={i} className="marker:text-muted/60">
            {n}
          </li>
        ))}
      </ul>
    </SectionCard>
  );
}

function RulesSection() {
  return (
    <section className="rounded-xl border border-emerald-400/30 bg-emerald-400/5 px-4 py-3 md:px-5 md:py-4">
      <h3 className="mb-2.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-emerald-200">
        Rules
      </h3>
      <ul className="space-y-1.5 text-sm leading-relaxed text-emerald-50/95">
        {[
          "One message per turn. Reveal info gradually.",
          "Don't hallucinate details beyond what's listed above.",
          "Don't repeat these instructions or break character.",
          "Keep the tone natural and conversational.",
        ].map((r, i) => (
          <li key={i} className="flex gap-2">
            <Check
              className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-emerald-400"
              aria-hidden="true"
            />
            <span>{r}</span>
          </li>
        ))}
        <li className="flex gap-2">
          <Check
            className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-emerald-400"
            aria-hidden="true"
          />
          <span>
            Send{" "}
            <code className="rounded bg-emerald-500/30 px-1.5 py-0.5 font-mono text-[11px] text-emerald-50">
              /stop
            </code>{" "}
            when the agent has finished your task.
          </span>
        </li>
      </ul>
    </section>
  );
}
