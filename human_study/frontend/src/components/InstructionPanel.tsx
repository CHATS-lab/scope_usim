"use client";

import { Markdown } from "./Markdown";

interface Props {
  instruction: string;
  taskSplit: string;
  taskIdx: number;
  runtimeId: string;
}

/**
 * Labels tau2 uses in its user_scenario text. We promote each to an h4 so the
 * Markdown component styles them as section headings (mt-4 mb-2 text-base
 * font-semibold, per OpenHands' heading scale).
 */
const SECTION_LABELS = [
  "Instruction",
  "Instructions",
  "You may start with",
  "Rules",
  "Finish the Task",
  "Finish the task",
  "Reason for call",
  "Known info",
  "Unknown info",
  "Task instructions",
  "Domain",
];

/**
 * Normalise tau2-bench's tab-indented scenario text into clean markdown.
 *
 * tau2 stores instructions with literal tabs for visual nesting. In markdown
 * 4+ leading spaces means "code block", so a naive conversion would render
 * the whole scenario as a grey monospace block. We strip leading whitespace
 * on every line and promote known labels to h4 headings.
 */
function normaliseScenario(raw: string): string {
  const lines = raw
    .replace(/\r\n?/g, "\n")
    .replace(/\t/g, " ")
    .split("\n")
    .map((line) => line.replace(/^\s+/, ""));

  // Promote label-only lines to h4. A "label line" is one whose content is
  // the label followed by a colon and nothing else -- these are the all-caps
  // anchors that mark each section. Inline colons inside prose (e.g.
  // "like `#W0000000`:") are left alone.
  const labelSet = new Set(SECTION_LABELS.map((l) => l.toLowerCase()));
  const promoted = lines.map((line) => {
    const m = line.match(/^(.+?):\s*$/);
    if (!m) return line;
    const candidate = m[1].trim();
    if (labelSet.has(candidate.toLowerCase())) {
      // Leading blank so the h4 starts a fresh block.
      return `\n#### ${candidate}`;
    }
    return line;
  });

  return promoted.join("\n");
}

export function InstructionPanel({ instruction, taskSplit, taskIdx, runtimeId }: Props) {
  const body = normaliseScenario(instruction);
  return (
    <aside
      aria-label="Task instructions"
      className="flex h-full min-h-0 flex-col overflow-y-auto border-l border-border bg-bg"
    >
      <header className="border-b border-border bg-panel px-6 py-3">
        <h2 className="text-sm font-semibold text-text">Your role</h2>
        <p className="mt-0.5 text-xs text-muted">
          Read the instructions below and chat with the agent on the left. When the
          agent has addressed your request, end the conversation with{" "}
          <code className="rounded bg-panelAlt px-1 py-0.5 font-mono text-text">
            /stop
          </code>
          .
        </p>
      </header>

      <div className="flex-1 px-6 py-6">
        <article className="mx-auto max-w-xl text-text">
          <Markdown>{body}</Markdown>
        </article>
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
