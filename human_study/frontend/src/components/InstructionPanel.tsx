"use client";

import { Markdown } from "./Markdown";

interface Props {
  instruction: string;
  taskSplit: string;
  taskIdx: number;
  runtimeId: string;
}

/**
 * Normalise tau2-bench's tab-indented scenario text into clean markdown so
 * section headers ("Instruction:", "Rules:", "Finish the Task:") render as
 * bold lines with proper spacing, matching the Sim2Real screenshot.
 */
function normaliseScenario(raw: string): string {
  let txt = raw.replace(/\r\n?/g, "\n").replace(/\t/g, "  ");
  // Bold the common top-level labels when they appear on their own line.
  const boldLabels = [
    "Instruction:",
    "Instructions:",
    "You may start with:",
    "Rules:",
    "Finish the Task:",
    "Finish the task:",
    "Task split:",
    "Task index:",
    "Runtime:",
    "Reason for call:",
    "Known info:",
    "Unknown info:",
    "Task instructions:",
    "Domain:",
  ];
  for (const label of boldLabels) {
    // Match label at start of line (possibly after spaces) -- bold it by prefixing **...**
    const re = new RegExp(`^(\\s*)(${label.replace(":", ":")})`, "gm");
    txt = txt.replace(re, (_m, pre, lab) => `${pre}**${lab}**`);
  }
  // Turn leading "- " hyphens into proper markdown bullets (they already are,
  // but ensure there's a blank line before the first bullet for GFM lists).
  txt = txt.replace(/(^|\n)(\S[^\n]*:?)\n(- )/g, (_m, l, line, bullet) => `${l}${line}\n\n${bullet}`);
  return txt;
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
        <article className="mx-auto max-w-xl space-y-2 text-[13px] leading-7">
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
