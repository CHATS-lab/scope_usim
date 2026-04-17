"use client";

import { Markdown } from "./Markdown";
import { StatusPill, type SessionPhase } from "./StatusPill";

interface Props {
  instruction: string;
  taskSplit: string;
  taskIdx: number;
  runtimeId: string;
  phase: SessionPhase;
}

export function InstructionPanel({
  instruction,
  taskSplit,
  taskIdx,
  runtimeId,
  phase,
}: Props) {
  return (
    <aside
      aria-label="Task instructions"
      className="flex h-full min-h-0 flex-col overflow-y-auto border-l border-border bg-bg"
    >
      <header className="flex items-center justify-between border-b border-border bg-panel px-6 py-3">
        <h2 className="text-sm font-semibold text-text">Your role</h2>
        <StatusPill phase={phase} />
      </header>

      <div className="flex-1 px-6 py-6">
        <div className="mx-auto max-w-xl space-y-4 text-sm leading-relaxed">
          <Markdown>{instruction}</Markdown>

          <hr className="border-border" />

          <section>
            <h3 className="mb-2 font-semibold text-text">Finish the task</h3>
            <ul className="list-disc space-y-1 pl-5 text-muted">
              <li>
                When you believe the agent has finished, send{" "}
                <code className="rounded bg-panelAlt px-1.5 py-0.5 font-mono text-text">
                  /stop
                </code>{" "}
                or click the <b>End</b> button to stop the conversation.
              </li>
              <li>A short survey will appear. Fill it out based on your experience.</li>
              <li>
                After submitting, you will receive a completion code. Paste it into the
                Prolific form to mark the task as completed.
              </li>
            </ul>
          </section>
        </div>
      </div>

      <footer className="border-t border-border bg-panel px-6 py-3 text-xs text-muted">
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          <span>Task split: <span className="text-text">{taskSplit}</span></span>
          <span>Task index: <span className="text-text">{taskIdx}</span></span>
        </div>
        <div className="mt-1 truncate" title={runtimeId}>
          Runtime: <span className="font-mono text-text/80">{runtimeId}</span>
        </div>
      </footer>
    </aside>
  );
}
