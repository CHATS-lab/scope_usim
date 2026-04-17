"use client";

interface Props {
  instruction: string;
  taskSplit: string;
  taskIdx: number;
  runtimeId: string;
}

export function InstructionPanel({ instruction, taskSplit, taskIdx, runtimeId }: Props) {
  return (
    <aside className="flex h-full flex-col overflow-y-auto border-l border-border bg-bg">
      <div className="flex-1 px-6 py-6">
        <div className="mx-auto max-w-xl space-y-4 text-sm leading-relaxed">
          <div className="prose prose-invert max-w-none whitespace-pre-wrap">
            {instruction}
          </div>

          <hr className="border-border" />

          <section>
            <h3 className="mb-2 font-semibold text-text">Finish the task</h3>
            <ul className="list-disc space-y-1 pl-5 text-muted">
              <li>
                When you believe the agent has finished, send <code className="font-mono text-text">/stop</code> to
                stop the conversation.
              </li>
              <li>A survey will appear. Fill it out based on your experience.</li>
              <li>
                After submitting the survey, you will receive a completion code. Paste it into
                the Prolific form to mark the task as completed.
              </li>
            </ul>
          </section>
        </div>
      </div>

      <footer className="border-t border-border px-6 py-3 text-xs text-muted">
        <div>Task split: {taskSplit}</div>
        <div>Task index: {taskIdx}</div>
        <div className="truncate">Runtime: {runtimeId}</div>
      </footer>
    </aside>
  );
}
