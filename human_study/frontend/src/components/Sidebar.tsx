"use client";

import { useState } from "react";
import { PanelLeftClose, PanelLeftOpen, MessageSquare, FileText, Info } from "lucide-react";
import { cn } from "@/lib/cn";

interface Props {
  taskType: string;
  taskSplit: string;
  taskIdx: number;
  condition: string;
  /** Open the instruction panel on mobile, where it's hidden by default. */
  onShowInstructions?: () => void;
}

export function Sidebar({ taskType, taskSplit, taskIdx, onShowInstructions }: Props) {
  const [open, setOpen] = useState(false); // default collapsed (icon-only)

  return (
    <aside
      aria-label="Session navigation"
      className={cn(
        "flex h-full flex-col border-r border-border bg-bg transition-[width] duration-200",
        open ? "w-60" : "w-12"
      )}
    >
      <div className="flex items-center justify-between border-b border-border px-2 py-2">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-label={open ? "Collapse sidebar" : "Expand sidebar"}
          aria-expanded={open}
          className="flex h-8 w-8 items-center justify-center rounded-md text-muted hover:bg-panelAlt hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          {open ? (
            <PanelLeftClose className="h-4 w-4" aria-hidden="true" />
          ) : (
            <PanelLeftOpen className="h-4 w-4" aria-hidden="true" />
          )}
        </button>
        {open && (
          <span className="text-xs font-semibold uppercase tracking-wider text-muted">
            USIM Study
          </span>
        )}
      </div>

      <nav className="flex-1 overflow-y-auto px-1 py-2">
        <SidebarItem
          open={open}
          icon={<MessageSquare className="h-4 w-4" aria-hidden="true" />}
          label="This conversation"
          active
        />
        {onShowInstructions && (
          <button
            type="button"
            onClick={onShowInstructions}
            className={cn(
              "mt-1 flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm text-muted hover:bg-panelAlt hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent md:hidden",
              open ? "justify-start" : "justify-center"
            )}
          >
            <FileText className="h-4 w-4" aria-hidden="true" />
            {open && <span>Task instructions</span>}
          </button>
        )}
      </nav>

      {open && (
        <div className="border-t border-border px-3 py-3 text-[11px] leading-relaxed text-muted">
          <div className="mb-1 inline-flex items-center gap-1 text-text">
            <Info className="h-3 w-3" aria-hidden="true" />
            <span className="font-medium">Session</span>
          </div>
          <div>Task type: <span className="text-text">{taskType}</span></div>
          <div>Split: <span className="text-text">{taskSplit}</span></div>
          <div>Index: <span className="text-text">{taskIdx}</span></div>
          {/* We deliberately don't reveal `condition` to the participant to keep
              the study blind. It's shown in small text only for logged-in
              researcher mode (future). */}
        </div>
      )}
    </aside>
  );
}

function SidebarItem({
  open,
  icon,
  label,
  active,
}: {
  open: boolean;
  icon: React.ReactNode;
  label: string;
  active?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm",
        active ? "bg-panelAlt text-text" : "text-muted",
        open ? "justify-start" : "justify-center"
      )}
      title={label}
    >
      {icon}
      {open && <span className="truncate">{label}</span>}
    </div>
  );
}
