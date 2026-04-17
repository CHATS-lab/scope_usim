"use client";

import { useEffect } from "react";
import TextareaAutosize from "react-textarea-autosize";
import { ArrowUp, Square } from "lucide-react";
import { cn } from "@/lib/cn";
import { useBreakpoint } from "@/hooks/useBreakpoint";

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onRequestStop?: () => void;
  disabled: boolean;
  placeholder?: string;
}

export function InputBar({
  value,
  onChange,
  onSubmit,
  onRequestStop,
  disabled,
  placeholder = "Ask anything",
}: Props) {
  const isDesktop = useBreakpoint(768);
  const canSend = !disabled && value.trim().length > 0;

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        if (canSend) onSubmit();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canSend, onSubmit]);

  return (
    <div className="border-t border-border bg-panel px-4 py-3 md:px-8 md:py-4">
      <form
        className="mx-auto max-w-3xl"
        onSubmit={(e) => {
          e.preventDefault();
          if (canSend) onSubmit();
        }}
      >
        <div
          className={cn(
            "group flex items-end gap-2 rounded-3xl border bg-panelAlt px-4 py-2 transition",
            "border-border focus-within:border-accent/70 focus-within:ring-1 focus-within:ring-accent/40"
          )}
        >
          <TextareaAutosize
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={disabled ? "Conversation ended." : placeholder}
            aria-label="Your message"
            disabled={disabled}
            minRows={1}
            maxRows={8}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && isDesktop) {
                e.preventDefault();
                if (canSend) onSubmit();
              }
            }}
            className="flex-1 resize-none bg-transparent py-1.5 text-sm leading-relaxed text-text outline-none placeholder:text-muted/70 disabled:opacity-60"
          />

          {onRequestStop && (
            <button
              type="button"
              onClick={onRequestStop}
              disabled={disabled}
              aria-label="End conversation and go to survey"
              title="End conversation"
              className="mb-0.5 hidden h-8 w-8 flex-shrink-0 items-center justify-center rounded-full text-muted hover:bg-border hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-50 md:inline-flex"
            >
              <Square className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          )}

          <button
            type="submit"
            disabled={!canSend}
            aria-label="Send message"
            className={cn(
              "mb-0.5 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full transition",
              canSend
                ? "bg-accent text-bg hover:opacity-90"
                : "bg-border text-muted/60",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            )}
          >
            <ArrowUp className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
      </form>

      <div className="mx-auto mt-1.5 flex max-w-3xl items-center justify-between px-4 text-[11px] text-muted">
        <div>
          {isDesktop ? (
            <>
              <kbd className="rounded bg-panelAlt px-1 py-[1px]">Enter</kbd> to send,
              {" "}
              <kbd className="rounded bg-panelAlt px-1 py-[1px]">Shift</kbd>+
              <kbd className="rounded bg-panelAlt px-1 py-[1px]">Enter</kbd> for a new
              line, or send <code className="rounded bg-panelAlt px-1">/stop</code> to
              finish.
            </>
          ) : (
            <>Tap the arrow to send. Send <code className="rounded bg-panelAlt px-1">/stop</code> to finish.</>
          )}
        </div>
      </div>
    </div>
  );
}
