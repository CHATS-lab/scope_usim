"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import TextareaAutosize from "react-textarea-autosize";
import { ArrowDown, Copy, Check, Square } from "lucide-react";
import type { ChatMessage } from "@/lib/api";
import { cn } from "@/lib/cn";
import { copyToClipboard } from "@/lib/copy";
import { Markdown } from "./Markdown";
import { ToolCallCard } from "./ToolCallCard";
import { TypingIndicator } from "./TypingIndicator";
import { useBreakpoint } from "@/hooks/useBreakpoint";
import { useDraftPersistence } from "@/hooks/useDraftPersistence";
import { useScrollToBottom } from "@/hooks/useScrollToBottom";

interface Props {
  sessionId: string | null;
  messages: ChatMessage[];
  onSend: (text: string) => Promise<void> | void;
  onRequestStop: () => void;
  disabled: boolean;
  awaiting: boolean;
  turnCount: number;
  maxTurns: number;
}

export function ChatPanel({
  sessionId,
  messages,
  onSend,
  onRequestStop,
  disabled,
  awaiting,
  turnCount,
  maxTurns,
}: Props) {
  const isDesktop = useBreakpoint(768);
  const { value: input, setValue: setInput, clear: clearDraft } = useDraftPersistence(sessionId);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const {
    ref: scrollRef,
    stuck,
    scrollToBottom,
  } = useScrollToBottom<HTMLDivElement>([messages, awaiting]);

  const toolResultsById = useMemo(() => {
    const map: Record<string, ChatMessage> = {};
    for (const m of messages) {
      if (m.role === "tool" && m.tool_call_id) map[m.tool_call_id] = m;
    }
    return map;
  }, [messages]);

  const handleSubmit = useCallback(
    async (e?: React.FormEvent) => {
      e?.preventDefault();
      const text = input.trim();
      if (!text || disabled) return;
      clearDraft();
      await onSend(text);
    },
    [input, disabled, clearDraft, onSend]
  );

  // Global Cmd/Ctrl+Enter to submit.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        void handleSubmit();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handleSubmit]);

  return (
    <div className="flex h-full min-h-0 flex-col bg-panel">
      <div
        ref={scrollRef}
        role="log"
        aria-live="polite"
        aria-atomic="false"
        aria-label="Conversation"
        className="relative flex-1 overflow-y-auto px-6 py-6"
      >
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.length === 0 && !awaiting && (
            <div className="rounded-lg border border-dashed border-border bg-bg/40 p-4 text-center text-sm text-muted">
              Your conversation will appear here. Read the instructions on the right, then
              write your first message below.
            </div>
          )}

          {messages.map((m, i) => (
            <MessageBubble
              key={i}
              msg={m}
              allToolResults={toolResultsById}
            />
          ))}

          {awaiting && <TypingIndicator />}
        </div>
      </div>

      {!stuck && (
        <button
          type="button"
          aria-label="Scroll to latest message"
          onClick={() => scrollToBottom(true)}
          className="pointer-events-auto absolute bottom-24 left-1/2 z-10 -translate-x-1/2 rounded-full border border-border bg-panelAlt px-3 py-1 text-xs text-muted shadow-lg hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          <ArrowDown className="mr-1 inline h-3 w-3" aria-hidden="true" />
          Jump to latest
        </button>
      )}

      <form
        onSubmit={handleSubmit}
        className="border-t border-border bg-panelAlt px-4 py-3 md:px-6"
      >
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <TextareaAutosize
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              disabled
                ? "Conversation ended."
                : isDesktop
                ? "Type your message. Enter to send, Shift+Enter for newline."
                : "Type your message. Tap Send when ready."
            }
            aria-label="Your message"
            disabled={disabled}
            minRows={1}
            maxRows={8}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && isDesktop) {
                e.preventDefault();
                void handleSubmit();
              }
            }}
            className="min-h-[44px] flex-1 resize-none rounded-lg bg-bg px-3 py-2 text-sm text-text outline-none ring-1 ring-border placeholder:text-muted/70 focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-60"
          />
          <button
            type="button"
            onClick={onRequestStop}
            disabled={disabled}
            aria-label="End conversation and go to survey"
            className="hidden h-[44px] items-center gap-1 rounded-lg border border-border px-3 text-xs text-muted hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-50 md:inline-flex"
          >
            <Square className="h-3.5 w-3.5" aria-hidden="true" />
            End
          </button>
          <button
            type="submit"
            disabled={disabled || !input.trim()}
            className="h-[44px] rounded-lg bg-accent px-4 text-sm font-medium text-bg hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-50"
          >
            Send
          </button>
        </div>

        <div className="mx-auto mt-1.5 flex max-w-3xl items-center justify-between text-[11px] text-muted">
          <div>
            {isDesktop ? (
              <>Press <kbd className="rounded bg-bg px-1 py-[1px]">⌘</kbd>+<kbd className="rounded bg-bg px-1 py-[1px]">Enter</kbd> to send, or type <code className="rounded bg-bg px-1">/stop</code> to end.</>
            ) : (
              <>Tap <b>Send</b> to submit, <b>End</b> to finish.</>
            )}
          </div>
          <div className="tabular-nums">
            Turn {turnCount} / {maxTurns}
          </div>
        </div>
      </form>
    </div>
  );
}

function MessageBubble({
  msg,
  allToolResults,
}: {
  msg: ChatMessage;
  allToolResults: Record<string, ChatMessage>;
}) {
  if (msg.role === "tool") return null; // Rendered inside the assistant bubble.

  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="group relative max-w-[80%]">
          <div className="rounded-2xl rounded-br-sm bg-accent/20 px-4 py-2 text-sm text-text">
            {msg.content}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="group space-y-2">
      {msg.content && (
        <div className="relative rounded-2xl bg-panelAlt px-4 py-3 text-text">
          <Markdown>{msg.content}</Markdown>
          <CopyButton text={msg.content} />
        </div>
      )}
      {msg.tool_calls?.map((tc) => (
        <ToolCallCard
          key={tc.id}
          name={tc.function.name}
          argsJson={tc.function.arguments}
          result={allToolResults[tc.id]?.content || null}
        />
      ))}
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  async function handle() {
    const ok = await copyToClipboard(text);
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    }
  }
  return (
    <button
      type="button"
      onClick={handle}
      aria-label={copied ? "Copied" : "Copy message"}
      className={cn(
        "absolute right-2 top-2 rounded p-1 text-muted opacity-0 transition hover:text-text focus-visible:outline-none focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-accent",
        "group-hover:opacity-100"
      )}
    >
      {copied ? <Check className="h-3.5 w-3.5" aria-hidden="true" /> : <Copy className="h-3.5 w-3.5" aria-hidden="true" />}
    </button>
  );
}
