"use client";

import { useMemo, useState } from "react";
import { ArrowDown, Copy, Check, Loader2 } from "lucide-react";
import type { ChatMessage } from "@/lib/api";
import { cn } from "@/lib/cn";
import { copyToClipboard } from "@/lib/copy";
import { Markdown } from "./Markdown";
import { ToolCallCard } from "./ToolCallCard";
import { TypingIndicator } from "./TypingIndicator";
import { InputBar } from "./InputBar";
import { SpeakerBadge } from "./SpeakerBadge";
import { useDraftPersistence } from "@/hooks/useDraftPersistence";
import { useScrollToBottom } from "@/hooks/useScrollToBottom";

interface Props {
  sessionId: string | null;
  messages: ChatMessage[];
  onSend: (text: string) => Promise<void> | void;
  onRequestStop: () => void;
  disabled: boolean;
  awaiting: boolean;
  /** "participant" = the user is the human talking; "annotator" = reviewing someone else's chat */
  perspective?: "participant" | "annotator";
  /** When true, hide the input bar entirely (annotator / read-only mode). */
  readOnly?: boolean;
}

export function ChatPanel({
  sessionId,
  messages,
  onSend,
  onRequestStop,
  disabled,
  awaiting,
  perspective = "participant",
  readOnly = false,
}: Props) {
  const { value: input, setValue: setInput, clear: clearDraft } = useDraftPersistence(sessionId);

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

  const handleSubmit = () => {
    const text = input.trim();
    if (!text || disabled) return;
    clearDraft();
    void onSend(text);
  };

  // Any assistant tool_call that doesn't yet have a matching tool result?
  const pendingTool = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role !== "assistant" || !m.tool_calls) continue;
      for (const tc of m.tool_calls) {
        if (!toolResultsById[tc.id]) return tc;
      }
      break;
    }
    return null;
  }, [messages, toolResultsById]);

  return (
    <div className="flex h-full min-h-0 flex-col bg-bg">
      <div
        ref={scrollRef}
        role="log"
        aria-live="polite"
        aria-atomic="false"
        aria-label="Conversation"
        className="relative flex-1 overflow-y-auto px-4 py-6 md:px-8"
      >
        <div className="mx-auto max-w-3xl space-y-3">
          {messages.length === 0 && !awaiting && (
            <div className="rounded-xl border border-dashed border-border bg-panel/40 p-6 text-center text-sm text-muted">
              Your conversation will appear here. Read the instructions on the right,
              then send your first message below.
            </div>
          )}

          {messages.map((m, i) => (
            <MessageBubble
              key={i}
              msg={m}
              allToolResults={toolResultsById}
              perspective={perspective}
            />
          ))}

          {pendingTool && (
            <StatusChip>
              <Loader2 className="h-3.5 w-3.5 animate-spin text-muted" aria-hidden="true" />
              <span>Running {pendingTool.function.name}…</span>
            </StatusChip>
          )}

          {awaiting && !pendingTool && <TypingIndicator />}
        </div>

        {!stuck && (
          <button
            type="button"
            aria-label="Scroll to latest message"
            onClick={() => scrollToBottom(true)}
            className="pointer-events-auto sticky bottom-4 left-1/2 z-10 -translate-x-1/2 rounded-full border border-border bg-panelAlt px-3 py-1 text-xs text-muted shadow-lg hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <ArrowDown className="mr-1 inline h-3 w-3" aria-hidden="true" />
            Jump to latest
          </button>
        )}
      </div>

      {!readOnly && (
        <InputBar
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          onRequestStop={onRequestStop}
          disabled={disabled}
          awaiting={awaiting}
        />
      )}
    </div>
  );
}

function StatusChip({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-border/60 bg-panel/40 px-3 py-2 text-xs text-muted">
      {children}
    </div>
  );
}

function MessageBubble({
  msg,
  allToolResults,
  perspective,
}: {
  msg: ChatMessage;
  allToolResults: Record<string, ChatMessage>;
  perspective: "participant" | "annotator";
}) {
  if (msg.role === "tool") return null; // Rendered inside the assistant bubble.

  if (msg.role === "user") {
    return (
      <div className="flex flex-col items-end gap-1">
        <SpeakerBadge role="user" perspective={perspective} align="right" size="sm" />
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-accent/15 px-4 py-2 text-sm text-text">
          {msg.content}
        </div>
      </div>
    );
  }

  return (
    <div className="group flex flex-col gap-1 py-1">
      <SpeakerBadge role="assistant" perspective={perspective} align="left" size="sm" />
      {msg.content && (
        <div className="relative rounded-2xl bg-panel/50 px-4 py-3 text-sm text-text">
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
