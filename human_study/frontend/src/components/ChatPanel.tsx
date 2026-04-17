"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ChatMessage } from "@/lib/api";
import { ToolCallCard } from "./ToolCallCard";

interface Props {
  messages: ChatMessage[];
  onSend: (text: string) => Promise<void> | void;
  disabled: boolean;
  awaiting: boolean;
}

export function ChatPanel({ messages, onSend, disabled, awaiting }: Props) {
  const [input, setInput] = useState("");
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, awaiting]);

  const toolResultsById = useMemo(() => {
    const map: Record<string, ChatMessage> = {};
    for (const m of messages) {
      if (m.role === "tool" && m.tool_call_id) map[m.tool_call_id] = m;
    }
    return map;
  }, [messages]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || disabled) return;
    setInput("");
    await onSend(text);
  }

  return (
    <div className="flex h-full flex-col bg-panel">
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.map((m, i) => (
            <MessageBubble
              key={i}
              msg={m}
              toolResult={m.tool_calls?.[0]?.id ? toolResultsById[m.tool_calls[0].id] : undefined}
              allToolResults={toolResultsById}
            />
          ))}
          {awaiting && (
            <div className="flex items-center gap-2 text-muted text-sm">
              <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-accent" />
              Awaiting response...
            </div>
          )}
          <div ref={endRef} />
        </div>
      </div>

      <form
        onSubmit={handleSubmit}
        className="border-t border-border bg-panelAlt px-6 py-4"
      >
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={disabled ? "Conversation ended." : "Type your message. Use /stop to end."}
            disabled={disabled}
            rows={2}
            className="min-h-[42px] max-h-40 flex-1 resize-none rounded-lg bg-bg px-3 py-2 text-sm text-text outline-none ring-1 ring-border focus:ring-accent disabled:opacity-60"
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
          />
          <button
            type="submit"
            disabled={disabled || !input.trim()}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-50"
          >
            Send
          </button>
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
  toolResult?: ChatMessage;
  allToolResults: Record<string, ChatMessage>;
}) {
  if (msg.role === "tool") return null; // tool results render inside the assistant bubble

  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-accent/20 px-4 py-2 text-sm text-text">
          {msg.content}
        </div>
      </div>
    );
  }

  // assistant
  return (
    <div className="space-y-2">
      {msg.content && (
        <div className="rounded-2xl bg-panelAlt px-4 py-3 text-sm leading-relaxed text-text">
          {msg.content}
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
