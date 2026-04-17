"use client";

import { useCallback, useRef, useState } from "react";
import type { ChatMessage } from "@/lib/api";
import { postSSE } from "@/lib/sse";

type StreamEvent =
  | { event: "assistant_start" }
  | { event: "assistant_delta"; content: string }
  | { event: "assistant_end"; content: string | null; tool_calls: any[] | null }
  | { event: "tool_start"; tool_call_id: string; name: string; arguments: any }
  | { event: "tool_end"; tool_call_id: string; name: string; result: any }
  | { event: "error"; message: string }
  | { event: "done"; session_status: string };

interface SendArgs {
  sessionId: string;
  url: string;
  userMessage: string;
  onError: (msg: string) => void;
}

/**
 * Streaming chat that mutates a messages array via setMessages.
 *
 * The tricky part is keeping the "currently streaming assistant message" in
 * sync across multiple setMessages calls when React batches updates. We track
 * the slot as a ref so every delta lands in the same bubble.
 */
export function useStreamingChat(
  setMessages: (updater: (m: ChatMessage[]) => ChatMessage[]) => void
) {
  const [awaiting, setAwaiting] = useState(false);
  // Slot index into the messages array for the in-flight assistant bubble.
  // null means "no open bubble -- open one on the next delta".
  const openSlot = useRef<number | null>(null);

  const send = useCallback(
    async ({ sessionId, url, userMessage, onError }: SendArgs) => {
      openSlot.current = null;
      setAwaiting(true);

      // Push user bubble immediately (optimistic).
      setMessages((m) => [...m, { role: "user", content: userMessage }]);

      const openAssistantBubble = () => {
        setMessages((m) => {
          openSlot.current = m.length;
          return [...m, { role: "assistant", content: "" }];
        });
      };

      const appendToOpenAssistant = (delta: string) => {
        setMessages((m) => {
          let slot = openSlot.current;
          // Defensive: if we somehow lost the slot, use the last assistant.
          if (slot === null || !m[slot] || m[slot].role !== "assistant") {
            for (let i = m.length - 1; i >= 0; i--) {
              if (m[i].role === "assistant") {
                slot = i;
                openSlot.current = i;
                break;
              }
            }
          }
          if (slot === null) {
            // No open bubble yet -- open one now.
            openSlot.current = m.length;
            return [...m, { role: "assistant", content: delta }];
          }
          const next = m.slice();
          const cur = next[slot];
          next[slot] = { ...cur, content: (cur.content ?? "") + delta };
          return next;
        });
      };

      const finalizeAssistant = (
        content: string | null,
        tool_calls: any[] | null
      ) => {
        setMessages((m) => {
          const slot = openSlot.current;
          if (slot === null || !m[slot] || m[slot].role !== "assistant") return m;
          const next = m.slice();
          next[slot] = {
            ...next[slot],
            content: content ?? next[slot].content ?? "",
            tool_calls: tool_calls ?? null,
          };
          return next;
        });
        // Next assistant_start (after a tool loop) will open a new slot.
        openSlot.current = null;
      };

      try {
        await postSSE(url, { session_id: sessionId, user_message: userMessage }, (ev, data: any) => {
          switch (ev) {
            case "assistant_start":
              openAssistantBubble();
              break;
            case "assistant_delta":
              appendToOpenAssistant(String(data?.content ?? ""));
              break;
            case "assistant_end":
              finalizeAssistant(
                data?.content ?? null,
                (data?.tool_calls as any[] | null) ?? null
              );
              break;
            case "tool_start":
              // Nothing to add here; the assistant_end already delivered the
              // tool_calls array and ToolCallCard renders a "pending" state
              // until the matching tool_end arrives.
              break;
            case "tool_end": {
              const tcId: string | undefined = data?.tool_call_id;
              if (!tcId) break;
              setMessages((m) => [
                ...m,
                {
                  role: "tool",
                  content: JSON.stringify(data.result),
                  tool_call_id: tcId,
                  tool_name: data?.name ?? null,
                },
              ]);
              break;
            }
            case "error":
              onError(String(data?.message ?? "stream error"));
              break;
            case "done":
              break;
          }
        });
      } catch (err) {
        onError(err instanceof Error ? err.message : String(err));
        // Remove empty trailing assistant bubble.
        setMessages((m) => {
          const last = m[m.length - 1];
          if (last && last.role === "assistant" && !last.content && !last.tool_calls) {
            return m.slice(0, -1);
          }
          return m;
        });
      } finally {
        openSlot.current = null;
        setAwaiting(false);
      }
    },
    [setMessages]
  );

  return { send, awaiting };
}
