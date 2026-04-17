"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  api,
  CHAT_STREAM_URL,
  type ChatMessage,
  type SessionStartResponse,
  type SurveySchema,
} from "@/lib/api";
import { postSSE } from "@/lib/sse";
import { withRetry } from "@/lib/retry";
import { ChatPanel } from "@/components/ChatPanel";
import { InstructionPanel } from "@/components/InstructionPanel";
import { SurveyPanel } from "@/components/SurveyPanel";
import { ErrorBanner } from "@/components/ErrorBanner";
import { StopConfirm } from "@/components/StopConfirm";
import type { SessionPhase } from "@/components/StatusPill";

type Phase = "loading" | "error" | "chatting" | "surveying" | "done";

const MAX_TURNS = 30;

export default function StudyClient() {
  const params = useSearchParams();

  const [phase, setPhase] = useState<Phase>("loading");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [transientError, setTransientError] = useState<string | null>(null);
  const [session, setSession] = useState<SessionStartResponse | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [awaiting, setAwaiting] = useState(false);
  const [surveySchema, setSurveySchema] = useState<SurveySchema | null>(null);
  const [completionCode, setCompletionCode] = useState<string | null>(null);
  const [stopDialogOpen, setStopDialogOpen] = useState(false);

  useEffect(() => {
    const prolific_pid = params.get("PROLIFIC_PID") || params.get("prolific_pid");
    const study_id = params.get("STUDY_ID") || params.get("study_id");
    const prolific_session_id = params.get("SESSION_ID") || params.get("session_id");
    const task_type = (params.get("task_type") as "tau2" | "p4g") || "tau2";

    if (!prolific_pid || !study_id || !prolific_session_id) {
      setErrorMsg(
        "Missing Prolific URL parameters. Please follow the link provided on Prolific."
      );
      setPhase("error");
      return;
    }

    withRetry(
      () => api.startSession({ prolific_pid, study_id, prolific_session_id, task_type }),
      { attempts: 3 }
    )
      .then((s) => {
        setSession(s);
        setPhase("chatting");
      })
      .catch((err: unknown) => {
        setErrorMsg(describeError(err));
        setPhase("error");
      });
  }, [params]);

  const turnCount = useMemo(
    () => messages.filter((m) => m.role === "user").length,
    [messages]
  );

  const handleStop = useCallback(async () => {
    if (!session) return;
    try {
      const res = await withRetry(() => api.stop({ session_id: session.session_id }), {
        attempts: 3,
      });
      setSurveySchema(res.survey_schema);
      setPhase("surveying");
    } catch (err) {
      setTransientError(describeError(err));
    }
  }, [session]);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!session) return;
      const trimmed = text.trim();
      if (trimmed === "/stop") {
        setStopDialogOpen(true);
        return;
      }

      // Optimistic user bubble + placeholder assistant we stream into.
      setMessages((m) => [
        ...m,
        { role: "user", content: trimmed },
        { role: "assistant", content: "" },
      ]);
      setAwaiting(true);
      setTransientError(null);

      // Index of the assistant placeholder we're currently streaming into.
      // We may create more assistant messages later if the model emits
      // multiple passes (tool-call loop).
      let activeAssistantIdx: number | null = null;
      const seenToolCalls: Record<string, boolean> = {};

      function ensureActiveAssistant(): number {
        if (activeAssistantIdx !== null) return activeAssistantIdx;
        let newIdx = -1;
        setMessages((m) => {
          newIdx = m.length;
          return [...m, { role: "assistant", content: "" }];
        });
        activeAssistantIdx = newIdx;
        return newIdx;
      }

      try {
        await postSSE(
          CHAT_STREAM_URL,
          { session_id: session.session_id, user_message: trimmed },
          async (event, data: any) => {
            switch (event) {
              case "assistant_start": {
                if (activeAssistantIdx === null) {
                  // The initial placeholder we pushed above is the current one.
                  activeAssistantIdx = -1;
                  setMessages((m) => {
                    activeAssistantIdx = m.length - 1;
                    return m;
                  });
                } else {
                  ensureActiveAssistant();
                }
                break;
              }
              case "assistant_delta": {
                const delta = String(data?.content ?? "");
                if (!delta) break;
                const idx = activeAssistantIdx ?? ensureActiveAssistant();
                setMessages((m) => {
                  const next = [...m];
                  const cur = next[idx];
                  if (cur && cur.role === "assistant") {
                    next[idx] = { ...cur, content: (cur.content ?? "") + delta };
                  }
                  return next;
                });
                break;
              }
              case "assistant_end": {
                const idx = activeAssistantIdx;
                if (idx !== null) {
                  setMessages((m) => {
                    const next = [...m];
                    const cur = next[idx];
                    if (cur && cur.role === "assistant") {
                      next[idx] = {
                        ...cur,
                        content: data?.content ?? cur.content,
                        tool_calls: data?.tool_calls ?? null,
                      };
                    }
                    return next;
                  });
                }
                activeAssistantIdx = null;
                break;
              }
              case "tool_start": {
                // nothing to render yet; card shows pending once assistant_end
                // landed the tool_calls array.
                break;
              }
              case "tool_end": {
                const tcId: string = data?.tool_call_id;
                if (!tcId || seenToolCalls[tcId]) break;
                seenToolCalls[tcId] = true;
                setMessages((m) => [
                  ...m,
                  {
                    role: "tool",
                    content: JSON.stringify(data.result, null, 0),
                    tool_call_id: tcId,
                    tool_name: data?.name ?? null,
                  },
                ]);
                break;
              }
              case "error": {
                setTransientError(String(data?.message ?? "stream error"));
                break;
              }
              case "done": {
                break;
              }
            }
          }
        );
      } catch (err) {
        setTransientError(describeError(err));
        // Clean up the empty placeholder if it received nothing.
        setMessages((m) => {
          const last = m[m.length - 1];
          if (last && last.role === "assistant" && !last.content && !last.tool_calls) {
            return m.slice(0, -1);
          }
          return m;
        });
      } finally {
        setAwaiting(false);
      }
    },
    [session]
  );

  const handleSurveySubmit = useCallback(
    async (responses: Record<string, unknown>, free_text: string) => {
      if (!session) return;
      try {
        const res = await withRetry(
          () =>
            api.survey({
              session_id: session.session_id,
              responses,
              free_text: free_text || null,
            }),
          { attempts: 3 }
        );
        setCompletionCode(res.completion_code);
        setPhase("done");
      } catch (err) {
        setTransientError(describeError(err));
      }
    },
    [session]
  );

  if (phase === "loading") return <CenterMessage>Loading session...</CenterMessage>;
  if (phase === "error") return <CenterMessage variant="error">{errorMsg}</CenterMessage>;

  if (phase === "done" && completionCode) {
    return (
      <CenterMessage>
        <div className="space-y-3 text-center">
          <h2 className="text-xl font-semibold text-text">Thank you for participating!</h2>
          <p className="text-muted">Your completion code is:</p>
          <code className="inline-block rounded bg-panelAlt px-3 py-2 font-mono text-lg text-text">
            {completionCode}
          </code>
          <p className="text-sm text-muted">
            Paste this code into the Prolific form to mark the task as completed.
          </p>
        </div>
      </CenterMessage>
    );
  }

  if (phase === "surveying" && surveySchema) {
    return (
      <div className="min-h-screen bg-bg">
        {transientError && (
          <div className="mx-auto max-w-2xl px-6 pt-6">
            <ErrorBanner
              message={transientError}
              onDismiss={() => setTransientError(null)}
            />
          </div>
        )}
        <SurveyPanel schema={surveySchema} onSubmit={handleSurveySubmit} />
      </div>
    );
  }

  if (!session) return null;

  const chatPhase: SessionPhase =
    phase !== "chatting"
      ? "stopped"
      : transientError
      ? "error"
      : awaiting
      ? "waiting"
      : "chatting";

  return (
    <div className="grid h-screen min-h-0 grid-cols-1 bg-bg md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="relative flex h-full min-h-0 flex-col">
        {transientError && (
          <div className="border-b border-border bg-panel px-4 py-2">
            <ErrorBanner
              message={transientError}
              onDismiss={() => setTransientError(null)}
            />
          </div>
        )}
        <ChatPanel
          sessionId={session.session_id}
          messages={messages}
          onSend={sendMessage}
          onRequestStop={() => setStopDialogOpen(true)}
          disabled={phase !== "chatting"}
          awaiting={awaiting}
          turnCount={turnCount}
          maxTurns={MAX_TURNS}
        />
      </div>
      <div className="hidden h-full min-h-0 md:block">
        <InstructionPanel
          instruction={session.task_instruction}
          taskSplit={session.task_split}
          taskIdx={session.task_idx}
          runtimeId={session.session_id}
          phase={chatPhase}
        />
      </div>

      <StopConfirm
        open={stopDialogOpen}
        onCancel={() => setStopDialogOpen(false)}
        onConfirm={() => {
          setStopDialogOpen(false);
          void handleStop();
        }}
      />
    </div>
  );
}

function CenterMessage({
  children,
  variant,
}: {
  children: React.ReactNode;
  variant?: "error";
}) {
  return (
    <main className="flex min-h-screen items-center justify-center px-6">
      <div
        className={`max-w-lg rounded-xl border px-6 py-6 ${
          variant === "error"
            ? "border-red-700/40 bg-red-950/30 text-red-200"
            : "border-border bg-panel text-text"
        }`}
      >
        {children}
      </div>
    </main>
  );
}

function describeError(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}
