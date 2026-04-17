"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  api,
  CHAT_STREAM_URL,
  type ChatMessage,
  type DebriefInfo,
  type SessionStartResponse,
  type SurveySchema,
} from "@/lib/api";
import { withRetry } from "@/lib/retry";
import { useStreamingChat } from "@/hooks/useStreamingChat";
import { ChatPanel } from "@/components/ChatPanel";
import { InstructionPanel } from "@/components/InstructionPanel";
import { SurveyPanel } from "@/components/SurveyPanel";
import { ErrorBanner } from "@/components/ErrorBanner";
import { StopConfirm } from "@/components/StopConfirm";
import { Sidebar } from "@/components/Sidebar";
import { TopBar } from "@/components/TopBar";
import { DebriefPanel } from "@/components/DebriefPanel";
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
  const [surveySchema, setSurveySchema] = useState<SurveySchema | null>(null);
  const [completionCode, setCompletionCode] = useState<string | null>(null);
  const [debrief, setDebrief] = useState<DebriefInfo | null>(null);
  const [stopDialogOpen, setStopDialogOpen] = useState(false);
  const [instructionsOpen, setInstructionsOpen] = useState(true);

  const { send: streamChat, awaiting } = useStreamingChat(setMessages);

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
      await streamChat({
        sessionId: session.session_id,
        url: CHAT_STREAM_URL,
        userMessage: trimmed,
        onError: (msg) => setTransientError(msg),
      });
    },
    [session, streamChat]
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
        setDebrief(res.debrief);
        setPhase("done");
      } catch (err) {
        setTransientError(describeError(err));
      }
    },
    [session]
  );

  if (phase === "loading") return <CenterMessage>Loading session…</CenterMessage>;
  if (phase === "error") return <CenterMessage variant="error">{errorMsg}</CenterMessage>;

  if (phase === "done" && completionCode && debrief) {
    return <DebriefPanel completionCode={completionCode} debrief={debrief} />;
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
    <div className="grid h-screen min-h-0 grid-cols-[auto_minmax(0,1fr)] bg-bg md:grid-cols-[auto_minmax(0,1fr)_minmax(0,1fr)]">
      <Sidebar
        taskType={session.task_type}
        taskSplit={session.task_split}
        taskIdx={session.task_idx}
        condition={session.condition}
      />

      <div className="relative flex h-full min-h-0 flex-col">
        <TopBar
          phase={chatPhase}
          turnCount={turnCount}
          maxTurns={MAX_TURNS}
          onToggleInstructions={() => setInstructionsOpen((v) => !v)}
          instructionsOpen={instructionsOpen}
        />

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
        />
      </div>

      {instructionsOpen && (
        <div className="hidden h-full min-h-0 md:block">
          <InstructionPanel
            instruction={session.task_instruction}
            taskSplit={session.task_split}
            taskIdx={session.task_idx}
            runtimeId={session.session_id}
          />
        </div>
      )}

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
