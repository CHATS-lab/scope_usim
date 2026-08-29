"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  api,
  CHAT_STREAM_URL,
  type ChatMessage,
  type DebriefInfo,
  type SessionStartResponse,
  type StudyVariant,
  type SurveySchema,
} from "@/lib/api";
import { withRetry } from "@/lib/retry";
import { useStreamingChat } from "@/hooks/useStreamingChat";
import { ChatPanel } from "@/components/ChatPanel";
import { InstructionPanel } from "@/components/InstructionPanel";
import {
  InstructionPanelV2,
  type StructuredTask,
} from "@/components/InstructionPanelV2";
import { SurveyPanel } from "@/components/SurveyPanel";
import { ErrorBanner } from "@/components/ErrorBanner";
import { StopConfirm } from "@/components/StopConfirm";
import { Sidebar } from "@/components/Sidebar";
import { TopBar } from "@/components/TopBar";
import { DebriefPanel } from "@/components/DebriefPanel";
import type { SessionPhase } from "@/components/StatusPill";
import { cn } from "@/lib/cn";

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
  const [prolificCode, setProlificCode] = useState<string | null>(null);
  const [prolificRedirectUrl, setProlificRedirectUrl] = useState<string | null>(null);
  const [debrief, setDebrief] = useState<DebriefInfo | null>(null);
  const [stopDialogOpen, setStopDialogOpen] = useState(false);
  const [instructionsOpen, setInstructionsOpen] = useState(true);
  // Right-panel pagination: during chatting only instructions are shown;
  // after `/stop` the survey appears as a second "page" participants can
  // flip to while still seeing the conversation on the left.
  const [rightPanelView, setRightPanelView] = useState<"instructions" | "survey">(
    "instructions"
  );

  // Auto-flip the right panel to the survey as soon as one becomes available
  // so participants don't have to hunt for it. They can still click back to
  // Instructions from the toggle header if they want to re-check the task.
  useEffect(() => {
    if (phase === "surveying" && surveySchema) {
      setRightPanelView("survey");
    }
  }, [phase, surveySchema]);

  const { send: streamChat, awaiting } = useStreamingChat(setMessages);

  useEffect(() => {
    const prolific_pid = params.get("PROLIFIC_PID") || params.get("prolific_pid");
    const study_id = params.get("STUDY_ID") || params.get("study_id");
    const prolific_session_id = params.get("SESSION_ID") || params.get("session_id");
    const task_type = (params.get("task_type") as "tau2" | "p4g") || "tau2";
    const variant: StudyVariant = params.get("variant") === "v1" ? "v1" : "v2";

    if (!prolific_pid || !study_id || !prolific_session_id) {
      setErrorMsg(
        "Missing Prolific URL parameters. Please follow the link provided on Prolific."
      );
      setPhase("error");
      return;
    }

    (async () => {
      try {
        const s = await withRetry(
          () =>
            api.startSession({
              prolific_pid,
              study_id,
              prolific_session_id,
              task_type,
              variant,
            }),
          { attempts: 3 }
        );
        setSession(s);

        // Participant is returning after already finishing — jump straight to
        // the debrief screen with their existing completion code + Prolific
        // redirect data (so the auto-redirect still fires on a re-open).
        if (s.already_completed && s.completion_code && s.debrief) {
          setCompletionCode(s.completion_code);
          setProlificCode(s.prolific_completion_code ?? null);
          setProlificRedirectUrl(s.prolific_redirect_url ?? null);
          setDebrief(s.debrief);
          setPhase("done");
          return;
        }

        // Resumed mid-conversation: restore history so the chat feels continuous.
        if (s.resumed) {
          try {
            const prior = await api.getTurns(s.session_id);
            if (prior.length > 0) setMessages(prior);
          } catch {
            // Non-fatal.
          }
        }
        setPhase("chatting");
      } catch (err: unknown) {
        setErrorMsg(describeError(err));
        setPhase("error");
      }
    })();
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
        // Agent called transfer_to_human_agents — backend already marked the
        // session STOPPED, so jump straight to the survey.
        onSessionEnded: () => {
          void handleStop();
        },
      });
    },
    [session, streamChat, handleStop]
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
        setProlificCode(res.prolific_completion_code ?? null);
        setProlificRedirectUrl(res.prolific_redirect_url ?? null);
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
    return (
      <DebriefPanel
        completionCode={completionCode}
        prolificCode={prolificCode}
        prolificRedirectUrl={prolificRedirectUrl}
        debrief={debrief}
      />
    );
  }

  if (!session) return null;

  const turnLimitHit = turnCount >= MAX_TURNS;
  const chatPhase: SessionPhase =
    phase !== "chatting"
      ? "stopped"
      : turnLimitHit
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
        <div className="hidden h-full min-h-0 flex-col border-l border-border bg-bg md:flex">
          {phase === "surveying" && surveySchema && (
            <div className="flex shrink-0 border-b border-border bg-panel">
              <button
                type="button"
                onClick={() => setRightPanelView("instructions")}
                className={cn(
                  "flex-1 px-4 py-2 text-xs font-medium transition-colors",
                  rightPanelView === "instructions"
                    ? "bg-bg text-text"
                    : "text-muted hover:text-text"
                )}
              >
                ← Instructions
              </button>
              <button
                type="button"
                onClick={() => setRightPanelView("survey")}
                className={cn(
                  "flex-1 border-l border-border px-4 py-2 text-xs font-medium transition-colors",
                  rightPanelView === "survey"
                    ? "bg-bg text-text"
                    : "text-muted hover:text-text"
                )}
              >
                Survey →
              </button>
            </div>
          )}

          {/*
            Both panels stay mounted so in-progress survey answers survive when
            the participant flips back to Instructions to double-check details.
            `hidden` toggles visibility without unmounting.
          */}
          <div
            className={cn(
              "min-h-0 flex-1",
              rightPanelView !== "instructions" && "hidden"
            )}
          >
            {session.variant === "v2" ? (
              <InstructionPanelV2
                structured={(session.task_structured as StructuredTask | null) ?? null}
                fallbackInstruction={session.task_instruction}
                taskSplit={session.task_split}
                taskIdx={session.task_idx}
                runtimeId={session.session_id}
              />
            ) : (
              <InstructionPanel
                instruction={session.task_instruction}
                taskSplit={session.task_split}
                taskIdx={session.task_idx}
                runtimeId={session.session_id}
              />
            )}
          </div>

          {surveySchema && (
            <div
              className={cn(
                "min-h-0 flex-1 overflow-y-auto",
                rightPanelView !== "survey" && "hidden"
              )}
            >
              <SurveyPanel schema={surveySchema} onSubmit={handleSurveySubmit} />
            </div>
          )}
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
