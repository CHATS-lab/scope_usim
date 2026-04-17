"use client";

import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, type ChatMessage, type SessionStartResponse, type SurveySchema } from "@/lib/api";
import { ChatPanel } from "@/components/ChatPanel";
import { InstructionPanel } from "@/components/InstructionPanel";
import { SurveyPanel } from "@/components/SurveyPanel";

type Phase = "loading" | "error" | "chatting" | "surveying" | "done";

export default function StudyClient() {
  const params = useSearchParams();

  const [phase, setPhase] = useState<Phase>("loading");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [session, setSession] = useState<SessionStartResponse | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [awaiting, setAwaiting] = useState(false);
  const [surveySchema, setSurveySchema] = useState<SurveySchema | null>(null);
  const [completionCode, setCompletionCode] = useState<string | null>(null);

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

    api
      .startSession({ prolific_pid, study_id, prolific_session_id, task_type })
      .then((s) => {
        setSession(s);
        setPhase("chatting");
      })
      .catch((err: unknown) => {
        setErrorMsg(String(err));
        setPhase("error");
      });
  }, [params]);

  const handleStop = useCallback(async () => {
    if (!session) return;
    const res = await api.stop({ session_id: session.session_id });
    setSurveySchema(res.survey_schema);
    setPhase("surveying");
  }, [session]);

  const handleSend = useCallback(
    async (text: string) => {
      if (!session) return;
      if (text.trim() === "/stop") {
        await handleStop();
        return;
      }
      setMessages((m) => [...m, { role: "user", content: text }]);
      setAwaiting(true);
      try {
        const reply = await api.chat({ session_id: session.session_id, user_message: text });
        setMessages((m) => [...m, ...reply.messages]);
      } catch (err) {
        setMessages((m) => [
          ...m,
          { role: "assistant", content: `[error: ${String(err)}]` },
        ]);
      } finally {
        setAwaiting(false);
      }
    },
    [session, handleStop]
  );

  const handleSurveySubmit = useCallback(
    async (responses: Record<string, unknown>, free_text: string) => {
      if (!session) return;
      const res = await api.survey({
        session_id: session.session_id,
        responses,
        free_text: free_text || null,
      });
      setCompletionCode(res.completion_code);
      setPhase("done");
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
        <SurveyPanel schema={surveySchema} onSubmit={handleSurveySubmit} />
      </div>
    );
  }

  if (!session) return null;

  return (
    <div className="grid h-screen grid-cols-[1fr_1fr] bg-bg">
      <ChatPanel
        messages={messages}
        onSend={handleSend}
        disabled={phase !== "chatting"}
        awaiting={awaiting}
      />
      <InstructionPanel
        instruction={session.task_instruction}
        taskSplit={session.task_split}
        taskIdx={session.task_idx}
        runtimeId={session.session_id}
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
