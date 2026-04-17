"use client";

import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, type AnnotationNextResponse } from "@/lib/api";
import { withRetry } from "@/lib/retry";
import { ChatPanel } from "@/components/ChatPanel";
import { InstructionPanel } from "@/components/InstructionPanel";
import { SurveyPanel } from "@/components/SurveyPanel";
import { ErrorBanner } from "@/components/ErrorBanner";
import { TopBar } from "@/components/TopBar";

type Phase = "loading" | "error" | "reviewing" | "annotating" | "done_all";

export default function AnnotateClient() {
  const params = useSearchParams();

  const [phase, setPhase] = useState<Phase>("loading");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [transientError, setTransientError] = useState<string | null>(null);
  const [next, setNext] = useState<AnnotationNextResponse | null>(null);
  const [completionCode, setCompletionCode] = useState<string | null>(null);

  const annotator_id = params.get("ANNOTATOR_ID") || params.get("annotator_id");
  const pinned_session = params.get("session_id");

  const loadNext = useCallback(async () => {
    if (!annotator_id) return;
    try {
      const res = await withRetry(
        () => api.annotationNext(annotator_id, pinned_session),
        { attempts: 3 }
      );
      if (res.done) {
        setPhase("done_all");
      } else {
        setNext(res);
        setPhase("reviewing");
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
      setPhase("error");
    }
  }, [annotator_id, pinned_session]);

  useEffect(() => {
    if (!annotator_id) {
      setErrorMsg(
        "Missing ANNOTATOR_ID URL parameter. Please use the link you were sent."
      );
      setPhase("error");
      return;
    }
    void loadNext();
  }, [annotator_id, loadNext]);

  const handleSubmitAnnotation = useCallback(
    async (responses: Record<string, unknown>, free_text: string) => {
      if (!annotator_id || !next) return;
      try {
        const res = await withRetry(
          () =>
            api.submitAnnotation({
              annotator_id,
              session_id: next.session_id,
              responses,
              free_text: free_text || null,
            }),
          { attempts: 3 }
        );
        // Pinned-session flow: always stop after this one so the annotator
        // doesn't loop on the same session (the URL pin would just re-serve
        // it). Normal queue flow: advance to next or finish.
        if (pinned_session || !res.next_available) {
          setCompletionCode(res.completion_code);
          setPhase("done_all");
          return;
        }
        setPhase("loading");
        setNext(null);
        await loadNext();
      } catch (err) {
        setTransientError(err instanceof Error ? err.message : String(err));
      }
    },
    [annotator_id, next, loadNext, pinned_session]
  );

  if (phase === "loading") {
    return <Center>Loading conversation for review…</Center>;
  }
  if (phase === "error") {
    return (
      <Center variant="error">
        <div className="space-y-2">
          <div className="font-semibold">Couldn't load annotation task</div>
          <div className="text-sm">{errorMsg}</div>
        </div>
      </Center>
    );
  }
  if (phase === "done_all") {
    return (
      <Center>
        <div className="space-y-3 text-center">
          <h2 className="text-xl font-semibold text-text">All done — thank you!</h2>
          <p className="text-muted">
            You've annotated every conversation currently available to you.
          </p>
          {completionCode && (
            <>
              <p className="text-muted">Your completion code is:</p>
              <code className="inline-block rounded bg-panelAlt px-3 py-2 font-mono text-lg text-text">
                {completionCode}
              </code>
              <p className="text-sm text-muted">
                Paste this code into the Prolific form to mark the task completed.
              </p>
            </>
          )}
        </div>
      </Center>
    );
  }

  if (!next) return null;

  if (phase === "annotating") {
    return (
      <div className="min-h-screen bg-bg">
        <div className="mx-auto max-w-2xl px-6 pt-4 text-xs text-muted">
          Annotator: <span className="text-text">{annotator_id}</span> ·{" "}
          {next.annotations_done} reviewed · {next.annotations_available} remaining
        </div>
        {transientError && (
          <div className="mx-auto max-w-2xl px-6 pt-4">
            <ErrorBanner
              message={transientError}
              onDismiss={() => setTransientError(null)}
            />
          </div>
        )}
        <SurveyPanel schema={next.survey_schema} onSubmit={handleSubmitAnnotation} />
      </div>
    );
  }

  // phase === "reviewing"
  return (
    <div className="grid h-screen min-h-0 grid-cols-1 bg-bg md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="relative flex h-full min-h-0 flex-col">
        <TopBar
          phase="stopped"
          turnCount={next.turn_count}
          maxTurns={30}
        />
        <div className="border-b border-border bg-panel px-4 py-2 text-xs text-muted">
          <span className="text-text">Annotator view.</span> This is a completed
          conversation. Read through, then click <b>Continue to annotation form</b>{" "}
          below. ({next.annotations_done} reviewed · {next.annotations_available}{" "}
          remaining)
        </div>
        <ChatPanel
          sessionId={next.session_id}
          messages={next.messages}
          onSend={async () => {}}
          onRequestStop={() => {}}
          disabled
          awaiting={false}
          perspective="annotator"
          readOnly
        />
        <div className="border-t border-border bg-panel px-4 py-3">
          <button
            type="button"
            onClick={() => setPhase("annotating")}
            className="w-full rounded-lg bg-accent px-4 py-2 text-sm font-medium text-bg hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            Continue to annotation form
          </button>
        </div>
      </div>
      <div className="hidden h-full min-h-0 md:block">
        <InstructionPanel
          instruction={next.task_instruction}
          taskSplit={next.task_split}
          taskIdx={next.task_idx}
          runtimeId={next.session_id}
        />
      </div>
    </div>
  );
}

function Center({
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
