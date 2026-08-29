"use client";

import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, type AnnotationNextResponse, type AnnotatorDebrief } from "@/lib/api";
import { withRetry } from "@/lib/retry";
import { ChatPanel } from "@/components/ChatPanel";
import { InstructionPanel } from "@/components/InstructionPanel";
import { SurveyPanel } from "@/components/SurveyPanel";
import { ErrorBanner } from "@/components/ErrorBanner";
import { TopBar } from "@/components/TopBar";
import { cn } from "@/lib/cn";

type Phase = "loading" | "error" | "reviewing" | "annotating" | "done_all";
type RightPanelView = "instructions" | "annotation";

export default function AnnotateClient() {
  const params = useSearchParams();

  const [phase, setPhase] = useState<Phase>("loading");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [transientError, setTransientError] = useState<string | null>(null);
  const [next, setNext] = useState<AnnotationNextResponse | null>(null);
  const [completionCode, setCompletionCode] = useState<string | null>(null);
  const [debrief, setDebrief] = useState<AnnotatorDebrief | null>(null);
  // Right-panel pagination: during reviewing the annotator can only see the
  // instructions; once they click "Continue to annotation form" the form
  // opens in the same pane with a toggle back to the task instructions.
  const [rightPanelView, setRightPanelView] = useState<RightPanelView>("instructions");

  // Auto-flip to the annotation form the moment the annotator advances past
  // reviewing, and reset back to instructions when a new session is loaded.
  useEffect(() => {
    if (phase === "annotating") {
      setRightPanelView("annotation");
    } else if (phase === "reviewing") {
      setRightPanelView("instructions");
    }
  }, [phase]);

  // Prolific injects PROLIFIC_PID / STUDY_ID / SESSION_ID. We accept either
  // PROLIFIC_PID (when recruited through Prolific) or ANNOTATOR_ID (internal
  // testing) as the annotator identifier.
  const annotator_id =
    params.get("PROLIFIC_PID") ||
    params.get("prolific_pid") ||
    params.get("ANNOTATOR_ID") ||
    params.get("annotator_id");
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
          setDebrief(res.debrief);
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
    return <AnnotatorDebriefView debrief={debrief} completionCode={completionCode} />;
  }

  if (!next) return null;

  const showAnnotationForm = phase === "annotating";

  return (
    <div className="grid h-screen min-h-0 grid-cols-1 bg-bg md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="relative flex h-full min-h-0 flex-col">
        <TopBar
          phase="stopped"
          turnCount={next.turn_count}
          maxTurns={30}
        />
        <div className="border-b border-border bg-panel px-4 py-2 text-xs text-muted">
          <span className="text-text">Annotator view.</span>{" "}
          {showAnnotationForm
            ? "Rate the conversation in the form on the right. You can click "
            : "This is a completed conversation. Read through, then click "}
          {!showAnnotationForm && <b>Continue to annotation form</b>}
          {showAnnotationForm && <b>← Instructions</b>}
          {showAnnotationForm
            ? " to re-check the task."
            : " below."}
          {" "}
          ({next.annotations_done} reviewed · {next.annotations_available}{" "}
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
        {!showAnnotationForm && (
          <div className="border-t border-border bg-panel px-4 py-3">
            <button
              type="button"
              onClick={() => setPhase("annotating")}
              className="w-full rounded-lg bg-accent px-4 py-2 text-sm font-medium text-bg hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              Continue to annotation form
            </button>
          </div>
        )}
      </div>

      <div className="hidden h-full min-h-0 flex-col border-l border-border bg-bg md:flex">
        {showAnnotationForm && (
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
              onClick={() => setRightPanelView("annotation")}
              className={cn(
                "flex-1 border-l border-border px-4 py-2 text-xs font-medium transition-colors",
                rightPanelView === "annotation"
                  ? "bg-bg text-text"
                  : "text-muted hover:text-text"
              )}
            >
              Annotation →
            </button>
          </div>
        )}

        {transientError && (
          <div className="border-b border-border bg-panel px-4 py-2">
            <ErrorBanner
              message={transientError}
              onDismiss={() => setTransientError(null)}
            />
          </div>
        )}

        {/*
          Both views stay mounted so in-progress annotation answers survive
          when the annotator flips back to re-read the task instructions.
        */}
        <div
          className={cn(
            "min-h-0 flex-1",
            rightPanelView !== "instructions" && "hidden"
          )}
        >
          <InstructionPanel
            instruction={next.task_instruction}
            taskSplit={next.task_split}
            taskIdx={next.task_idx}
            runtimeId={next.session_id}
          />
        </div>

        {showAnnotationForm && (
          <div
            className={cn(
              "min-h-0 flex-1 overflow-y-auto",
              rightPanelView !== "annotation" && "hidden"
            )}
          >
            <div className="px-6 pt-4 text-xs text-muted">
              Annotator:{" "}
              <span className="text-text">{annotator_id}</span> ·{" "}
              {next.annotations_done} reviewed ·{" "}
              {next.annotations_available} remaining
            </div>
            <SurveyPanel
              schema={next.survey_schema}
              onSubmit={handleSubmitAnnotation}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function AnnotatorDebriefView({
  debrief,
  completionCode,
}: {
  debrief: AnnotatorDebrief | null;
  completionCode: string | null;
}) {
  const summary = debrief?.session_summary;
  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-10">
      <article className="w-full max-w-xl space-y-6 rounded-xl border border-border bg-panel p-8 shadow-xl">
        <header>
          <h1 className="text-xl font-semibold text-text">Thank you for annotating!</h1>
          <p className="mt-1 text-sm text-muted">
            Your ratings have been recorded and will help us evaluate how well
            our trained agents actually converse with real people.
          </p>
          {debrief && (
            <p className="mt-2 text-xs text-muted">
              Annotator:{" "}
              <span className="text-text">{debrief.annotator_id}</span> ·{" "}
              {debrief.total_annotated}{" "}
              {debrief.total_annotated === 1 ? "session" : "sessions"} rated in
              total.
            </p>
          )}
        </header>

        {completionCode && (
          <section
            aria-labelledby="ann-code-heading"
            className="rounded-lg border border-accent/30 bg-accent/5 p-4"
          >
            <h2 id="ann-code-heading" className="text-sm font-semibold text-text">
              Your Prolific completion code
            </h2>
            <p className="mt-1 text-xs text-muted">
              Paste this into the Prolific submission page to mark the task as
              completed.
            </p>
            <code className="mt-3 inline-block rounded bg-panelAlt px-3 py-2 font-mono text-sm text-text">
              {completionCode}
            </code>
          </section>
        )}

        {summary && (
          <section aria-labelledby="ann-summary-heading" className="space-y-3">
            <h2 id="ann-summary-heading" className="text-sm font-semibold text-text">
              About the session you just rated
            </h2>
            <p className="text-sm text-muted">
              During annotation we didn't tell you which agent produced this
              conversation so your ratings would be unbiased. Here's the
              disclosure now that you're done.
            </p>

            <div className="rounded-lg border border-border bg-bg/60 p-4">
              <div className="text-sm font-semibold text-text">
                {summary.condition_label}
              </div>
              <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted sm:grid-cols-4">
                <div>
                  <div className="text-text">Task type</div>
                  <div>{summary.task_type}</div>
                </div>
                <div>
                  <div className="text-text">Split</div>
                  <div>{summary.task_split}</div>
                </div>
                <div>
                  <div className="text-text">Task index</div>
                  <div>{summary.task_idx}</div>
                </div>
                <div>
                  <div className="text-text">User turns</div>
                  <div>{summary.turn_count}</div>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-border bg-bg/40 p-4 text-sm">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted">
                Participant's task outcome
              </div>
              <div className="mt-1 font-medium text-text">
                {summary.participant_outcome.label}
              </div>
              <p className="mt-1 text-muted">{summary.participant_outcome.detail}</p>
            </div>
          </section>
        )}

        <footer className="border-t border-border pt-4 text-xs leading-relaxed text-muted">
          You can close this tab once you have your completion code. Thanks
          again for reviewing.
        </footer>
      </article>
    </main>
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
