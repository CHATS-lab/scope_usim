"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

export default function Landing() {
  const params = useSearchParams();
  const router = useRouter();
  const [taskType, setTaskType] = useState<"tau2" | "p4g">("tau2");

  useEffect(() => {
    // If Prolific params are present, preserve them when navigating into the study.
    const hasPid = params.get("PROLIFIC_PID");
    if (hasPid) {
      const qs = new URLSearchParams(params.toString());
      qs.set("task_type", taskType);
      router.replace(`/study?${qs.toString()}`);
    }
  }, [params, router, taskType]);

  return (
    <main className="flex min-h-screen items-center justify-center px-6">
      <div className="w-full max-w-lg space-y-6 rounded-xl border border-border bg-panel p-8">
        <div>
          <h1 className="text-2xl font-semibold text-text">USIM Dialogue Study</h1>
          <p className="mt-2 text-sm text-muted">
            You have reached the participant interface for the USIM multi-turn dialogue
            study. Please open this page from the Prolific link you received so that your
            Prolific ID is attached to your session.
          </p>
        </div>

        <div className="rounded-lg bg-panelAlt p-4 text-sm text-muted">
          Expected URL parameters: <code className="font-mono text-text">PROLIFIC_PID</code>,{" "}
          <code className="font-mono text-text">STUDY_ID</code>,{" "}
          <code className="font-mono text-text">SESSION_ID</code>.
        </div>
      </div>
    </main>
  );
}
