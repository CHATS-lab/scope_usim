"use client";

import { useState } from "react";
import type { SurveySchema } from "@/lib/api";

interface Props {
  schema: SurveySchema;
  onSubmit: (
    responses: Record<string, unknown>,
    freeText: string
  ) => Promise<void> | void;
}

export function SurveyPanel({ schema, onSubmit }: Props) {
  const [responses, setResponses] = useState<Record<string, unknown>>({});
  const [freeText, setFreeText] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const complete = schema.items.every(
    (it) => responses[it.key] !== undefined && responses[it.key] !== null
  );

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!complete || submitting) return;
    setSubmitting(true);
    try {
      await onSubmit(responses, freeText);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="mx-auto max-w-2xl space-y-6 px-6 py-8 text-sm"
    >
      <header>
        <h2 className="text-xl font-semibold text-text">{schema.title}</h2>
        <p className="text-muted">
          Please answer each item based on your experience in the conversation you just
          had.
        </p>
      </header>

      {schema.items.map((item) => (
        <div key={item.key} className="space-y-2">
          <label className="block text-text">{item.prompt}</label>
          {item.kind === "likert" && (
            <LikertRow
              min={item.min ?? 1}
              max={item.max ?? 7}
              value={responses[item.key] as number | undefined}
              onChange={(v) => setResponses((r) => ({ ...r, [item.key]: v }))}
            />
          )}
          {item.kind === "number" && (
            <input
              type="number"
              min={item.min}
              max={item.max}
              value={(responses[item.key] as number | undefined) ?? ""}
              onChange={(e) =>
                setResponses((r) => ({
                  ...r,
                  [item.key]: e.target.value === "" ? null : Number(e.target.value),
                }))
              }
              className="w-32 rounded bg-panelAlt px-2 py-1 ring-1 ring-border focus:ring-accent"
              required
            />
          )}
          {item.kind === "text" && (
            <textarea
              rows={3}
              value={(responses[item.key] as string | undefined) ?? ""}
              onChange={(e) =>
                setResponses((r) => ({ ...r, [item.key]: e.target.value }))
              }
              className="w-full rounded bg-panelAlt px-2 py-2 ring-1 ring-border focus:ring-accent"
              required
            />
          )}
        </div>
      ))}

      <div className="space-y-2">
        <label className="block text-text">Anything else you want to share?</label>
        <textarea
          rows={4}
          value={freeText}
          onChange={(e) => setFreeText(e.target.value)}
          className="w-full rounded bg-panelAlt px-2 py-2 ring-1 ring-border focus:ring-accent"
          placeholder="Optional"
        />
      </div>

      <button
        type="submit"
        disabled={!complete || submitting}
        className="rounded-lg bg-accent px-4 py-2 font-medium text-bg disabled:opacity-50"
      >
        {submitting ? "Submitting..." : "Submit survey"}
      </button>
    </form>
  );
}

function LikertRow({
  min,
  max,
  value,
  onChange,
}: {
  min: number;
  max: number;
  value: number | undefined;
  onChange: (v: number) => void;
}) {
  const options = Array.from({ length: max - min + 1 }, (_, i) => min + i);
  return (
    <div className="flex items-center gap-2">
      {options.map((v) => (
        <button
          key={v}
          type="button"
          onClick={() => onChange(v)}
          className={`h-9 w-9 rounded-full text-xs font-medium transition ${
            value === v
              ? "bg-accent text-bg"
              : "bg-panelAlt text-text hover:bg-border"
          }`}
        >
          {v}
        </button>
      ))}
    </div>
  );
}
