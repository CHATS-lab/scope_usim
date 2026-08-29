export type Condition = "base" | "rl_single" | "cotraining";
export type TaskType = "tau2" | "p4g";
export type SessionStatus = "active" | "stopped" | "survey_done" | "abandoned";
export type StudyVariant = "v1" | "v2";

export interface SessionStartResponse {
  session_id: string;
  condition: Condition;
  task_type: TaskType;
  task_split: string;
  task_idx: number;
  task_instruction: string;
  task_metadata: Record<string, unknown>;
  task_structured?: Record<string, unknown> | null;
  variant: StudyVariant;
  resumed: boolean;
  already_completed?: boolean;
  completion_code?: string | null;
  prolific_completion_code?: string | null;
  prolific_redirect_url?: string | null;
  debrief?: DebriefInfo | null;
}

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface ChatMessage {
  role: "assistant" | "user" | "tool";
  content: string | null;
  tool_calls?: ToolCall[] | null;
  tool_call_id?: string | null;
  tool_name?: string | null;
}

export interface ChatResponse {
  messages: ChatMessage[];
  session_status: SessionStatus;
}

export interface StopResponse {
  session_status: SessionStatus;
  survey_schema: SurveySchema;
}

export interface SurveyItem {
  key: string;
  prompt: string;
  kind: "likert" | "text" | "number";
  min?: number;
  max?: number;
  reverse_coded?: boolean;
}

export interface SurveySchema {
  title: string;
  items: SurveyItem[];
}

const BASE = "/api";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const raw = await res.text();
    // FastAPI HTTPException returns {detail: "..."} — surface that directly
    // instead of "HTTP 409 {detail: ...}" noise.
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.detail === "string") throw new Error(parsed.detail);
    } catch (e) {
      if (e instanceof Error && e.message && e.message !== raw) throw e;
    }
    throw new Error(raw || `Request failed (${res.status})`);
  }
  return (await res.json()) as T;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const raw = await res.text();
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.detail === "string") throw new Error(parsed.detail);
    } catch (e) {
      if (e instanceof Error && e.message && e.message !== raw) throw e;
    }
    throw new Error(raw || `Request failed (${res.status})`);
  }
  return (await res.json()) as T;
}

export const api = {
  startSession: (p: {
    prolific_pid: string;
    study_id: string;
    prolific_session_id: string;
    task_type: TaskType;
    variant?: StudyVariant;
  }) => post<SessionStartResponse>("/session/start", p),

  getTurns: (session_id: string) =>
    get<ChatMessage[]>(`/session/${session_id}/turns`),

  chat: (p: { session_id: string; user_message: string }) =>
    post<ChatResponse>("/chat", p),

  stop: (p: { session_id: string }) => post<StopResponse>("/stop", p),

  survey: (p: {
    session_id: string;
    responses: Record<string, unknown>;
    free_text?: string | null;
  }) => post<SurveyCompleteResponse>("/survey", p),

  annotationNext: (annotator_id: string, session_id?: string | null) => {
    const qs = new URLSearchParams({ annotator_id });
    if (session_id) qs.set("session_id", session_id);
    return get<AnnotationNextResponse>(`/annotation/next?${qs.toString()}`);
  },

  submitAnnotation: (p: {
    annotator_id: string;
    session_id: string;
    responses: Record<string, unknown>;
    free_text?: string | null;
  }) => post<AnnotationSubmitResponse>("/annotation", p),
};

export interface AnnotationNextResponse {
  session_id: string;
  task_type: TaskType;
  task_split: string;
  task_idx: number;
  task_instruction: string;
  turn_count: number;
  messages: ChatMessage[];
  survey_schema: SurveySchema;
  annotations_done: number;
  annotations_available: number;
  done: boolean;
}

export interface AnnotationSubmitResponse {
  completion_code: string | null;
  next_available: boolean;
  debrief: AnnotatorDebrief | null;
}

export interface AnnotatorDebrief {
  annotator_id: string;
  total_annotated: number;
  session_summary: AnnotatedSessionSummary | null;
}

export interface AnnotatedSessionSummary {
  session_id: string;
  task_type: TaskType;
  task_split: string;
  task_idx: number;
  turn_count: number;
  condition: Condition;
  condition_label: string;
  participant_outcome: TaskOutcome;
}

export interface DebriefInfo {
  condition: Condition;
  condition_label: string;
  condition_description: string;
  task_type: TaskType;
  task_split: string;
  task_idx: number;
  turn_count: number;
  completed: boolean;
  task_outcome: TaskOutcome;
}

export interface TaskOutcome {
  status: "success" | "partial" | "failure" | "not_evaluated";
  label: string;
  detail: string;
}

export interface SurveyCompleteResponse {
  completion_code: string;
  prolific_completion_code?: string | null;
  prolific_redirect_url?: string | null;
  debrief: DebriefInfo;
}

export const CHAT_STREAM_URL = `${BASE}/chat/stream`;
