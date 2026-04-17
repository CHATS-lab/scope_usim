export type Condition = "base" | "rl_single" | "cotraining";
export type TaskType = "tau2" | "p4g";
export type SessionStatus = "active" | "stopped" | "survey_done" | "abandoned";

export interface SessionStartResponse {
  session_id: string;
  condition: Condition;
  task_type: TaskType;
  task_split: string;
  task_idx: number;
  task_instruction: string;
  task_metadata: Record<string, unknown>;
  resumed: boolean;
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
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return (await res.json()) as T;
}

export const api = {
  startSession: (p: {
    prolific_pid: string;
    study_id: string;
    prolific_session_id: string;
    task_type: TaskType;
  }) => post<SessionStartResponse>("/session/start", p),

  chat: (p: { session_id: string; user_message: string }) =>
    post<ChatResponse>("/chat", p),

  stop: (p: { session_id: string }) => post<StopResponse>("/stop", p),

  survey: (p: {
    session_id: string;
    responses: Record<string, unknown>;
    free_text?: string | null;
  }) => post<{ completion_code: string }>("/survey", p),
};
