/** Tiny POST-compatible SSE client built on fetch + ReadableStream.
 *
 * EventSource only supports GET; we need to POST a JSON body. This reader
 * parses `event:` + `data:` frames from the response stream and calls the
 * provided handler for each. Returns when the server closes the stream.
 */
export type SSEHandler = (event: string, data: unknown) => void | Promise<void>;

export async function postSSE(
  url: string,
  body: unknown,
  onEvent: SSEHandler,
  { signal }: { signal?: AbortSignal } = {}
): Promise<void> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`SSE: HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const event = parseFrame(frame);
        if (event) await onEvent(event.name, event.data);
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(frame: string): { name: string; data: unknown } | null {
  let name = "message";
  let dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith(":")) continue; // comment
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  const raw = dataLines.join("\n");
  if (!raw) return null;
  try {
    return { name, data: JSON.parse(raw) };
  } catch {
    return { name, data: raw };
  }
}
