export async function withRetry<T>(
  fn: () => Promise<T>,
  {
    attempts = 3,
    baseDelayMs = 600,
    shouldRetry = () => true,
  }: {
    attempts?: number;
    baseDelayMs?: number;
    shouldRetry?: (err: unknown, attemptIdx: number) => boolean;
  } = {}
): Promise<T> {
  let lastErr: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (i === attempts - 1 || !shouldRetry(err, i)) break;
      const jitter = Math.random() * 200;
      await new Promise((r) => setTimeout(r, baseDelayMs * Math.pow(2, i) + jitter));
    }
  }
  throw lastErr;
}
