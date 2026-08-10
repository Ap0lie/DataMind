import { apiFetch } from "../../api-client";

export async function consumeAssistantEvents(
  runId: string,
  afterSequence: number,
  signal: AbortSignal,
  onEvent: (data: string) => void,
) {
  const response = await apiFetch(
    `/assistant/runs/${runId}/events?after_sequence=${afterSequence}`,
    { headers: { Accept: "text/event-stream" }, signal },
  );
  if (!response.ok || !response.body) {
    throw new Error(`Assistant event stream unavailable: ${response.status}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const eventName = frame
        .split(/\r?\n/)
        .find((line) => line.startsWith("event:"))
        ?.slice(6)
        .trim();
      const data = frame
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (eventName === "assistant" && data) onEvent(data);
      if (eventName === "end") return;
    }
  }
}
