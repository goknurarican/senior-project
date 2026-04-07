let ws: WebSocket | null = null;

export function initMarkers() {
  if (typeof window === "undefined") return;
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  ws = new WebSocket("ws://localhost:8765");
  ws.onopen = () => console.log("[markers] ws connected");
  ws.onerror = (e) => console.log("[markers] ws error", e);
}

export function mark(event: Record<string, any>) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ ...event, web_ts: Date.now() }));
}
