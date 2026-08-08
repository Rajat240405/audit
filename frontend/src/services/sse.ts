// Thin re-export of the SSE reader so components depend on `services`
// rather than `api` directly (clean separation of concerns).
export { consumeSSE, ApiError } from "@/api/client";
export type { StreamHandlers } from "@/api/client";
