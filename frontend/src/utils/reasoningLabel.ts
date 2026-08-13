/** Wait-text while the model is thinking. Never names qwen3 unless the live model is qwen3. */
export function reasoningWaitMessage(model: string | null | undefined): string {
  const name = (model || "").trim();
  if (!name) {
    return "Model is reasoning… (this can take a bit)";
  }
  const lower = name.toLowerCase();
  if (lower.includes("qwen3")) {
    return `Model is reasoning… (${name} thinks before answering; this can take a bit)`;
  }
  return `Model is reasoning… (${name}; this can take a bit)`;
}
