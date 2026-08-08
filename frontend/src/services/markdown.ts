// Markdown helpers for the drafting workspace.

/** Minimal heading extraction for a document outline. */
export function extractOutline(markdown: string): Array<{ level: number; text: string }> {
  const lines = markdown.split("\n");
  const outline: Array<{ level: number; text: string }> = [];
  for (const line of lines) {
    const m = line.match(/^(#{1,3})\s+(.*)$/);
    if (m) {
      outline.push({ level: m[1].length, text: m[2].trim() });
    }
  }
  return outline;
}

export function countWords(markdown: string): number {
  const text = markdown.replace(/[#*_`>\[\]()!-]/g, " ");
  return text.split(/\s+/).filter(Boolean).length;
}

export function wordCountLabel(markdown: string): string {
  return `${countWords(markdown).toLocaleString("en-IN")} words`;
}
