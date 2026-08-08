import { exportDocument } from "@/api/chat";

export type ExportFormat = "md" | "txt" | "docx";

export async function exportDraft(format: ExportFormat, title: string, content: string): Promise<void> {
  const blob = await exportDocument(format, sanitizeFilename(title), content);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${sanitizeFilename(title)}.${format}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function sanitizeFilename(name: string): string {
  return name.replace(/[^a-zA-Z0-9 _-]/g, "").trim().slice(0, 60) || "draft";
}
