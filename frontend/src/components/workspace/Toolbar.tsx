import { useState } from "react";
import { useDraftStore } from "@/store/useDraftStore";
import { useEditDraft } from "@/hooks/useEditDraft";

const CHIPS = [
  { label: "• Bullets", prompt: "Convert this draft into bullet points." },
  { label: "↑ Formal", prompt: "Rewrite in a formal register." },
  { label: "⊖ Concise", prompt: "Make this more concise." },
  { label: "+ Exec Summary", prompt: "Convert into an executive summary." },
  { label: "✓ Grammar", prompt: "Fix grammar and punctuation." },
  { label: "↔ Prose", prompt: "Rewrite as flowing prose." },
];

/** Docked AI editing tools: formatting chips + free-form instruction box. */
export function Toolbar({ editing }: { editing: boolean }) {
  const { edit } = useEditDraft();
  const hasContent = useDraftStore((s) => !!s.content);
  const [instruction, setInstruction] = useState("");

  const run = () => {
    const i = instruction.trim();
    if (!i) return;
    setInstruction("");
    edit(i);
  };

  return (
    <div className="space-y-3">
      <div className="flex gap-2 overflow-x-auto pb-1">
        {CHIPS.map((c) => (
          <button
            key={c.label}
            disabled={!hasContent || editing}
            onClick={() => edit(c.prompt)}
            className="flex shrink-0 items-center gap-1 rounded-full border border-border bg-surface-2 px-3 py-1 text-xs text-foreground/80 hover:bg-surface disabled:opacity-50"
          >
            {c.label}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-2 rounded-md border border-border bg-surface-2 p-1">
        <input
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          placeholder="Edit instruction: 'add bullets', 'make formal', 'shorten'..."
          className="flex-1 border-none bg-transparent px-2 py-1 text-sm text-foreground placeholder:text-muted focus:outline-none"
        />
        <button
          onClick={run}
          disabled={!hasContent || editing || !instruction.trim()}
          className="rounded-md bg-foreground px-4 py-1.5 text-xs font-bold uppercase text-background hover:opacity-90 disabled:opacity-50"
        >
          Edit with AI
        </button>
      </div>
    </div>
  );
}
