import { useState } from "react";
import { Loader2 } from "lucide-react";
import { useDraftStore } from "@/store/useDraftStore";
import { useEditDraft } from "@/hooks/useEditDraft";

// Each chip has a SPECIFIC transformation prompt so the outputs are clearly
// distinguishable (not near-identical like before). The prompt tells the LLM
// exactly what to change and what to preserve.
const CHIPS = [
  {
    label: "• Bullets",
    prompt:
      "Reformat the draft into a bulleted list. Keep EVERY fact, name, figure, and citation — only change the formatting to bullets. One bullet per key point. Do not add or remove content.",
  },
  {
    label: "↑ Formal",
    prompt:
      "Rewrite the draft in a formal, official register (third-person, passive, ministerial tone). Keep ALL facts, names, figures, dates, and [Source N] citations exactly. Only change the wording to be more formal — do not change the meaning or structure.",
  },
  {
    label: "⊖ Concise",
    prompt:
      "Make the draft concise by REMOVING secondary details, repetition, and filler. Keep ONLY the essential facts, names, figures, dates, and [Source N] citations. Cut it to roughly half the length. Do not add new content.",
  },
  {
    label: "+ Exec Summary",
    prompt:
      "Convert the draft into a genuine executive summary: 3-5 bullet points capturing ONLY the most important findings, decisions, and key numbers. Drop all supporting detail. Keep the [Source N] citations for the key claims. This must be MUCH shorter than the original.",
  },
  {
    label: "✓ Grammar",
    prompt:
      "Correct ONLY grammar, spelling, and punctuation errors. CRITICAL: do NOT summarize, compress, add bullet points, remove bullet points, change the number of points, or reorganize. Reproduce the draft with the SAME structure, SAME number of sentences/points, SAME facts, names, figures, and [Source N] citations. If the draft has no grammar errors, output it nearly unchanged. The result must be almost the same length and layout as the input — only the language is corrected.",
  },
  {
    label: "↔ Prose",
    prompt:
      "Convert the draft into natural, continuous prose paragraphs (no bullets, no lists). Merge the content into flowing sentences. Keep ALL facts, names, figures, dates, and [Source N] citations. Only change the format from lists to prose — do not change the content.",
  },
];

/** Docked AI editing tools: formatting chips + free-form instruction box. */
export function Toolbar({ editing: _propEditing }: { editing: boolean }) {
  // Use OUR OWN hook instance so editing/editingLabel state is local to this
  // component (the prop from DraftCanvas is a separate instance and was
  // breaking the spinner).
  const { edit, editing, editingLabel, cancel } = useEditDraft();
  const hasContent = useDraftStore((s) => !!s.content);
  const [instruction, setInstruction] = useState("");

  const run = () => {
    const i = instruction.trim();
    if (!i) return;
    setInstruction("");
    edit(i, "Edit with AI");
  };

  return (
    <div className="space-y-3">
      <div className="flex gap-2 overflow-x-auto pb-1">
        {CHIPS.map((c) => {
          const isActive = editing && editingLabel === c.label;
          return (
            <button
              key={c.label}
              disabled={!hasContent || editing}
              onClick={() => edit(c.prompt, c.label)}
              className="flex shrink-0 items-center gap-1 rounded-full border border-border bg-surface-2 px-3 py-1 text-xs text-foreground/80 hover:bg-surface disabled:opacity-50"
            >
              {isActive && <Loader2 className="h-3 w-3 animate-spin" />}
              {c.label}
            </button>
          );
        })}
      </div>

      <div className="flex items-center gap-2 rounded-md border border-border bg-surface-2 p-1">
        <input
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          placeholder="Edit instruction: 'add bullets', 'make formal', 'shorten'..."
          className="flex-1 border-none bg-transparent px-2 py-1 text-sm text-foreground placeholder:text-muted focus:outline-none"
        />
        {editing ? (
          <button
            onClick={cancel}
            className="flex items-center gap-1 rounded-md bg-danger px-4 py-1.5 text-xs font-bold uppercase text-white hover:opacity-90"
          >
            <Loader2 className="h-3 w-3 animate-spin" />
            Stop
          </button>
        ) : (
          <button
            onClick={run}
            disabled={!hasContent || !instruction.trim()}
            className="rounded-md bg-foreground px-4 py-1.5 text-xs font-bold uppercase text-background hover:opacity-90 disabled:opacity-50"
          >
            Edit with AI
          </button>
        )}
      </div>
    </div>
  );
}
