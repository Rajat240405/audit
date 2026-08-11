import { useRef, useEffect } from "react";
import {
  Bold,
  Italic,
  Heading1,
  Heading2,
  List,
  ListOrdered,
  Code,
  Undo2,
  Trash2,
} from "lucide-react";
import { useNotesStore } from "@/store/useNotesStore";
import { useToastStore } from "@/store/useToastStore";

/**
 * Notes workspace — a real rich-text editor (Word-like). Bold/Italic/Headings/
 * Lists apply the actual formatting (via contentEditable + execCommand), not
 * markdown markers. Notes persist to localStorage as HTML.
 */
export function NotesEditor() {
  const notes = useNotesStore((s) => s.notes);
  const setNotes = useNotesStore((s) => s.setNotes);
  const clear = useNotesStore((s) => s.clear);
  const pushToast = useToastStore((s) => s.push);
  const editorRef = useRef<HTMLDivElement>(null);

  // Sync store → DOM only when they diverge (initial hydration from
  // localStorage, clear, undo). We deliberately do NOT feed `notes` back
  // through dangerouslySetInnerHTML: re-rendering a contentEditable on every
  // keystroke resets its DOM and snaps the caret back to position 0, so each
  // typed character gets inserted at the start and the whole text comes out
  // reversed ("hello" → "olleh"). While typing, the browser owns the DOM;
  // this effect only applies external changes.
  useEffect(() => {
    const el = editorRef.current;
    if (!el) return;
    if (el.innerHTML !== notes) el.innerHTML = notes;
  }, [notes]);

  const exec = (command: string, value?: string) => {
    editorRef.current?.focus();
    document.execCommand(command, false, value);
    // sync the store from the editor content
    if (editorRef.current) setNotes(editorRef.current.innerHTML);
  };

  const btn =
    "flex h-7 w-7 items-center justify-center rounded hover:bg-surface-2 text-foreground/80";

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar — applies real formatting */}
      <div className="flex shrink-0 items-center gap-0.5 border-b border-border bg-surface-2/50 px-2 py-1.5">
        <button className={btn} onMouseDown={(e) => { e.preventDefault(); exec("bold"); }} title="Bold">
          <Bold className="h-3.5 w-3.5" />
        </button>
        <button className={btn} onMouseDown={(e) => { e.preventDefault(); exec("italic"); }} title="Italic">
          <Italic className="h-3.5 w-3.5" />
        </button>
        <span className="mx-1 h-4 w-px bg-border" />
        <button
          className={btn}
          onMouseDown={(e) => { e.preventDefault(); exec("formatBlock", "h1"); }}
          title="Heading 1"
        >
          <Heading1 className="h-3.5 w-3.5" />
        </button>
        <button
          className={btn}
          onMouseDown={(e) => { e.preventDefault(); exec("formatBlock", "h2"); }}
          title="Heading 2"
        >
          <Heading2 className="h-3.5 w-3.5" />
        </button>
        <span className="mx-1 h-4 w-px bg-border" />
        <button
          className={btn}
          onMouseDown={(e) => { e.preventDefault(); exec("insertUnorderedList"); }}
          title="Bullet list"
        >
          <List className="h-3.5 w-3.5" />
        </button>
        <button
          className={btn}
          onMouseDown={(e) => { e.preventDefault(); exec("insertOrderedList"); }}
          title="Numbered list"
        >
          <ListOrdered className="h-3.5 w-3.5" />
        </button>
        <span className="mx-1 h-4 w-px bg-border" />
        <button
          className={btn}
          onMouseDown={(e) => { e.preventDefault(); exec("formatBlock", "pre"); }}
          title="Code block"
        >
          <Code className="h-3.5 w-3.5" />
        </button>
        <div className="ml-auto flex items-center gap-0.5">
          <button className={btn} onClick={() => document.execCommand("undo")} title="Undo">
            <Undo2 className="h-3.5 w-3.5" />
          </button>
          <button
            className={`${btn} hover:text-danger`}
            onClick={() => {
              if (notes && window.confirm("Clear all notes?")) {
                clear();
                if (editorRef.current) editorRef.current.innerHTML = "";
                pushToast("success", "Notes cleared");
              }
            }}
            title="Clear all"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Rich-text editor — uncontrolled contentEditable. Keystrokes update
          the store via onInput; React never rewrites the innerHTML while
          typing, so the caret stays where you left it. */}
      <div
        ref={editorRef}
        contentEditable
        suppressContentEditableWarning
        onInput={(e) => setNotes((e.target as HTMLDivElement).innerHTML)}
        className="min-h-0 flex-1 overflow-y-auto bg-transparent px-4 py-3 text-sm leading-relaxed text-foreground focus:outline-none [&_h1]:text-xl [&_h1]:font-bold [&_h2]:text-lg [&_h2]:font-bold [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5 [&_pre]:bg-surface-2 [&_pre]:p-2 [&_pre]:rounded [&_pre]:font-mono [&_pre]:text-xs"
        data-placeholder="Bookmarks, important findings, needs-verification items, scientist feedback…"
      />
    </div>
  );
}
