/**
 * Product tour — step content and targeting.
 *
 * Every quoted label is copied verbatim from the component that renders it, so
 * the tour cannot drift out of sync with a renamed control:
 *   Provider / Model / Build Graph / Activity        components/layout/Header.tsx
 *   Mode / Draft Style / Thinking Effort /
 *   Retrieval Mode                                   components/layout/Header.tsx
 *   Draft · History · Sources · RAG Pipeline ·
 *   Notes · Graph                                    pages/Workspace.tsx (TABS)
 *   Markdown / DOCX / TXT                            components/workspace/ExportMenu.tsx
 *   Cross-Verify Facts / Formalize / ✏️ Edit         components/workspace/DraftCanvas.tsx
 *   • Bullets ↑ Formal ⊖ Concise + Exec Summary
 *   ✓ Grammar ↔ Prose / Edit with AI                 components/workspace/Toolbar.tsx
 *   Cross-Verified Facts / grounded / unsupported    components/workspace/FactVerificationPanel.tsx
 *   Retrieving / Thinking / Generating               components/activity/ModelActivityPanel.tsx
 *   Newly processed / Reconciled / changed /
 *   Skipped unchanged / Failed                       components/graph/BuildGraphModal.tsx
 *
 * The tour explains *capabilities*, not widgets: what a feature does, what
 * changes when you use it, and when it is worth using.
 */

export type TourTab = "draft" | "history" | "sources" | "pipeline" | "notes" | "graph";

export type TourPlacement = "top" | "bottom" | "left" | "right";

export interface TourStep {
  id: string;
  title: string;
  /** 1–3 sentences. What it does + what changes. */
  body: string;
  /** Optional closing line, rendered under a "When to use it" label. */
  when?: string;
  /**
   * ``data-tour`` value(s) to highlight. Pass an array to list fallbacks —
   * the first one present in the DOM wins, which is how conditionally
   * rendered controls (e.g. Thinking Effort, Deep mode only) are handled.
   * Omit entirely for a centred card.
   */
  target?: string | string[];
  /** Workspace tab that must be open before this step can be shown. */
  tab?: TourTab;
  /** Preferred popover side; the overlay flips it when there is no room. */
  placement?: TourPlacement;
}

export const TOUR_STEPS: TourStep[] = [
  {
    id: "welcome",
    title: "Welcome to INCOIS Audit Pro",
    body:
      "This is a retrieval-augmented research workspace for INCOIS, Lok Sabha and Rajya Sabha records. " +
      "You ask an audit question; the system retrieves the underlying source documents and drafts a grounded, " +
      "citable answer. This tour covers the controls that shape that process — about two minutes.",
    when: "Skip at any point with Esc, and reopen it later from the Guide button in the header.",
  },
  {
    id: "provider-model",
    title: "Provider & Model",
    body:
      "Provider is the inference backend serving the language models; Model is the model family that reads the " +
      "retrieved evidence and writes your draft. Only the providers your deployment has enabled are listed. " +
      "Switching the model changes how the draft is written — it does not change what gets retrieved.",
    when: "Change the model when draft quality or response time is not what you need. The provider list is fixed by the backend.",
    target: "header-provider-model",
    placement: "bottom",
  },
  {
    id: "mode-thinking",
    title: "Mode & Thinking Effort",
    body:
      "Mode sets how much reasoning the model does before answering: Standard answers directly, Deep spends extra " +
      "reasoning steps on the question first. Thinking Effort — Low, Medium or XHigh — appears only in Deep mode " +
      "and controls how long that reasoning runs; Standard has no reasoning to steer, so the selector is hidden.",
    when: "Use Standard for straightforward lookups. Switch to Deep for multi-part or comparative questions, and raise Thinking Effort if the reasoning looks shallow.",
    // Thinking Effort is Deep-only, so fall back to the Mode selector.
    target: ["header-thinking-effort", "header-mode"],
    placement: "bottom",
  },
  {
    id: "retrieval-mode",
    title: "Retrieval Mode",
    body:
      "This chooses how evidence is found. Auto lets the query agent pick Hybrid, Graph or both. Hybrid RAG runs " +
      "dense + BM25 + rerank over document chunks. GraphRAG traverses the knowledge graph of entities and facts. " +
      "Hybrid + Graph runs both and merges the evidence.",
    when: "Leave it on Auto unless you are checking a specific kind of evidence — Hybrid RAG for verbatim document text, GraphRAG for relationships between entities.",
    target: "header-retrieval-mode",
    placement: "bottom",
  },
  {
    id: "draft-style",
    title: "Draft Style",
    body:
      "Sets the register of the generated draft. It starts on Parliamentary — the default, which mirrors a Lok Sabha " +
      "ministry reply — and can switch to Professional, Concise, Detailed or back to Default Tone. It affects wording " +
      "and structure only — the same evidence is retrieved either way. Your choice is remembered between visits.",
    when: "Leave it on Parliamentary for submission-ready wording; switch to Concise for a quick briefing note.",
    target: "header-draft-style",
    placement: "bottom",
  },
  {
    id: "tabs",
    title: "The workspace tabs",
    body:
      "Every session opens on Draft. History lists your saved sessions, Sources shows the retrieved evidence behind " +
      "the current answer, RAG Pipeline shows how that retrieval was performed, Notes is your own scratchpad, and " +
      "Graph visualises the knowledge graph for the current answer.",
    when: "Stay on Draft while you work; move to Sources or Graph when you want to inspect how a conclusion was reached.",
    target: "workspace-tabs",
    tab: "draft",
    placement: "bottom",
  },
  {
    id: "draft-editing",
    title: "Editing the draft",
    body:
      "The Drafting Canvas holds the generated answer, with a Grounding score showing the share of its claims that " +
      "were found in the retrieved evidence. The Edit button rewrites any part by hand; Save writes your version " +
      "back into the session transcript, so your edits persist and become the base for the next AI edit.",
    when: "Edit by hand for factual corrections or wording the model got wrong; use AI editing for restructuring.",
    target: "draft-canvas",
    tab: "draft",
    placement: "bottom",
  },
  {
    id: "ai-editing",
    title: "AI editing",
    body:
      "The chips restructure the draft in one pass — • Bullets, ↑ Formal, ⊖ Concise, + Exec Summary, ✓ Grammar and " +
      "↔ Prose — and the instruction box takes anything you type, run with Edit with AI. Nothing is applied " +
      "silently: the result opens in a side-by-side Compare panel where you choose Keep Original or Use Edited. " +
      "Formalize is a one-click shortcut for the formal-register rewrite.",
    when: "Use a chip for a predictable transform, and the instruction box when you need something specific like \"add a recommendations section\".",
    target: "draft-toolbar",
    tab: "draft",
    placement: "bottom",
  },
  {
    id: "verify",
    title: "Cross-Verify Facts",
    body:
      "This re-checks the finished draft against the evidence instead of trusting the generation. It extracts each " +
      "claim from the answer, looks for it in the retrieved sources, and reports the result in the Cross-Verified " +
      "Facts panel as a count of grounded and unsupported claims, listed one by one. The Grounding percentage in " +
      "the canvas header is the same calculation. It changes the score and the claim list — it never rewrites the draft.",
    when: "Run this before anything leaves your desk. Treat every unsupported claim as something to confirm in the original document before it is quoted.",
    target: "draft-verify",
    tab: "draft",
    placement: "bottom",
  },
  {
    id: "export",
    title: "Export",
    body:
      "Export writes the current draft out as Markdown, DOCX or TXT; Copy puts it on the clipboard. PDF and " +
      "government templates are listed as coming soon and are not available yet.",
    when: "DOCX for anything going into a formal submission, Markdown for pasting into another tool.",
    target: "workspace-actions",
    tab: "draft",
    placement: "bottom",
  },
  {
    id: "history-sessions",
    title: "History & sessions",
    body:
      "Every session is saved and listed here, labelled by its first query and sorted most recent first. Opening one " +
      "loads that session's conversation in the sidebar and its last answer on the canvas. Each entry can be renamed " +
      "or deleted, and New session starts a clean one. The pill at the top of the sidebar shows which session is open.",
    when: "Start a new session for each separate research task and return here to continue earlier work — sessions persist between visits.",
    target: "workspace-panel-history",
    tab: "history",
    placement: "bottom",
  },
  {
    id: "sources",
    title: "Sources & evidence",
    body:
      "The evidence base for the current answer. Each card carries a confidence label; click one to highlight the " +
      "passage it supports, or open the full source document to read it in context. Cards also show which retrieval " +
      "component found the passage — Dense, BM25, RRF and Rerank, each to three decimals, with the dominant one " +
      "emphasised — and a graph badge when the evidence came from the knowledge graph.",
    when: "Use it to check a specific claim before relying on it, and to see how strongly the retrieval components agreed on a passage.",
    target: "workspace-panel-sources",
    tab: "sources",
    placement: "bottom",
  },
  {
    id: "notes",
    title: "Notes",
    body:
      "A persistent scratchpad alongside your research — bookmarks, findings, things still needing verification, " +
      "feedback to send back. It has its own formatting toolbar and is saved in the browser, so it is still here the " +
      "next time you open the app.",
    when: "Keep open questions and follow-ups here while the draft is still being written, rather than in a separate file.",
    target: "workspace-panel-notes",
    tab: "notes",
    placement: "bottom",
  },
  {
    id: "pipeline",
    title: "RAG Pipeline & Performance",
    body:
      "The pipeline lists the retrieval stages as the backend runs them, so you can see what was actually done for " +
      "this query. Performance records the run: Model, Provider, Execution Profile, Response Time, Retrieved " +
      "Documents, Retrieved Chunks, Confidence and Context Used, plus per-stage retrieval timings.",
    when: "Check it when an answer looks thin — the retrieved document and chunk counts tell you whether the problem was retrieval or drafting.",
    target: "workspace-panel-pipeline",
    tab: "pipeline",
    placement: "bottom",
  },
  {
    id: "graph",
    title: "Graph",
    body:
      "A knowledge-graph view of the current answer's evidence: the entities extracted from the source documents, the " +
      "facts attached to them, and the documents they came from. Select a node to read it in full; Fit and Reset " +
      "control the layout. Evidence with graph provenance is badged as such in the Sources tab too.",
    when: "Useful when the question is about relationships — who reported what, which documents corroborate a fact — rather than about a single passage.",
    target: "workspace-panel-graph",
    tab: "graph",
    placement: "bottom",
  },
  {
    id: "activity",
    title: "Model Activity",
    body:
      "A live side drawer for the current query. It shows the documents the model received — each one openable in " +
      "full — its reasoning, and the answer streaming in, moving through Retrieving, Thinking and Generating. The " +
      "header button pulses while a phase is active.",
    when: "Open it on long Deep-mode runs to confirm the model is retrieving and reasoning rather than stalled.",
    target: "header-activity",
    placement: "bottom",
  },
  {
    id: "build-graph",
    title: "Build Graph",
    body:
      "Builds or refreshes the knowledge graph from the ingested corpus. The progress modal reports Status along with " +
      "Newly processed, Reconciled / changed, Skipped unchanged and Failed. Unchanged documents are skipped, so " +
      "re-running it is cheap.",
    when: "Run it after new documents are ingested, or when the Graph tab reports that the graph is unavailable.",
    target: "header-build-graph",
    placement: "bottom",
  },
];
