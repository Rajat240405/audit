/**
 * Matched-chunk display: presentation rules (pure) and the two components that
 * show them. The model never reads any of this; `answer` stays the parent
 * evidence. These tests pin what the reader sees.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { MatchedChunk, SourceItem } from "@/types";
import {
  hasMatchedSections,
  matchedTextUnits,
  matchedTotal,
  rerankBasis,
  scoreBasisLabel,
  unitTypeLabel,
} from "@/lib/matchedChunks";
import { DocViewerModal } from "@/components/evidence/DocViewerModal";
import { EvidenceCard } from "@/components/evidence/EvidenceCard";
import { useDocViewerStore } from "@/store/useDocViewerStore";
import { useDraftStore } from "@/store/useDraftStore";

const PARENT_EVIDENCE =
  "(a) Yes Sir. There are 33 stations.\nAnnexure-I\n1 WEST BENGAL BANKURA\n33 WEST BENGAL WEST MEDINIPUR";

function unit(over: Partial<MatchedChunk> & { chunk_id: string }): MatchedChunk {
  return {
    parent_doc_id: "17-7-2936",
    unit_type: "annexure",
    text: `text of ${over.chunk_id}`,
    dense_score: null,
    bm25_score: null,
    rrf_score: null,
    rerank_score: null,
    reranked: false,
    ...over,
  };
}

function source(over: Partial<SourceItem> = {}): SourceItem {
  return {
    doc_id: "17-7-2936",
    ministry: "EARTH SCIENCES",
    subject: "Modern AWS",
    score: 0.8,
    question: "Which stations are installed?",
    answer: PARENT_EVIDENCE,
    dense_score: 0.5,
    bm25_score: 14.3,
    rrf_score: 0.033,
    rerank_score: 8,
    ...over,
  };
}

beforeEach(() => {
  useDocViewerStore.setState({ source: null });
  useDraftStore.setState({ selectedEvidence: null } as never);
});

describe("presentation rules", () => {
  it("returns only units with stored text, in server order", () => {
    const s = source({
      matched_chunks: [
        unit({ chunk_id: "17-7-2936_L0", reranked: true }),
        unit({ chunk_id: "17-7-2936", unit_type: "parent", text: null }),
        unit({ chunk_id: "17-7-2936_L2", text: "  " }),
        unit({ chunk_id: "17-7-2936_L5" }),
      ],
    });
    expect(matchedTextUnits(s).map((m) => m.chunk_id)).toEqual([
      "17-7-2936_L0",
      "17-7-2936_L5",
    ]);
    expect(hasMatchedSections(s)).toBe(true);
  });

  it("reports no matched sections when metadata is missing or parent-only", () => {
    expect(hasMatchedSections(source())).toBe(false);
    expect(hasMatchedSections(source({ matched_chunks: [] }))).toBe(false);
    expect(
      hasMatchedSections(
        source({ matched_chunks: [unit({ chunk_id: "17-7-2936", unit_type: "parent", text: null })] }),
      ),
    ).toBe(false);
  });

  it("uses the server's real total when it exceeds the shown units", () => {
    const s = source({ matched_chunks: [unit({ chunk_id: "a" })], matched_chunk_total: 7 });
    expect(matchedTotal(s)).toBe(7);
    expect(matchedTotal(source())).toBe(0);
  });

  it("labels score bases explicitly and never guesses an unknown basis", () => {
    expect(scoreBasisLabel("parent")).toBe("parent document");
    expect(scoreBasisLabel("best_chunk")).toBe("best matched section");
    expect(scoreBasisLabel(null)).toBeNull();
    expect(scoreBasisLabel(undefined)).toBeNull();
  });

  it("attributes the rerank score to the unit that was actually reranked", () => {
    expect(
      rerankBasis(source({ matched_chunks: [unit({ chunk_id: "17-7-2936_L0", reranked: true })] })),
    ).toBe("best_chunk");
    expect(
      rerankBasis(source({ matched_chunks: [unit({ chunk_id: "17-7-2936", unit_type: "parent", text: null, reranked: true })] })),
    ).toBe("parent");
    expect(rerankBasis(source({ rerank_score: null, matched_chunks: [] }))).toBeNull();
    expect(rerankBasis(source())).toBeNull();
  });

  it("maps unit types to readable labels with a neutral fallback", () => {
    expect(unitTypeLabel("annexure")).toBe("Annexure section");
    expect(unitTypeLabel("v2_table")).toBe("Table");
    expect(unitTypeLabel("something_new")).toBe("Section");
  });
});

describe("DocViewerModal", () => {
  it("shows the matched section as primary and the parent evidence collapsed", () => {
    const s = source({
      matched_chunks: [unit({ chunk_id: "17-7-2936_L0", text: "MATCHED-CHUNK-TEXT", reranked: true })],
    });
    useDocViewerStore.setState({ source: s });
    render(<DocViewerModal />);

    expect(screen.getByText("MATCHED-CHUNK-TEXT")).toBeTruthy();
    expect(screen.getByText("17-7-2936_L0")).toBeTruthy();
    expect(screen.getByText("Top reranked")).toBeTruthy();
    const summary = screen.getByText("Parent evidence (as used for the answer)");
    const details = summary.closest("details");
    expect(details).not.toBeNull();
    expect(details!.open).toBe(false);
    // the parent text is still present (expandable), and distinct from the match
    expect(details!.textContent).toContain("33 WEST BENGAL WEST MEDINIPUR");
    expect(screen.queryByText("Full answer (as retrieved)")).toBeNull();
  });

  it("lists every matched section when several match, with the real total", () => {
    const s = source({
      matched_chunks: [
        unit({ chunk_id: "17-7-2936_L0", text: "FIRST-MATCH" }),
        unit({ chunk_id: "17-7-2936_L2", text: "SECOND-MATCH" }),
        unit({ chunk_id: "17-7-2936_L5", text: "THIRD-MATCH" }),
      ],
      matched_chunk_total: 9,
    });
    useDocViewerStore.setState({ source: s });
    render(<DocViewerModal />);

    expect(screen.getByText("FIRST-MATCH")).toBeTruthy();
    expect(screen.getByText("SECOND-MATCH")).toBeTruthy();
    expect(screen.getByText("THIRD-MATCH")).toBeTruthy();
    expect(screen.getByText(/showing 3 of 9/)).toBeTruthy();
  });

  it("falls back to the existing full-answer display when there is no matched text", () => {
    useDocViewerStore.setState({ source: source() });
    render(<DocViewerModal />);

    expect(screen.getByText("Full answer (as retrieved)")).toBeTruthy();
    expect(screen.getByText(/33 WEST BENGAL WEST MEDINIPUR/)).toBeTruthy();
    expect(screen.queryByText("Parent evidence (as used for the answer)")).toBeNull();
  });

  it("falls back when only the parent row matched (no matched text to show)", () => {
    useDocViewerStore.setState({
      source: source({
        matched_chunks: [unit({ chunk_id: "17-7-2936", unit_type: "parent", text: null })],
      }),
    });
    render(<DocViewerModal />);
    expect(screen.getByText("Full answer (as retrieved)")).toBeTruthy();
  });
});

describe("EvidenceCard score labels", () => {
  function renderCard(s: SourceItem) {
    return render(<EvidenceCard source={s} selected={false} isGraph={false} />);
  }

  it("marks chunk-derived summary scores and leaves parent scores unmarked", () => {
    renderCard(
      source({
        score_sources: { dense: "best_chunk", bm25: "parent", rrf: "best_chunk" },
        matched_chunks: [unit({ chunk_id: "17-7-2936_L0", reranked: true }), unit({ chunk_id: "17-7-2936_L2" })],
        matched_chunk_total: 2,
      }),
    );
    const dense = screen.getByText(/^Dense 0\.500/);
    expect(dense.textContent).toContain("chunk");
    expect(dense.getAttribute("title")).toBe("Score from the best matched section");
    const bm25 = screen.getByText(/^BM25 14\.300/);
    expect(bm25.textContent).not.toContain("chunk");
    expect(bm25.getAttribute("title")).toBe("Score from the parent document");
    expect(screen.getByText("2 matched sections")).toBeTruthy();
  });

  it("shows no basis label or match count for legacy sources", () => {
    renderCard(source());
    const dense = screen.getByText(/^Dense 0\.500/);
    expect(dense.textContent).not.toContain("chunk");
    expect(dense.getAttribute("title")).toBeNull();
    expect(screen.queryByText(/matched section/)).toBeNull();
  });

  it("opens the viewer for the same source (View full document still works)", () => {
    const s = source({ matched_chunks: [unit({ chunk_id: "17-7-2936_L0" })] });
    renderCard(s);
    fireEvent.click(screen.getByText("View full document"));
    expect(useDocViewerStore.getState().source?.doc_id).toBe("17-7-2936");
  });
});

describe("duplicate parent evidence", () => {
  it("omits the collapsed parent block when it repeats the matched section", () => {
    const text = "ANSWER: the same chunk text";
    useDocViewerStore.setState({
      source: source({
        answer: text,
        matched_chunks: [unit({ chunk_id: "17-7-2936_A", unit_type: "answer", text })],
      }),
    });
    render(<DocViewerModal />);
    expect(screen.queryByText("Parent evidence (as used for the answer)")).toBeNull();
    expect(screen.getByText("The matched section is the full evidence for this document.")).toBeTruthy();
  });

  it("keeps the collapsed parent block when it adds text beyond the match", () => {
    useDocViewerStore.setState({
      source: source({ matched_chunks: [unit({ chunk_id: "17-7-2936_L0", text: "just one part" })] }),
    });
    render(<DocViewerModal />);
    expect(screen.getByText("Parent evidence (as used for the answer)")).toBeTruthy();
  });
});
