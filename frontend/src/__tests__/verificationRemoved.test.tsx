/**
 * Regression guard: the Cross-Verified Facts feature stays removed from the UI
 * and the API client. The canvas must offer no verification control, and the
 * chat API must not export the verification or investigation calls.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import * as chatApi from "@/api/chat";
import { DraftCanvas } from "@/components/workspace/DraftCanvas";
import { useDraftStore } from "@/store/useDraftStore";
import { resetAppState } from "@/test/harness";

beforeEach(() => {
  resetAppState();
  // jsdom has no layout engine; the canvas auto-follows the stream with
  // scrollIntoView, so stub it for this file only.
  if (!Element.prototype.scrollIntoView) {
    Element.prototype.scrollIntoView = () => {};
  }
});

describe("Cross-Verified Facts is removed", () => {
  it("the chat API no longer exports verifyAnswer or investigateClaim", () => {
    expect("verifyAnswer" in chatApi).toBe(false);
    expect("investigateClaim" in chatApi).toBe(false);
  });

  it("the canvas shows no Cross-Verify control or verification text", () => {
    useDraftStore.setState({
      content: "The grant was approved in 2024 [1].",
      sources: [],
    });
    const { container } = render(<DraftCanvas />);

    // Positive control: the canvas really rendered, including its remaining
    // editing tool, so the absence checks below are not vacuous.
    expect(screen.getByText("Formalize")).toBeTruthy();
    expect(screen.getByText("The grant was approved in 2024 [1].", { exact: false })).toBeTruthy();

    const text = container.textContent ?? "";
    expect(text).not.toMatch(/cross-?verif/i);
    expect(text).not.toMatch(/grounding/i);
    expect(text).not.toMatch(/facts? supported/i);
    expect(screen.queryByRole("button", { name: /verify/i })).toBeNull();
  });

  it("the draft store holds no verification state", () => {
    const state = useDraftStore.getState() as unknown as Record<string, unknown>;
    for (const key of ["grounding", "groundingFor", "factsOpen", "investigations", "investigating", "deepVerify"]) {
      expect(key in state, `draft store still has ${key}`).toBe(false);
    }
  });
});
