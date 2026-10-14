/**
 * Answer-scoped transient UI: the saved-knowledge comparison, the
 * Edit-with-AI comparison, and any staged or in-flight AI edit. These belong
 * to ONE answer, so they are closed on a new query and on every session or
 * answer switch. Kept in its own module so the run registry and the session
 * transitions can share it without importing each other.
 */
import { useDraftStore } from "@/store/useDraftStore";
import { cancelInFlightEdit, useEditStore } from "@/store/useEditStore";

export function clearAnswerScopedPanels(): void {
  useDraftStore.getState().clearAnswerScopedUi();
  // An edit started for the previous answer must not land on this one.
  cancelInFlightEdit();
  useEditStore.setState({ pendingEdit: null, editing: false, editingLabel: null });
}
