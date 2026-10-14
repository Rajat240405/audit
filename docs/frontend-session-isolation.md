# Frontend session isolation, reload and Markdown links

This records the policy implemented for GitHub Issue #3. The code is the source
of truth. The key files are `frontend/src/lib/chatRun.ts`,
`frontend/src/lib/sessionTransitions.ts`, `frontend/src/lib/answerStatus.ts`,
`frontend/src/store/useSessionStore.ts`, `frontend/src/store/useEditStore.ts`,
and `frontend/src/hooks/useEditDraft.ts`.

## 1. Streaming isolation (chat A → chat B during a stream)

- **One run at a time, in any chat.** A send is refused while any answer is
  generating. The refusal is synchronous, so double clicks cannot start two
  runs. The typed text stays in the input, with a hint.
- **The run belongs to its chat.** Tokens, reasoning, status, phase, sources,
  trace, metadata, errors and completion are written
  to the session message the run started in. They are never written to the
  chat on screen.
- **The canvas, Model Activity and RAG Pipeline are a projection** of the
  active run. They show only the chat that the run belongs to. Switching away
  clears them. Switching back re-attaches the accumulated state, and the stream
  continues from there.
- **Stop** works from any chat, including from the other chat's input. The
  partial text is kept and labelled "stopped". Late events from the aborted
  stream are ignored.
- **Deleting** a chat that is generating stops its run first.
- **Edits** (AI edit, accept) are bound to the answer they started from. A
  switch or a new query cancels an edit in flight. A staged edit is applied
  only while the canvas still shows that same answer.

Not restored: the knowledge comparison panel is answer-scoped and is cleared on
switch, as before.

## 2. Reload, Back and New Session

- A reload or return to the page re-opens the **persisted active chat** if it
  still exists. Nothing is created on reload.
- **New session** creates a fresh, empty chat with a clean canvas. The previous
  chat stays in History.
- Leaving the page mid-answer (reload, close, navigation) is recorded as
  `interrupted` with the reload notice, not as an error. The browser's
  connection drop during page exit is not a generation failure.
- A message still marked `streaming` at load cannot be resumed. It is recovered
  as `interrupted`, with its partial text kept and labelled.
- While streaming, the partial text is written to the chat about once a second,
  and again on page hide. So a reload or crash keeps the text written so far.
- Every assistant message has a status: `streaming`, `done`, `stopped`,
  `error` or `interrupted` (see `MessageStatus`). A stream that ends without the
  backend's explicit `done` event is `interrupted`. It is never shown as a
  finished answer.
- **Limitation (unchanged policy):** `sources`, `trace` and `meta` are still
  not persisted (localStorage quota: source objects can be hundreds of KB). A
  reloaded answer shows its text, but the Sources and Graph tabs are empty until
  the next query.

## 3. External Markdown links

Shared by the chat transcript and the canvas (`components/common/markdownComponents.tsx`).

| Target | Rendering |
| --- | --- |
| `http(s)` to another origin | `target="_blank" rel="noopener noreferrer"`. The app session is not touched. |
| `http(s)` to this origin, or a relative path | Same-tab link (existing behaviour). |
| `#fragment` | In-page anchor. |
| `mailto:`, `tel:` | Contact link. |
| Anything else (`javascript:`, `data:`, `file:`, `vbscript:`, unparseable) | Not navigable. The text is shown as plain text. |

## 4. Known gaps and decisions

- Concurrent answers in different chats are not supported. A second send is
  refused while any answer runs. Stop is available from any chat.
- The backend stream protocol is unchanged. The only addition is client-side:
  `consumeSSE` reports whether the explicit `done` event arrived.
- Browser check: a headless-Chromium smoke run (30 checks) against the
  production build and a mock backend covered reload, Back, switching during a
  stream, refused sends, background completion, Stop, interrupted streams,
  reload mid-stream, New Session, and link attributes. The harness is saved as
  `/home/user/issue3_smoke/` and is not part of the repository.
- Tests: `frontend/src/__tests__` (run with `npm test` in `frontend/`). They
  cover the registry, the session transitions, the edit binding, the link
  rules, the SSE completion flag, and the input and message components.
