import { useCallback, useEffect, useRef, useState } from "react";
import { Mic, MicOff, Send, Square } from "lucide-react";
import { useDraftStore } from "@/store/useDraftStore";
import { useSessionStore } from "@/store/useSessionStore";
import { cn } from "@/utils/cn";

interface ChatInputProps {
  onSend: (q: string) => void;
  onStop: () => void;
  streaming: boolean;
}

/** Height the audit-query textarea auto-grows to before it scrolls. */
const TEXTAREA_MAX = 220;

/**
 * Query input docked in the left sidebar (matches the Stitch design).
 * Mode (Standard/Deep), draft style, and Hybrid/GraphRAG selection live in
 * the header — this stays a pure query box. Supports speech-to-text: the mic
 * button transcribes into the textarea (never auto-submits) so the user can
 * edit before sending.
 */
export function ChatInput({ onSend, onStop, streaming }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const resetDraft = useDraftStore((s) => s.reset);
  const clearMessages = useSessionStore((s) => s.clearMessages);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const turns = useSessionStore(
    (s) => s.sessions.find((x) => x.id === s.activeSessionId)?.messages.length ?? 0
  );

  // ── auto-grow textarea ────────────────────────────────────────────────────
  const autosize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, TEXTAREA_MAX) + "px";
  }, []);

  useEffect(() => {
    autosize();
  }, [value, autosize]);

  // ── speech-to-text ────────────────────────────────────────────────────────
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const speechBaseRef = useRef("");
  const [listening, setListening] = useState(false);
  const [speechError, setSpeechError] = useState<string | null>(null);

  const SpeechRecognitionCtor =
    typeof window !== "undefined"
      ? window.SpeechRecognition ?? window.webkitSpeechRecognition
      : undefined;

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
    recognitionRef.current = null;
    setListening(false);
  }, []);

  const startListening = useCallback(() => {
    setSpeechError(null);
    if (!SpeechRecognitionCtor) {
      setSpeechError("Speech recognition is not supported in this browser.");
      return;
    }
    try {
      const rec = new SpeechRecognitionCtor();
      recognitionRef.current = rec;
      rec.lang = "en-IN";
      rec.continuous = false;
      rec.interimResults = true;
      rec.maxAlternatives = 1;
      speechBaseRef.current = value;

      rec.onresult = (event: SpeechRecognitionEvent) => {
        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const r = event.results[i];
          transcript += r[0].transcript;
        }
        // append the recognized speech to whatever the user already typed
        const base = speechBaseRef.current.trim();
        setValue(base ? `${base} ${transcript.trim()}`.trim() : transcript.trim());
      };

      rec.onerror = (event: SpeechRecognitionErrorEvent) => {
        setListening(false);
        if (event.error === "not-allowed" || event.error === "service-not-allowed") {
          setSpeechError("Microphone permission was denied. Enable it and try again.");
        } else if (event.error === "no-speech") {
          setSpeechError("No speech was detected. Please try again.");
        } else if (event.error === "audio-capture") {
          setSpeechError("No microphone was found on this device.");
        } else {
          setSpeechError("Speech recognition failed. Please try again.");
        }
      };

      rec.onend = () => {
        recognitionRef.current = null;
        setListening(false);
      };

      rec.start();
      setListening(true);
    } catch {
      setSpeechError("Could not start speech recognition. Please try again.");
      setListening(false);
    }
  }, [SpeechRecognitionCtor, value]);

  // ensure we stop the recognizer if the component unmounts mid-recording
  useEffect(() => {
    return () => {
      recognitionRef.current?.abort();
      recognitionRef.current = null;
    };
  }, []);

  const submit = () => {
    const q = value.trim();
    if (!q || streaming) return;
    setValue("");
    setSpeechError(null);
    onSend(q);
  };

  // Clear the whole conversation memory + reset the draft (the top "clear"
  // link next to "Memory: N turns").
  const clearAll = () => {
    resetDraft();
    if (activeSessionId) clearMessages(activeSessionId);
  };

  // Clear ONLY the current input textarea (the CLEAR button in the input row).
  // This must not delete the conversation/history.
  const clearInput = () => {
    setValue("");
    setSpeechError(null);
    textareaRef.current?.focus();
  };

  const micTitle = SpeechRecognitionCtor
    ? listening
      ? "Stop listening"
      : "Speak your query"
    : "Speech recognition not supported in this browser";

  return (
    <div className="border-t border-border bg-background p-3">
      <div className="mb-1.5 flex items-center justify-between text-[11px] text-muted">
        <div className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-gray-400" />
          <span>Memory: {turns} turns</span>
        </div>
        <button className="text-accent hover:underline" onClick={clearAll}>
          clear
        </button>
      </div>

      <div className="rounded-xl border border-border bg-surface p-3 shadow-sm">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={4}
          placeholder="Enter audit query... (or use the mic)"
          className="w-full resize-none overflow-y-auto border-none bg-transparent p-0 text-sm text-foreground placeholder:text-muted focus:outline-none"
          style={{ minHeight: "6rem" }}
        />

        {speechError && (
          <p className="mt-1.5 text-[11px] text-danger" role="alert">
            {speechError}
          </p>
        )}

        <div className="mt-2 flex items-center justify-between border-t border-border pt-2">
          <span className="text-[11px] text-muted">{turns} turns</span>
          <div className="flex items-center gap-2">
            <button
              className="text-[11px] font-semibold text-muted hover:text-foreground"
              onClick={clearInput}
            >
              CLEAR
            </button>
            <button
              onClick={listening ? stopListening : startListening}
              title={micTitle}
              aria-label={listening ? "Stop listening" : "Start speech input"}
              aria-pressed={listening}
              className={cn(
                "flex h-7 w-7 items-center justify-center rounded-full border transition-colors",
                listening
                  ? "animate-pulse border-danger bg-danger/15 text-danger"
                  : "border-border text-muted hover:bg-surface-2 hover:text-foreground"
              )}
            >
              {listening ? <MicOff className="h-3.5 w-3.5" /> : <Mic className="h-3.5 w-3.5" />}
            </button>
            {streaming ? (
              <button
                onClick={onStop}
                className="flex h-7 w-7 items-center justify-center rounded-full bg-danger text-white"
                title="Stop"
              >
                <Square className="h-3.5 w-3.5" />
              </button>
            ) : (
              <button
                onClick={submit}
                disabled={!value.trim()}
                className={cn(
                  "flex h-7 w-7 items-center justify-center rounded-full bg-foreground text-background",
                  value.trim() ? "disabled:opacity-40" : "opacity-40"
                )}
                title="Send"
              >
                <Send className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
