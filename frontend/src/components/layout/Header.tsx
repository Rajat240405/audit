import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Moon, Sun, Brain, Settings as SettingsIcon } from "lucide-react";
import { fetchModels, fetchProviders, setProvider } from "@/api/model";
import { useAppStore } from "@/store/useAppStore";
import { SourceFilter } from "./SourceFilter";
import { useThemeStore } from "@/store/useThemeStore";
import { useActivityStore } from "@/store/useActivityStore";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/utils/cn";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ExecutionMode } from "@/types";

const MODES: Array<{ value: ExecutionMode; label: string }> = [
  { value: "fast", label: "Standard" },
  { value: "deep", label: "Deep" },
];

const DRAFT_STYLES = [
  { value: "default", label: "Default Tone" },
  { value: "professional", label: "Professional" },
  { value: "parliamentary", label: "Parliamentary" },
  { value: "concise", label: "Concise" },
  { value: "detailed", label: "Detailed" },
];

export function Header() {
  const app = useAppStore();

  // The deployment's ENABLED providers come from the backend (/api/providers)
  // — never hardcoded. A single-provider deployment (PC=ollama, HPC=vllm)
  // renders exactly that one option.
  const { data: providers } = useQuery({
    queryKey: ["providers"],
    queryFn: fetchProviders,
    staleTime: 300_000,
  });

  const { data: models } = useQuery({
    queryKey: ["models", app.provider],
    queryFn: () => fetchModels(app.provider),
    enabled: Boolean(app.provider),
    staleTime: 60_000,
  });

  const changeProvider = async (p: string) => {
    app.setProvider(p);
    try {
      // Auto-select the first model family of the new provider so the backend
      // never ends up with a mismatched (old-provider) family id.
      const fams = await fetchModels(p);
      const first = fams[0];
      if (first) {
        app.setModelFamily(first.id);
        app.setModel(first.model_name);
        await setProvider(p, first.id);
      }
    } catch {
      /* backend may be offline */
    }
  };

  // If the locally-stored provider isn't enabled in this deployment
  // (e.g. "ollama" default on an HPC vLLM backend), adopt the active one.
  useEffect(() => {
    if (!providers || providers.length === 0) return;
    if (!providers.some((p) => p.name === app.provider)) {
      const preferred = providers.find((p) => p.active) ?? providers[0];
      void changeProvider(preferred.name);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [providers]);

  return (
    <header className="flex shrink-0 items-center gap-3 border-b border-border bg-surface px-4 py-2">
      <div className="flex shrink-0 items-center gap-2 text-lg font-bold tracking-wide">
        <span className="text-accent">◆</span>
        <span className="text-sm">INCOIS AUDIT PRO</span>
      </div>

      <div className="flex flex-1 items-center justify-center gap-3 text-sm">
        <Select value={app.provider} onValueChange={changeProvider}>
          <SelectTrigger className="w-28">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {(providers ?? []).map((p) => (
              <SelectItem key={p.name} value={p.name}>
                {p.label ?? p.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select
          value={app.modelFamily}
          onValueChange={(f) => {
            app.setModelFamily(f);
            const fam = models?.find((m) => m.id === f);
            if (fam) {
              app.setModel(fam.model_name);
              // tell the BACKEND which model is active (this was missing —
              // the backend kept the default qwen2.5:7b no matter what we
              // selected in the UI)
              void setProvider(app.provider, f).catch(() => {
                /* backend may be offline */
              });
            }
          }}
        >
          <SelectTrigger className="w-40">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {(models ?? []).map((m) => (
              <SelectItem key={m.id} value={m.id}>
                {m.display_name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button size="sm" onClick={() => app.setBuildModalOpen(true)}>
          Build Graph
        </Button>

        {/* Model Activity — shows what the LLM received + its live reasoning */}
        <ActivityButton />

        <GpuBadge gpu={app.gpu} />

        <Select value={app.mode} onValueChange={(m) => app.setMode(m as ExecutionMode)}>
          <SelectTrigger className="w-24">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {MODES.map((m) => (
              <SelectItem key={m.value} value={m.value}>
                {m.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select
          value={app.draftStyle}
          onValueChange={(s) => app.setDraftStyle(s as typeof app.draftStyle)}
        >
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {DRAFT_STYLES.map((s) => (
              <SelectItem key={s.value} value={s.value}>
                {s.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Retrieval mode — Hybrid RAG vs GraphRAG (was the chatbox pill) */}
        <div className="flex items-center rounded-full border border-border bg-surface-2 p-0.5 text-[10px] font-semibold">
          {(["hybrid", "graph"] as const).map((m) => (
            <button
              key={m}
              onClick={() => app.setRetrievalMode(m)}
              className={cn(
                "rounded-full px-2.5 py-1 transition-colors",
                app.retrievalMode === m
                  ? "bg-foreground text-background"
                  : "text-muted hover:text-foreground"
              )}
              title={m === "hybrid" ? "Hybrid RAG: dense + BM25 + rerank" : "GraphRAG: knowledge-graph traversal"}
            >
              {m === "hybrid" ? "Hybrid RAG" : "GraphRAG"}
            </button>
          ))}
        </div>

        {/* Source filter — parliament / INCOIS reports / MoES reports / combo */}
        <SourceFilter />
      </div>

      <div className="flex shrink-0 items-center gap-1">
        <SettingsButton />
        <ThemeToggle />
      </div>
    </header>
  );
}

function SettingsButton() {
  const setSettingsOpen = useAppStore((s) => s.setSettingsOpen);
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setSettingsOpen(true)}
      title="Settings — ingest documents, backend info, advanced"
    >
      <SettingsIcon className="h-4 w-4" />
    </Button>
  );
}


function ActivityButton() {
  const open = useActivityStore((s) => s.open);
  const phase = useActivityStore((s) => s.phase);
  const toggle = useActivityStore((s) => s.toggle);
  const active = phase !== "idle" && phase !== "done" && phase !== "error";
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={toggle}
      title="Model activity — what the model received and what it's thinking"
      className={cn("relative", open && "bg-surface-2")}
    >
      <Brain className="h-4 w-4" />
      <span className="ml-1 hidden text-xs lg:inline">Activity</span>
      {active && (
        <span className="absolute right-1 top-1 flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
        </span>
      )}
    </Button>
  );
}

function GpuBadge({ gpu }: { gpu: string }) {
  const unavailable = gpu.toUpperCase().includes("CPU");
  if (unavailable) {
    return (
      <Badge variant="danger" className="uppercase tracking-wider">
        GPU Unavailable
      </Badge>
    );
  }
  return <Badge variant="success">{gpu}</Badge>;
}

function ThemeToggle() {
  const theme = useThemeStore((s) => s.theme);
  const toggleTheme = useThemeStore((s) => s.toggleTheme);
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggleTheme}
      title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
    >
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  );
}
