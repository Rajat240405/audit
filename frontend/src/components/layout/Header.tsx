import { useQuery } from "@tanstack/react-query";
import { Moon, Sun } from "lucide-react";
import { fetchModels, setProvider } from "@/api/model";
import { useAppStore } from "@/store/useAppStore";
import { useThemeStore } from "@/store/useThemeStore";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
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
  { value: "default", label: "Smart Auto" },
  { value: "formal", label: "Formal" },
  { value: "concise", label: "Concise" },
  { value: "executive", label: "Executive" },
  { value: "scientific", label: "Scientific" },
  { value: "government", label: "Government" },
];

export function Header() {
  const app = useAppStore();

  const { data: models } = useQuery({
    queryKey: ["models", app.provider],
    queryFn: () => fetchModels(app.provider),
    enabled: Boolean(app.provider),
    staleTime: 60_000,
  });

  const changeProvider = async (p: string) => {
    app.setProvider(p);
    try {
      await setProvider(p, app.modelFamily);
    } catch {
      /* backend may be offline */
    }
  };

  return (
    <header className="flex shrink-0 items-center gap-3 border-b border-border bg-surface px-4 py-2">
      <div className="flex shrink-0 items-center gap-2 text-lg font-bold tracking-wide">
        <span className="text-accent">◆</span>
        <span className="text-sm">INCOIS AUDIT PRO</span>
      </div>

      <div className="flex flex-1 items-center justify-center gap-3 text-sm">
        <Select value={app.provider} onValueChange={changeProvider}>
          <SelectTrigger className="w-24">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="ollama">Ollama</SelectItem>
            <SelectItem value="groq">Groq</SelectItem>
            <SelectItem value="openai">OpenAI</SelectItem>
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

        <Button variant="secondary" size="sm">
          Load Model
        </Button>

        <Button size="sm" onClick={() => app.setBuildModalOpen(true)}>
          Build Graph
        </Button>

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

        {/* RAG toggle — Hybrid on, Graph off */}
        <label className="flex cursor-pointer select-none items-center gap-1.5">
          <input
            type="checkbox"
            checked={app.retrievalMode === "hybrid"}
            onChange={(e) => app.setRetrievalMode(e.target.checked ? "hybrid" : "graph")}
            className="h-3.5 w-3.5 rounded border-border accent-[#10b981]"
          />
          <span className="text-xs font-medium">RAG</span>
        </label>
      </div>

      <div className="flex shrink-0 items-center gap-1">
        <ThemeToggle />
      </div>
    </header>
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
