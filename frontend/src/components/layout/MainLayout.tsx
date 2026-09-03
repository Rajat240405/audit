import { Header } from "./Header";
import { Sidebar } from "./Sidebar";
import { Toast } from "@/components/common/Toast";
import { useToastStore } from "@/store/useToastStore";

/** 2-column workstation layout (matches the Stitch design — no right rail). */
export function MainLayout({ children }: { children: React.ReactNode }) {
  const toasts = useToastStore((s) => s.toasts);
  const dismiss = useToastStore((s) => s.dismiss);
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background text-foreground">
      <Header />
      <div className="flex min-h-0 flex-1">
        <Sidebar />
        <main className="flex min-w-0 flex-1 flex-col">{children}</main>
      </div>
      <Toast toasts={toasts} onDismiss={dismiss} />
    </div>
  );
}
