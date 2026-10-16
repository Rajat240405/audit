/**
 * Standalone shell for the sign-in / sign-up / forgot-password screens.
 *
 * This is deliberately NOT `MainLayout`. It renders no Header, no Sidebar, no
 * toast region, no Build-Graph modal and no product tour — an authentication
 * screen must not sit inside the dashboard chrome it is gating access to.
 *
 * The dashboard layout is applied separately by `AppShell`, which wraps only
 * the in-app routes.
 */
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { Diamond } from "lucide-react";

interface AuthLayoutProps {
  title: string;
  /** One line of context shown under the title. */
  subtitle?: string;
  children: ReactNode;
  /** Cross-links rendered under the card ("Create account" / "Back to sign in"). */
  footer?: ReactNode;
  /** An error or notice banner rendered above the form. */
  alert?: ReactNode;
}

export function AuthLayout({ title, subtitle, children, footer, alert }: AuthLayoutProps) {
  return (
    <div className="flex min-h-screen w-full flex-col items-center justify-center bg-background px-4 py-10">
      <div className="w-full max-w-md">
        {/* Branding — same marks as the dashboard header, so the product is
            recognisable, but it is a static heading here, not app chrome. */}
        <div className="mb-6 flex flex-col items-center gap-2 text-center">
          <div className="flex items-center gap-2 text-lg font-bold tracking-wide text-foreground">
            <Diamond className="h-4 w-4 fill-current text-accent" aria-hidden="true" />
            <span className="text-sm">INCOIS AUDIT PRO</span>
          </div>
          <h1 className="text-xl font-semibold text-foreground">{title}</h1>
          {subtitle && <p className="text-xs text-muted">{subtitle}</p>}
        </div>

        <div className="rounded-lg border border-border bg-surface p-6 shadow-sm">
          {alert}
          {children}
        </div>

        {footer && <div className="mt-5 text-center text-xs text-muted">{footer}</div>}

        <p className="mt-8 text-center text-[11px] leading-relaxed text-muted">
          Access is limited to <span className="text-foreground">@incois.gov.in</span>{" "}
          addresses and one configured developer address.
        </p>
      </div>
    </div>
  );
}

/** A standard error / notice banner for the auth screens. */
export function AuthAlert({
  kind,
  children,
}: {
  kind: "error" | "success";
  children: ReactNode;
}) {
  if (!children) return null;
  return (
    <div
      role={kind === "error" ? "alert" : "status"}
      className={
        kind === "error"
          ? "mb-4 rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger"
          : "mb-4 rounded-md border border-success/30 bg-success/10 px-3 py-2 text-xs text-success"
      }
    >
      {children}
    </div>
  );
}

/** The muted "or" cross-link line used under each auth card. */
export function AuthLink({ to, children }: { to: string; children: ReactNode }) {
  return (
    <Link to={to} className="font-medium text-accent underline-offset-2 hover:underline">
      {children}
    </Link>
  );
}
