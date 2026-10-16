/**
 * Header account chip: who is signed in, and a one-click way to sign out.
 *
 * Sign-out revokes the session server-side, clears the cache and navigates to
 * /signin. It does not wait for a later request to fail and notice — the
 * redirect is explicit, so a single click is enough.
 */
import { LogIn, LogOut, User as UserIcon } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useSession, useSignOut } from "@/hooks/useAuthSession";

export function SessionMenu() {
  const { user, authenticated } = useSession();
  const { signOut: performSignOut, pending } = useSignOut();

  const signedIn = authenticated;
  const email = user?.email ?? null;

  // One click: revoke, clear, navigate. The hook owns the ordering; on failure
  // it surfaces a toast and leaves the session untouched, so the user is never
  // left half-signed-out without being told.
  const handleSignOut = () => {
    void performSignOut();
  };

  if (!signedIn) {
    return (
      <Link
        to="/signin"
        title="Sign in"
        aria-label="Sign in"
        data-testid="session-signin"
        className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted transition-colors hover:bg-surface-2 hover:text-foreground"
      >
        <LogIn className="h-4 w-4" />
      </Link>
    );
  }

  return (
    <div className="flex items-center gap-1" data-testid="session-menu">
      <span
        className="hidden max-w-[180px] truncate text-xs text-muted lg:inline"
        title={email ?? undefined}
      >
        <UserIcon className="mr-1 inline h-3.5 w-3.5" aria-hidden="true" />
        {email}
      </span>
      <Button
        variant="ghost"
        size="icon"
        onClick={handleSignOut}
        disabled={pending}
        title="Sign out"
        aria-label="Sign out"
        data-testid="session-signout"
      >
        <LogOut className="h-4 w-4" />
      </Button>
    </div>
  );
}

export default SessionMenu;
