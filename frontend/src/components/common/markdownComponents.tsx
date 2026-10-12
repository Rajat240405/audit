import type { Components } from "react-markdown";
import { classifyLink } from "@/lib/markdownLinks";

/**
 * Shared react-markdown component overrides for every rendered answer (chat
 * transcript and canvas), so link handling is identical everywhere.
 *
 * External http(s) links open in a new tab with noopener/noreferrer. Anything
 * unsafe is rendered as plain text, never as a navigable link.
 */
export const markdownComponents: Components = {
  // Keep wide tables from overflowing the card: each table scrolls on its own.
  table: ({ children }) => (
    <div className="overflow-x-auto">
      <table>{children}</table>
    </div>
  ),
  a: ({ href, children }) => {
    const kind = classifyLink(href, window.location.href);
    switch (kind) {
      case "external":
        return (
          <a href={href} target="_blank" rel="noopener noreferrer">
            {children}
          </a>
        );
      case "internal":
      case "fragment":
      case "contact":
        return <a href={href}>{children}</a>;
      default:
        return (
          <span className="underline decoration-dotted" title="This link cannot be opened">
            {children}
          </span>
        );
    }
  },
};
