/**
 * External Markdown links open in a new tab with noopener/noreferrer; unsafe
 * schemes are not navigable; internal and in-page links stay in the same tab.
 */
import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import ReactMarkdown from "react-markdown";
import { markdownComponents } from "@/components/common/markdownComponents";
import { classifyLink } from "@/lib/markdownLinks";

const BASE = "http://localhost:3000/workspace";

describe("classifyLink", () => {
  it.each([
    ["https://example.com/report", "external"],
    ["HTTP://Example.com/", "external"],
    ["//cdn.example.org/x.pdf", "external"],
    ["http://localhost:3000/api/docs", "internal"],
    ["/docs/guide", "internal"],
    ["#section-2", "fragment"],
    ["mailto:officer@example.gov", "contact"],
    ["tel:+911234567890", "contact"],
    ["javascript:alert(1)", "unsafe"],
    ["java\tscript:alert(1)", "unsafe"],
    ["data:text/html,<script>alert(1)</script>", "unsafe"],
    ["file:///etc/passwd", "unsafe"],
    ["vbscript:msgbox(1)", "unsafe"],
    ["", "unsafe"],
    [undefined, "unsafe"],
  ] as const)("%s -> %s", (href, kind) => {
    expect(classifyLink(href, BASE)).toBe(kind);
  });
});

function renderMarkdown(md: string) {
  return render(<ReactMarkdown components={markdownComponents}>{md}</ReactMarkdown>);
}

describe("Markdown links in answers", () => {
  it("opens an external link in a new tab without giving the page's window back", () => {
    const { container } = renderMarkdown("[Ministry report](https://example.com/report)");
    const a = container.querySelector("a");
    expect(a).not.toBeNull();
    expect(a!.getAttribute("href")).toBe("https://example.com/report");
    expect(a!.getAttribute("target")).toBe("_blank");
    expect(a!.getAttribute("rel")).toBe("noopener noreferrer");
    expect(a!.textContent).toBe("Ministry report");
  });

  it("does not make unsafe schemes navigable; the text is kept", () => {
    const { container } = renderMarkdown(
      "[click](javascript:alert(1)) and [data](data:text/html,hi)"
    );
    const anchors = Array.from(container.querySelectorAll("a"));
    for (const a of anchors) {
      expect(a.getAttribute("href") ?? "").not.toMatch(/^(javascript|data):/i);
    }
    expect(container.textContent).toContain("click");
    expect(container.textContent).toContain("data");
  });

  it("keeps in-page and same-origin links in the same tab", () => {
    const { container } = renderMarkdown("[top](#top) [guide](/docs/guide)");
    const [top, guide] = Array.from(container.querySelectorAll("a"));
    expect(top.getAttribute("target")).toBeNull();
    expect(guide.getAttribute("target")).toBeNull();
    expect(guide.getAttribute("href")).toBe("/docs/guide");
  });

  it("keeps mailto links as contact links, not new tabs", () => {
    const { container } = renderMarkdown("[mail](mailto:officer@example.gov)");
    const a = container.querySelector("a")!;
    expect(a.getAttribute("href")).toBe("mailto:officer@example.gov");
    expect(a.getAttribute("target")).toBeNull();
  });
});
