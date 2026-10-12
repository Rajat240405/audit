/**
 * Classify a Markdown link target for rendering.
 *
 *  - external  http(s) to another origin: opens in a NEW tab, with
 *              rel="noopener noreferrer". The app session is not touched.
 *  - internal  http(s) to this origin, or a relative path: same-tab link.
 *  - fragment  "#section": in-page anchor, same tab.
 *  - contact   mailto: / tel: — opens the user's mail or phone app.
 *  - unsafe    any other scheme (javascript:, data:, file:, vbscript:, …) and
 *              anything that does not parse. Not navigable: rendered as text.
 */
export type LinkKind = "external" | "internal" | "fragment" | "contact" | "unsafe";

export function classifyLink(href: string | undefined | null, base: string): LinkKind {
  const raw = (href ?? "").trim();
  if (!raw) return "unsafe";
  if (raw.startsWith("#")) return "fragment";
  let url: URL;
  let baseUrl: URL;
  try {
    baseUrl = new URL(base);
    url = new URL(raw, baseUrl);
  } catch {
    return "unsafe";
  }
  if (url.protocol === "http:" || url.protocol === "https:") {
    return url.origin === baseUrl.origin ? "internal" : "external";
  }
  if (url.protocol === "mailto:" || url.protocol === "tel:") return "contact";
  return "unsafe";
}
