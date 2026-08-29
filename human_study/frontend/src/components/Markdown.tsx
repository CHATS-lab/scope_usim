"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { CodeBlock } from "./CodeBlock";
import { cn } from "@/lib/cn";

/**
 * Markdown renderer. Heading + list + paragraph styles are ported from
 * OpenHands (MIT) — see frontend/src/components/features/markdown/headings.tsx
 * for the upstream reference — so line-heights, margins and first:mt-0
 * behave consistently regardless of surrounding wrappers.
 */
export function Markdown({ children, className }: { children: string; className?: string }) {
  return (
    <ReactMarkdown
      className={cn("max-w-none text-[14px] leading-7", className)}
      remarkPlugins={[remarkGfm, remarkBreaks]}
      components={{
        h1: ({ children }) => (
          <h1 className="mb-4 mt-6 text-2xl font-bold leading-8 text-text first:mt-0">
            {children}
          </h1>
        ),
        h2: ({ children }) => (
          <h2 className="mb-3 mt-5 text-xl font-semibold leading-6 tracking-tight text-text first:mt-0">
            {children}
          </h2>
        ),
        h3: ({ children }) => (
          <h3 className="mb-2 mt-4 text-lg font-semibold text-text first:mt-0">
            {children}
          </h3>
        ),
        h4: ({ children }) => (
          <h4 className="mb-2 mt-4 text-base font-semibold text-text first:mt-0">
            {children}
          </h4>
        ),
        h5: ({ children }) => (
          <h5 className="mb-2 mt-3 text-sm font-semibold text-text first:mt-0">
            {children}
          </h5>
        ),
        h6: ({ children }) => (
          <h6 className="mb-2 mt-3 text-sm font-medium text-muted first:mt-0">
            {children}
          </h6>
        ),
        p: ({ children }) => <p className="my-3 leading-7 first:mt-0 last:mb-0">{children}</p>,
        strong: ({ children }) => (
          <strong className="font-semibold text-text">{children}</strong>
        ),
        ul: ({ children }) => (
          <ul className="my-3 list-disc space-y-1.5 pl-5 first:mt-0 last:mb-0">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="my-3 list-decimal space-y-1.5 pl-5 first:mt-0 last:mb-0">{children}</ol>
        ),
        li: ({ children }) => <li className="leading-7 [&>p]:my-0">{children}</li>,
        a: ({ children, href }) => (
          <a
            href={href}
            target="_blank"
            rel="noreferrer"
            className="text-accent underline underline-offset-2 hover:opacity-90"
          >
            {children}
          </a>
        ),
        blockquote: ({ children }) => (
          <blockquote className="my-3 border-l-2 border-border pl-3 text-muted">
            {children}
          </blockquote>
        ),
        hr: () => <hr className="my-4 border-border" />,
        code: ({ inline, className, children, ...props }: any) => {
          const match = /language-(\w+)/.exec(className || "");
          const text = String(children ?? "").replace(/\n$/, "");
          if (inline) {
            return (
              <code
                className="rounded bg-panelAlt px-1.5 py-0.5 font-mono text-[0.85em] text-text"
                {...props}
              >
                {children}
              </code>
            );
          }
          return <CodeBlock language={match?.[1]} value={text} />;
        },
        table: ({ children }) => (
          <div className="my-3 overflow-x-auto">
            <table className="w-full border-collapse text-xs">{children}</table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border-b border-border px-2 py-1 text-left font-medium">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border-b border-border/40 px-2 py-1">{children}</td>
        ),
      }}
    >
      {children}
    </ReactMarkdown>
  );
}
