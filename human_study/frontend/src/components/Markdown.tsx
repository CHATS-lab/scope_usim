"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { CodeBlock } from "./CodeBlock";
import { cn } from "@/lib/cn";

export function Markdown({ children, className }: { children: string; className?: string }) {
  return (
    <ReactMarkdown
      className={cn(
        "prose-usim prose-invert max-w-none text-sm leading-relaxed",
        className
      )}
      remarkPlugins={[remarkGfm, remarkBreaks]}
      components={{
        code({ inline, className, children, ...props }: any) {
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
        a({ children, href }) {
          return (
            <a
              href={href}
              target="_blank"
              rel="noreferrer"
              className="text-accent underline underline-offset-2 hover:opacity-90"
            >
              {children}
            </a>
          );
        },
        ul({ children }) {
          return <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>;
        },
        ol({ children }) {
          return <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>;
        },
        p({ children }) {
          return <p className="my-2 first:mt-0 last:mb-0">{children}</p>;
        },
        blockquote({ children }) {
          return (
            <blockquote className="my-2 border-l-2 border-border pl-3 text-muted">
              {children}
            </blockquote>
          );
        },
        table({ children }) {
          return (
            <div className="my-3 overflow-x-auto">
              <table className="w-full border-collapse text-xs">{children}</table>
            </div>
          );
        },
        th({ children }) {
          return (
            <th className="border-b border-border px-2 py-1 text-left font-medium">
              {children}
            </th>
          );
        },
        td({ children }) {
          return <td className="border-b border-border/40 px-2 py-1">{children}</td>;
        },
      }}
    >
      {children}
    </ReactMarkdown>
  );
}
