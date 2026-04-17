"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { CodeBlock } from "./CodeBlock";
import { cn } from "@/lib/cn";

export function Markdown({ children, className }: { children: string; className?: string }) {
  return (
    <ReactMarkdown
      className={cn("max-w-none", className)}
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
          return <ul className="my-3 list-disc space-y-1.5 pl-5">{children}</ul>;
        },
        ol({ children }) {
          return <ol className="my-3 list-decimal space-y-1.5 pl-5">{children}</ol>;
        },
        li({ children }) {
          return <li className="[&>p]:my-0">{children}</li>;
        },
        p({ children }) {
          return <p className="my-2 first:mt-0 last:mb-0">{children}</p>;
        },
        strong({ children }) {
          return <strong className="font-semibold text-text">{children}</strong>;
        },
        h1({ children }) {
          return <h1 className="mt-4 text-lg font-semibold text-text">{children}</h1>;
        },
        h2({ children }) {
          return <h2 className="mt-4 text-base font-semibold text-text">{children}</h2>;
        },
        h3({ children }) {
          return <h3 className="mt-3 text-sm font-semibold text-text">{children}</h3>;
        },
        blockquote({ children }) {
          return (
            <blockquote className="my-2 border-l-2 border-border pl-3 text-muted">
              {children}
            </blockquote>
          );
        },
        hr() {
          return <hr className="my-4 border-border" />;
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
