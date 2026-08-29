"use client";

import { Bot, User, Wrench } from "lucide-react";
import { cn } from "@/lib/cn";

export type Speaker = "user" | "assistant" | "tool";

interface Props {
  role: Speaker;
  name?: string;
  /** "participant" = the viewer is talking to the agent (user is "You");
   *  "annotator"   = a third party reviewing the transcript (user is "User"). */
  perspective?: "participant" | "annotator";
  align?: "left" | "right";
  size?: "sm" | "md";
}

export function SpeakerBadge({
  role,
  name,
  perspective = "participant",
  align = "left",
  size = "md",
}: Props) {
  const config = STYLES[role];
  const label =
    name ??
    (role === "user"
      ? perspective === "participant"
        ? "You"
        : "User"
      : role === "assistant"
      ? "Assistant"
      : "Tool");
  const Icon = config.icon;
  const avatarSize = size === "sm" ? "h-5 w-5" : "h-6 w-6";
  const iconSize = size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5";

  return (
    <div
      className={cn(
        "flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted",
        align === "right" ? "justify-end" : "justify-start"
      )}
    >
      {align === "right" && <span>{label}</span>}
      <span
        className={cn(
          "flex shrink-0 items-center justify-center rounded-full",
          avatarSize,
          config.wrap
        )}
        aria-hidden="true"
      >
        <Icon className={cn(iconSize, config.iconColor)} strokeWidth={2.25} />
      </span>
      {align === "left" && <span>{label}</span>}
    </div>
  );
}

const STYLES = {
  user: {
    icon: User,
    wrap: "bg-accent/20 ring-1 ring-accent/40",
    iconColor: "text-accent",
  },
  assistant: {
    icon: Bot,
    wrap: "bg-emerald-400/20 ring-1 ring-emerald-400/40",
    iconColor: "text-emerald-300",
  },
  tool: {
    icon: Wrench,
    wrap: "bg-amber-400/20 ring-1 ring-amber-400/40",
    iconColor: "text-amber-300",
  },
} as const;
