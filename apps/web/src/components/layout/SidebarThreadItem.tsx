'use client';
import { cn } from "@/lib/utils";
import type { Thread } from "@/types/session";

interface Props {
  thread: Thread;
  isActive: boolean;
  onClick: () => void;
}

export function SidebarThreadItem({ thread, isActive, onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full rounded-md px-2 py-1.5 text-left text-sm transition-colors",
        isActive
          ? "bg-primary/10 text-primary font-medium"
          : "text-foreground hover:bg-muted"
      )}
    >
      <span className="block truncate">{thread.title || "Nova Sessão"}</span>
    </button>
  );
}
