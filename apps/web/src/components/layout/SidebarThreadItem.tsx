'use client';
import { Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Thread } from "@/types/session";

interface Props {
  thread: Thread;
  isActive: boolean;
  onClick: () => void;
  onDelete: () => void;
}

export function SidebarThreadItem({ thread, isActive, onClick, onDelete }: Props) {
  return (
    <div className="relative group">
      <button
        onClick={onClick}
        className={cn(
          "w-full rounded-md px-2 py-1.5 text-left text-sm transition-colors pr-8",
          isActive
            ? "bg-primary/10 text-primary font-medium"
            : "text-foreground hover:bg-muted"
        )}
      >
        <span className="block truncate">{thread.title || "Nova Sessão"}</span>
      </button>
      <button
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        className="absolute right-1 top-1/2 -translate-y-1/2 opacity-100 md:opacity-0 md:group-hover:opacity-100 p-1 rounded text-muted-foreground hover:text-destructive transition-opacity"
        title="Excluir sessão"
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
