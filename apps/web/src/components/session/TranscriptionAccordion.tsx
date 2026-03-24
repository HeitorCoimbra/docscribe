'use client';
import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  transcription: string;
}

export function TranscriptionAccordion({ transcription }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-md border border-border bg-muted/50 text-sm">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-3 py-2 text-muted-foreground hover:text-foreground"
      >
        <span>Transcrição do áudio ({transcription.length} chars)</span>
        <ChevronDown className={cn("h-4 w-4 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="border-t border-border px-3 py-2 text-foreground">
          <p className="whitespace-pre-wrap">{transcription}</p>
        </div>
      )}
    </div>
  );
}
