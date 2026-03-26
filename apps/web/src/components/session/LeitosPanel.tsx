'use client';
import { useState } from "react";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSession } from "next-auth/react";
import { getApiUrl } from "@/lib/api";
import { LeitoCard } from "./LeitoCard";
import type { ConfirmedLeitos } from "@/types/session";

interface Props {
  confirmedLeitos: ConfirmedLeitos;
  threadId: string;
}

export function LeitosPanel({ confirmedLeitos, threadId }: Props) {
  const { data: session } = useSession();
  const token = (session as { accessToken?: string })?.accessToken;
  const [downloading, setDownloading] = useState(false);

  const downloadPdf = async () => {
    setDownloading(true);
    try {
      const res = await fetch(`${getApiUrl()}/api/v1/threads/${threadId}/pdf`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `sumarios_plantao_${new Date().toISOString().slice(0, 10)}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setDownloading(false);
    }
  };

  const sortedLeitos = Object.entries(confirmedLeitos).sort(([a], [b]) => {
    const na = parseInt(a) || 0;
    const nb = parseInt(b) || 0;
    return na - nb;
  });

  return (
    <aside className="w-full md:w-80 border-l border-border flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <span className="text-sm font-medium">Leitos confirmados</span>
        <Button
          size="sm"
          variant="outline"
          onClick={downloadPdf}
          disabled={downloading || sortedLeitos.length === 0}
          className="gap-1.5 text-xs"
        >
          <Download className="h-3.5 w-3.5" />
          PDF
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-3">
        {sortedLeitos.map(([num, summary]) => (
          <LeitoCard key={num} leito={num} summary={summary} />
        ))}
      </div>
    </aside>
  );
}
