'use client';
import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { LeitoSummary } from '@/types/session';
import ReactMarkdown from 'react-markdown';

interface Props {
  leito: string;
  summary: LeitoSummary;
}

function InlineMd({ children }: { children: string }) {
  return (
    <ReactMarkdown components={{ p: ({ children }) => <>{children}</> }}>
      {children}
    </ReactMarkdown>
  );
}

export function LeitoCard({ leito, summary }: Props) {
  const [isOpen, setIsOpen] = useState(false);

  const hasAlert =
    summary.pendencias.length > 0 ||
    !summary.nome_paciente ||
    [...summary.quadro_clinico, ...summary.pendencias, ...summary.condutas].some(
      (s) => s.includes('PENDENTE') || s.includes('🔴')
    );

  return (
    <div className="rounded-md border border-border bg-card text-sm overflow-hidden">
      {/* Header / toggle */}
      <button
        onClick={() => setIsOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-muted/50 transition-colors"
      >
        {isOpen ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        )}
        <span className="font-medium flex-1 truncate">
          Leito {leito}
          {summary.nome_paciente && (
            <span className="ml-1 text-muted-foreground font-normal">— {summary.nome_paciente}</span>
          )}
        </span>
        {hasAlert && (
          <span className="shrink-0 rounded-full bg-red-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-red-600 dark:text-red-400 leading-none">
            Pendência
          </span>
        )}
      </button>

      {/* Collapsible body */}
      {isOpen && (
        <div className="px-3 pb-3 pt-1 border-t border-border space-y-2">
          {summary.quadro_clinico.length > 0 && (
            <div>
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">
                Quadro Clínico
              </div>
              <ol className="list-decimal list-inside space-y-0.5 text-xs leading-relaxed">
                {summary.quadro_clinico.map((item, i) => (
                  <li key={i}><InlineMd>{item}</InlineMd></li>
                ))}
              </ol>
            </div>
          )}

          {summary.pendencias.length > 0 && (
            <div>
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">
                Pendências
              </div>
              <ol className="list-decimal list-inside space-y-0.5 text-xs leading-relaxed">
                {summary.pendencias.map((item, i) => (
                  <li key={i}><InlineMd>{item}</InlineMd></li>
                ))}
              </ol>
            </div>
          )}

          {summary.condutas.length > 0 && (
            <div>
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">
                Condutas
              </div>
              <ul className="space-y-0.5 text-xs leading-relaxed">
                {summary.condutas.map((item, i) => (
                  <li key={i} className="flex gap-1">
                    <span className="text-muted-foreground shrink-0">•</span>
                    <span><InlineMd>{item}</InlineMd></span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
