'use client';
import { useState } from 'react';
import { ChevronDown, ChevronRight, Copy } from 'lucide-react';
import { toast } from 'sonner';
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

function formatSummary(leito: string, summary: LeitoSummary): string {
  const lines: string[] = [];
  const header = summary.nome_paciente
    ? `Leito ${leito} — ${summary.nome_paciente}`
    : `Leito ${leito}`;
  lines.push(header, '');

  if (summary.quadro_clinico.length > 0) {
    lines.push('Quadro Clínico:');
    summary.quadro_clinico.forEach((item, i) => lines.push(`${i + 1}. ${item}`));
    lines.push('');
  }

  if (summary.pendencias.length > 0) {
    lines.push('Pendências:');
    summary.pendencias.forEach((item, i) => lines.push(`${i + 1}. ${item}`));
    lines.push('');
  }

  if (summary.condutas.length > 0) {
    lines.push('Condutas:');
    summary.condutas.forEach((item) => lines.push(`- ${item}`));
  }

  return lines.join('\n').trim();
}

export function LeitoCard({ leito, summary }: Props) {
  const [isOpen, setIsOpen] = useState(false);

  const hasAlert =
    !summary.nome_paciente ||
    [...summary.quadro_clinico, ...summary.pendencias, ...summary.condutas].some(
      (s) => s.includes('PENDENTE') || s.includes('🔴')
    );

  return (
    <div className="rounded-md border border-border bg-card text-sm overflow-hidden">
      {/* Header / toggle */}
      <div className="group relative flex items-center">
        <button
          onClick={() => setIsOpen((o) => !o)}
          className="flex-1 flex items-center gap-2 px-3 py-2.5 text-left hover:bg-muted/50 transition-colors"
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
        <button
          onClick={(e) => {
            e.stopPropagation();
            navigator.clipboard.writeText(formatSummary(leito, summary)).then(() => toast.success('Copiado!'));
          }}
          className="opacity-100 md:opacity-0 md:group-hover:opacity-100 px-2 py-2.5 text-muted-foreground hover:text-foreground transition-opacity shrink-0"
          title="Copiar resumo"
        >
          <Copy className="h-3.5 w-3.5" />
        </button>
      </div>

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
