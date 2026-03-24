import type { LeitoSummary } from '@/types/session';
import ReactMarkdown from 'react-markdown';

interface Props {
  leito: string;
  summary: LeitoSummary;
}

export function LeitoCard({ leito, summary }: Props) {
  return (
    <div className="rounded-md border border-border bg-card p-3 text-sm">
      <div className="mb-2 font-medium">
        Leito {leito}
        {summary.nome_paciente && (
          <span className="ml-1 text-muted-foreground font-normal">— {summary.nome_paciente}</span>
        )}
      </div>

      {summary.quadro_clinico.length > 0 && (
        <div className="mb-2">
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">
            Quadro Clínico
          </div>
          <ol className="list-decimal list-inside space-y-0.5 text-xs leading-relaxed">
            {summary.quadro_clinico.map((item, i) => (
              <li key={i}><ReactMarkdown components={{ p: ({ children }) => <>{children}</> }}>{item}</ReactMarkdown></li>
            ))}
          </ol>
        </div>
      )}

      {summary.pendencias.length > 0 && (
        <div className="mb-2">
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">
            Pendências
          </div>
          <ol className="list-decimal list-inside space-y-0.5 text-xs leading-relaxed">
            {summary.pendencias.map((item, i) => (
              <li key={i}><ReactMarkdown components={{ p: ({ children }) => <>{children}</> }}>{item}</ReactMarkdown></li>
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
                <span className="text-muted-foreground">•</span>
                <span><ReactMarkdown components={{ p: ({ children }) => <>{children}</> }}>{item}</ReactMarkdown></span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
