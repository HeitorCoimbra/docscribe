import { cn } from "@/lib/utils";
import type { Message } from "@/types/session";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Mic } from 'lucide-react';

interface Props {
  message: Message;
  isStreaming?: boolean;
}

export function MessageBubble({ message, isStreaming }: Props) {
  const isUser = message.role === 'user';

  if (isUser && message.has_audio) {
    return (
      <div className="flex justify-end">
        {message.audioUrl ? (
          <div className="inline-flex items-center gap-2 rounded-2xl bg-primary/10 px-3 py-2">
            <Mic className="h-3.5 w-3.5 text-primary shrink-0" />
            <audio controls src={message.audioUrl} className="h-8 max-w-[220px]" />
          </div>
        ) : (
          <div className="inline-flex items-center gap-1.5 rounded-full bg-primary/20 text-primary px-3 py-1.5 text-xs font-medium">
            <Mic className="h-3.5 w-3.5" />
            Áudio gravado
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[80%] rounded-lg px-4 py-2.5 text-sm",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground border border-border",
          isStreaming && "animate-pulse"
        )}
      >
        <div className="prose-sm max-w-none dark:prose-invert">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              p: ({ children }) => <p className="mb-1 last:mb-0">{children}</p>,
              ul: ({ children }) => <ul className="list-disc list-inside space-y-0.5 mb-1">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal list-inside space-y-0.5 mb-1">{children}</ol>,
              li: ({ children }) => <li className="leading-relaxed">{children}</li>,
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>
        {isStreaming && !message.content && (
          <span className="inline-block h-4 w-1 bg-current animate-pulse" />
        )}
      </div>
    </div>
  );
}
