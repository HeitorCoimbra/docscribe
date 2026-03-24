import { cn } from "@/lib/utils";
import type { Message } from "@/types/session";

interface Props {
  message: Message;
  isStreaming?: boolean;
}

export function MessageBubble({ message, isStreaming }: Props) {
  const isUser = message.role === 'user';

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
        <pre className="whitespace-pre-wrap font-sans">{message.content}</pre>
        {isStreaming && !message.content && (
          <span className="inline-block h-4 w-1 bg-current animate-pulse" />
        )}
      </div>
    </div>
  );
}
