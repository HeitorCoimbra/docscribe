import { MessageBubble } from "./MessageBubble";
import { TranscriptionAccordion } from "./TranscriptionAccordion";
import type { Message } from "@/types/session";

interface Props {
  messages: Message[];
  streamingContent: string;
  transcription: string | null;
  isStreaming: boolean;
}

export function MessageList({ messages, streamingContent, transcription, isStreaming }: Props) {
  return (
    <div className="flex flex-col gap-4">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      {isStreaming && (
        <div className="flex flex-col gap-2">
          {transcription && <TranscriptionAccordion transcription={transcription} />}
          <MessageBubble
            message={{
              id: 'streaming',
              thread_id: '',
              role: 'assistant',
              content: streamingContent,
              has_audio: false,
              transcription: null,
              created_at: new Date().toISOString(),
            }}
            isStreaming
          />
        </div>
      )}
    </div>
  );
}
