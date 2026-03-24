'use client';
import { useRef, useEffect } from "react";
import { toast } from "sonner";
import { useQueryClient } from "@tanstack/react-query";
import type { ThreadDetail } from "@/types/session";
import { MessageList } from "./MessageList";
import { LeitosPanel } from "./LeitosPanel";
import { MessageInput } from "@/components/input/MessageInput";
import { useSessionState } from "@/hooks/useSession";
import { useStreamingSummary } from "@/hooks/useStreamingSummary";

interface Props {
  thread: ThreadDetail;
}

export function SessionView({ thread }: Props) {
  const { state, startStreaming, appendDelta, setTranscription, finishStreaming, addUserMessage } =
    useSessionState(thread);
  const { streamAudio, streamText } = useStreamingSummary();
  const queryClient = useQueryClient();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [state.messages, state.streamingContent]);

  const handleSendText = async (content: string) => {
    addUserMessage(content, false);
    startStreaming();
    let fullContent = '';

    try {
      await streamText(thread.id, content, {
        onDelta: (text) => {
          fullContent += text;
          appendDelta(text);
        },
        onDone: (data) => {
          finishStreaming(fullContent, data);
          queryClient.invalidateQueries({ queryKey: ['threads'] });
        },
        onError: (msg) => {
          toast.error(msg);
        },
      });
    } catch {
      toast.error('Erro ao enviar mensagem');
    }
  };

  const handleSendAudio = async (blob: Blob): Promise<void> => {
    const audioUrl = URL.createObjectURL(blob);
    addUserMessage('', true, audioUrl);
    startStreaming();
    let fullContent = '';

    try {
      await streamAudio(thread.id, blob, {
        onTranscription: setTranscription,
        onDelta: (text) => {
          fullContent += text;
          appendDelta(text);
        },
        onDone: (data) => {
          finishStreaming(fullContent, data);
          queryClient.invalidateQueries({ queryKey: ['threads'] });
        },
        onError: (msg) => {
          toast.error(msg);
        },
      });
    } catch {
      toast.error('Erro ao processar áudio');
    }
  };

  return (
    <div className="flex h-full">
      {/* Main chat column */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Thread title */}
        <div className="border-b border-border px-6 py-3">
          <h2 className="text-sm font-medium">{state.thread?.title || "Nova Sessão"}</h2>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <MessageList
            messages={state.messages}
            streamingContent={state.streamingContent}
            transcription={state.transcription}
            isStreaming={state.isStreaming}
          />
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-border px-6 py-4">
          <MessageInput
            onSendText={handleSendText}
            onSendAudio={handleSendAudio}
            disabled={state.isStreaming}
            threadId={thread.id}
          />
        </div>
      </div>

      {/* Leitos panel */}
      {Object.keys(state.confirmedLeitos).length > 0 && (
        <LeitosPanel confirmedLeitos={state.confirmedLeitos} threadId={thread.id} />
      )}
    </div>
  );
}
