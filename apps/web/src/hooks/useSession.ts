'use client';
import { useState, useCallback } from 'react';
import type { ThreadDetail, Message, ConfirmedLeitos } from '@/types/session';

export interface SessionState {
  thread: ThreadDetail | null;
  messages: Message[];
  streamingContent: string;
  transcription: string | null;
  confirmedLeitos: ConfirmedLeitos;
  isStreaming: boolean;
}

export function useSessionState(initial: ThreadDetail | null) {
  const [state, setState] = useState<SessionState>({
    thread: initial,
    messages: initial?.messages ?? [],
    streamingContent: '',
    transcription: null,
    confirmedLeitos: initial?.confirmed_leitos ?? {},
    isStreaming: false,
  });

  const startStreaming = useCallback(() => {
    setState((s) => ({ ...s, streamingContent: '', transcription: null, isStreaming: true }));
  }, []);

  const appendDelta = useCallback((text: string) => {
    setState((s) => ({ ...s, streamingContent: s.streamingContent + text }));
  }, []);

  const setTranscription = useCallback((text: string) => {
    setState((s) => ({ ...s, transcription: text }));
  }, []);

  const finishStreaming = useCallback(
    (
      assistantContent: string,
      data: { thread_title: string; confirmed_leitos: ConfirmedLeitos }
    ) => {
      setState((s) => {
        const assistantMsg: Message = {
          id: crypto.randomUUID(),
          thread_id: s.thread?.id ?? '',
          role: 'assistant',
          content: assistantContent,
          has_audio: false,
          transcription: null,
          created_at: new Date().toISOString(),
        };
        return {
          ...s,
          messages: [...s.messages, assistantMsg],
          streamingContent: '',
          isStreaming: false,
          confirmedLeitos: data.confirmed_leitos,
          thread: s.thread
            ? { ...s.thread, title: data.thread_title, confirmed_leitos: data.confirmed_leitos }
            : s.thread,
        };
      });
    },
    []
  );

  const addUserMessage = useCallback((content: string, hasAudio = false) => {
    setState((s) => {
      const msg: Message = {
        id: crypto.randomUUID(),
        thread_id: s.thread?.id ?? '',
        role: 'user',
        content,
        has_audio: hasAudio,
        transcription: null,
        created_at: new Date().toISOString(),
      };
      return { ...s, messages: [...s.messages, msg] };
    });
  }, []);

  return { state, startStreaming, appendDelta, setTranscription, finishStreaming, addUserMessage };
}
