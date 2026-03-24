'use client';
import { useCallback, useRef } from 'react';
import { useSession } from 'next-auth/react';
import { getApiUrl } from '@/lib/api';
import type { SSEEvent } from '@/types/api';
import type { ConfirmedLeitos } from '@/types/session';

export type StreamCallbacks = {
  onTranscription?: (text: string) => void;
  onDelta?: (text: string) => void;
  onDone?: (data: { thread_title: string; confirmed_leitos: ConfirmedLeitos }) => void;
  onError?: (message: string) => void;
};

export function useStreamingSummary() {
  const { data: session } = useSession();
  const token = (session as { accessToken?: string })?.accessToken;
  const abortRef = useRef<AbortController | null>(null);

  const streamAudio = useCallback(
    async (threadId: string, audioBlob: Blob, callbacks: StreamCallbacks) => {
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');

      const res = await fetch(`${getApiUrl()}/api/v1/threads/${threadId}/audio`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
        signal: abortRef.current.signal,
      });

      await consumeSSE(res, callbacks);
    },
    [token]
  );

  const streamText = useCallback(
    async (threadId: string, content: string, callbacks: StreamCallbacks) => {
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      const res = await fetch(`${getApiUrl()}/api/v1/threads/${threadId}/message`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content }),
        signal: abortRef.current.signal,
      });

      await consumeSSE(res, callbacks);
    },
    [token]
  );

  return { streamAudio, streamText };
}

async function consumeSSE(res: Response, callbacks: StreamCallbacks) {
  if (!res.ok) {
    callbacks.onError?.(`HTTP ${res.status}`);
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        const event: SSEEvent = JSON.parse(line.slice(6));
        if (event.type === 'transcription') callbacks.onTranscription?.(event.text);
        else if (event.type === 'delta') callbacks.onDelta?.(event.text);
        else if (event.type === 'done') callbacks.onDone?.(event);
        else if (event.type === 'error') callbacks.onError?.(event.message);
      } catch {
        // ignore parse errors
      }
    }
  }
}
