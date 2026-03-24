import type { ConfirmedLeitos } from './session';

export interface SSETranscriptionEvent {
  type: 'transcription';
  text: string;
}

export interface SSEDeltaEvent {
  type: 'delta';
  text: string;
}

export interface SSEDoneEvent {
  type: 'done';
  thread_title: string;
  confirmed_leitos: ConfirmedLeitos;
}

export interface SSEErrorEvent {
  type: 'error';
  message: string;
}

export type SSEEvent = SSETranscriptionEvent | SSEDeltaEvent | SSEDoneEvent | SSEErrorEvent;
