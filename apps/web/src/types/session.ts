export interface LeitoSummary {
  leito: string;
  nome_paciente: string;
  quadro_clinico: string[];
  pendencias: string[];
  condutas: string[];
}

export type ConfirmedLeitos = Record<string, LeitoSummary>;

export interface Thread {
  id: string;
  title: string | null;
  leito: string | null;
  patient_name: string | null;
  created_at: string;
  updated_at: string;
  created_date: string;
  confirmed_leitos: ConfirmedLeitos;
  is_complete: boolean;
}

export interface Message {
  id: string;
  thread_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  has_audio: boolean;
  transcription: string | null;
  created_at: string;
}

export interface ThreadDetail extends Thread {
  messages: Message[];
}

export interface ThreadGroup {
  date: string;
  label: string;
  threads: Thread[];
}
