'use client';
import { Mic, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { RecordingIndicator } from "./RecordingIndicator";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";

interface Props {
  onAudioReady: (blob: Blob) => Promise<void>;
  disabled?: boolean;
}

export function AudioRecorder({ onAudioReady, disabled }: Props) {
  const { state, startRecording, stopRecording } = useAudioRecorder();

  const handleClick = async () => {
    if (state === 'recording') {
      const blob = await stopRecording();
      onAudioReady(blob);
    } else if (state === 'idle') {
      await startRecording();
    }
  };

  return (
    <div className="flex items-center gap-2">
      {state === 'recording' && <RecordingIndicator />}
      <Button
        type="button"
        variant={state === 'recording' ? 'destructive' : 'outline'}
        size="icon"
        onClick={handleClick}
        disabled={disabled || state === 'processing'}
        title={state === 'recording' ? 'Parar gravação' : 'Gravar áudio'}
      >
        {state === 'recording' ? (
          <Square className="h-4 w-4" />
        ) : (
          <Mic className="h-4 w-4" />
        )}
      </Button>
    </div>
  );
}
