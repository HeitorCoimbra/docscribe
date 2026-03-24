export function RecordingIndicator() {
  return (
    <div className="flex items-center gap-1.5 text-xs text-destructive">
      <span className="h-2 w-2 rounded-full bg-destructive animate-pulse" />
      Gravando...
    </div>
  );
}
