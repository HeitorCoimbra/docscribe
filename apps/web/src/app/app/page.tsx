import { FileText } from "lucide-react";

export default function AppPage() {
  return (
    <div className="flex h-full items-center justify-center text-muted-foreground">
      <div className="flex flex-col items-center gap-3">
        <FileText className="h-10 w-10 opacity-30" />
        <p className="text-sm">Selecione uma sessão ou crie uma nova</p>
      </div>
    </div>
  );
}
