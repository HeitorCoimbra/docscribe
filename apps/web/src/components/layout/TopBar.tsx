'use client';
import { useSession, signOut } from "next-auth/react";
import { LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  title?: string;
}

export function TopBar({ title }: Props) {
  const { data: session } = useSession();

  return (
    <header className="flex h-12 items-center justify-between border-b border-border px-6">
      <span className="text-sm font-medium">{title ?? "DocScribe"}</span>
      <div className="flex items-center gap-3">
        {session?.user?.name && (
          <span className="text-xs text-muted-foreground">{session.user.name}</span>
        )}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => signOut({ callbackUrl: "/login" })}
          className="gap-1.5 text-xs"
        >
          <LogOut className="h-3.5 w-3.5" />
          Sair
        </Button>
      </div>
    </header>
  );
}
