'use client';
import { useSession, signOut } from "next-auth/react";
import { LogOut, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSidebar } from "./SidebarContext";

interface Props {
  title?: string;
}

export function TopBar({ title }: Props) {
  const { data: session } = useSession();
  const { openMobile } = useSidebar();

  return (
    <header className="flex h-12 items-center justify-between border-b border-border px-4">
      <div className="flex items-center gap-2">
        {/* Hamburger — mobile only */}
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden h-8 w-8"
          onClick={openMobile}
        >
          <Menu className="h-5 w-5" />
        </Button>
        <span className="text-sm font-medium">{title ?? "DocScribe"}</span>
      </div>
      <div className="flex items-center gap-3">
        {session?.user?.name && (
          <span className="hidden sm:block text-xs text-muted-foreground">{session.user.name}</span>
        )}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => signOut({ callbackUrl: "/login" })}
          className="gap-1.5 text-xs"
        >
          <LogOut className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Sair</span>
        </Button>
      </div>
    </header>
  );
}
