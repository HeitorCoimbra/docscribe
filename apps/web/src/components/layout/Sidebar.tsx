'use client';
import { usePathname, useRouter } from "next/navigation";
import { PlusIcon, Moon, Sun, Stethoscope } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { useThreads } from "@/hooks/useThreads";
import { SidebarThreadItem } from "./SidebarThreadItem";

export function Sidebar() {
  const { data: groups, isLoading } = useThreads();
  const pathname = usePathname();
  const router = useRouter();
  const { theme, setTheme } = useTheme();

  return (
    <aside className="flex h-screen w-64 flex-col border-r border-border bg-background">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border">
        <Stethoscope className="h-5 w-5 text-primary" />
        <span className="font-semibold text-sm">DocScribe</span>
      </div>

      {/* New session button */}
      <div className="px-3 py-2">
        <Button
          variant="outline"
          size="sm"
          className="w-full justify-start gap-2"
          onClick={() => router.push("/app/sessao/nova")}
        >
          <PlusIcon className="h-4 w-4" />
          Nova Sessão
        </Button>
      </div>

      {/* Thread list */}
      <div className="flex-1 overflow-y-auto px-2">
        {isLoading && (
          <div className="px-2 py-4 text-xs text-muted-foreground">Carregando...</div>
        )}
        {groups?.map((group) => {
          const visibleThreads = group.threads.filter((thread) => {
            const isEmpty =
              (!thread.title || thread.title === 'Nova Sessão') &&
              Object.keys(thread.confirmed_leitos).length === 0;
            return !isEmpty || pathname === `/app/sessao/${thread.id}`;
          });
          if (visibleThreads.length === 0) return null;
          return (
            <div key={group.date} className="mb-3">
              <div className="px-2 py-1 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                {group.label}
              </div>
              {visibleThreads.map((thread) => (
                <SidebarThreadItem
                  key={thread.id}
                  thread={thread}
                  isActive={pathname === `/app/sessao/${thread.id}`}
                  onClick={() => router.push(`/app/sessao/${thread.id}`)}
                />
              ))}
            </div>
          );
        })}
      </div>

      {/* Footer: theme toggle */}
      <div className="border-t border-border px-3 py-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className="w-full justify-start gap-2 text-xs"
        >
          {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          {theme === 'dark' ? 'Modo claro' : 'Modo escuro'}
        </Button>
      </div>
    </aside>
  );
}
