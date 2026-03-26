'use client';
import { usePathname, useRouter } from "next/navigation";
import { PlusIcon, Moon, Sun, Stethoscope, ChevronLeft, ChevronRight, X } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { useThreads, useDeleteThread } from "@/hooks/useThreads";
import { toast } from "sonner";
import { SidebarThreadItem } from "./SidebarThreadItem";
import { useSidebar } from "./SidebarContext";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";

export function Sidebar() {
  const { data: groups, isLoading } = useThreads();
  const deleteThread = useDeleteThread();
  const pathname = usePathname();
  const router = useRouter();
  const { theme, setTheme } = useTheme();
  const { collapsed, mobileOpen, toggleCollapse, closeMobile } = useSidebar();

  const navigate = (path: string) => {
    router.push(path);
    closeMobile();
  };

  return (
    <>
      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/50 md:hidden"
          onClick={closeMobile}
        />
      )}

      <aside
        className={cn(
          "flex h-screen flex-col border-r border-border bg-background transition-all duration-200",
          collapsed ? "w-14" : "w-64",
          // Mobile: fixed overlay; desktop: in-flow
          "fixed inset-y-0 left-0 z-30 md:relative",
          mobileOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
        )}
      >
        {/* Header */}
        <div className={cn(
          "flex items-center border-b border-border",
          collapsed ? "justify-center px-2 py-3" : "gap-2 px-4 py-3"
        )}>
          <Stethoscope className="h-5 w-5 text-primary shrink-0" />
          {!collapsed && <span className="font-semibold text-sm flex-1">DocScribe</span>}
          {/* Desktop collapse toggle */}
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 hidden md:flex"
            onClick={toggleCollapse}
            title={collapsed ? "Expandir menu" : "Recolher menu"}
          >
            {collapsed
              ? <ChevronRight className="h-4 w-4" />
              : <ChevronLeft className="h-4 w-4" />
            }
          </Button>
          {/* Mobile close button */}
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 md:hidden"
            onClick={closeMobile}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* New session button */}
        <div className={cn("py-2", collapsed ? "px-2" : "px-3")}>
          <Button
            variant="outline"
            size="sm"
            className={cn("w-full", collapsed ? "justify-center px-0" : "justify-start gap-2")}
            onClick={() => navigate("/app/sessao/nova")}
            title={collapsed ? "Nova Sessão" : undefined}
          >
            <PlusIcon className="h-4 w-4 shrink-0" />
            {!collapsed && "Nova Sessão"}
          </Button>
        </div>

        {/* Thread list — hidden when collapsed */}
        {!collapsed && (
          <div className="flex-1 overflow-y-auto px-2">
            {isLoading && (
              <div className="px-2 py-3 space-y-1">
                {[80, 64, 72, 56].map((w) => (
                  <Skeleton key={w} className="h-7 rounded-md" style={{ width: `${w}%` }} />
                ))}
              </div>
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
                      onClick={() => navigate(`/app/sessao/${thread.id}`)}
                      onDelete={() => {
                        deleteThread.mutate(thread.id, {
                          onSuccess: () => {
                            toast.success('Sessão excluída');
                            if (pathname === `/app/sessao/${thread.id}`) {
                              router.push('/app/sessao/nova');
                            }
                          },
                        });
                      }}
                    />
                  ))}
                </div>
              );
            })}
          </div>
        )}
        {collapsed && <div className="flex-1" />}

        {/* Footer: theme toggle */}
        <div className={cn("border-t border-border py-2", collapsed ? "px-2" : "px-3")}>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className={cn("w-full text-xs", collapsed ? "justify-center px-0" : "justify-start gap-2")}
            title={collapsed ? (theme === 'dark' ? 'Modo claro' : 'Modo escuro') : undefined}
          >
            {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            {!collapsed && (theme === 'dark' ? 'Modo claro' : 'Modo escuro')}
          </Button>
        </div>
      </aside>
    </>
  );
}
