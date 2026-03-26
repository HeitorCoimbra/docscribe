import { Skeleton } from "@/components/ui/skeleton";

export default function Loading() {
  return (
    <div className="flex flex-1 overflow-hidden">
      {/* Chat area */}
      <div className="flex flex-col flex-1 gap-3 p-4 overflow-hidden">
        <Skeleton className="h-14 w-2/3 self-end" />
        <Skeleton className="h-20 w-3/4" />
        <Skeleton className="h-14 w-1/2 self-end" />
        <Skeleton className="h-24 w-3/4" />
      </div>
      {/* Leitos panel */}
      <aside className="w-80 border-l border-border flex flex-col gap-3 p-3">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </aside>
    </div>
  );
}
