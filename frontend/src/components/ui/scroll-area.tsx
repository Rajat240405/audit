import * as React from "react";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area";
import { cn } from "@/utils/cn";

export function ScrollArea({
  className,
  children,
  ...props
}: React.ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.Root>) {
  return (
    <ScrollAreaPrimitive.Root className={cn("relative overflow-hidden", className)} {...props}>
      <ScrollAreaPrimitive.Viewport className="h-full w-full rounded-[inherit]">
        {children}
      </ScrollAreaPrimitive.Viewport>
      <ScrollAreaPrimitive.Scrollbar
        orientation="vertical"
        // w-2 is required: without an explicit width the scrollbar renders
        // zero-width and the user gets no visible affordance.
        className="flex w-2 touch-none select-none p-0.5 transition-colors hover:bg-surface-2"
      >
        <ScrollAreaPrimitive.Thumb className="relative flex-1 rounded-full bg-border" />
      </ScrollAreaPrimitive.Scrollbar>
    </ScrollAreaPrimitive.Root>
  );
}

export function Separator({ className }: { className?: string }) {
  return <div className={cn("h-px w-full bg-border", className)} />;
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-md bg-surface-2", className)} />;
}
