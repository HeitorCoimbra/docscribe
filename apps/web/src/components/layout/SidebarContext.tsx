'use client';
import { createContext, useContext, useState, ReactNode } from 'react';

interface SidebarCtx {
  collapsed: boolean;
  mobileOpen: boolean;
  toggleCollapse: () => void;
  openMobile: () => void;
  closeMobile: () => void;
}

const SidebarContext = createContext<SidebarCtx>({
  collapsed: false,
  mobileOpen: false,
  toggleCollapse: () => {},
  openMobile: () => {},
  closeMobile: () => {},
});

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  return (
    <SidebarContext.Provider value={{
      collapsed,
      mobileOpen,
      toggleCollapse: () => setCollapsed(c => !c),
      openMobile: () => setMobileOpen(true),
      closeMobile: () => setMobileOpen(false),
    }}>
      {children}
    </SidebarContext.Provider>
  );
}

export const useSidebar = () => useContext(SidebarContext);
