# DocScribe — Task Backlog

Reference file for pending work on the Next.js + FastAPI migration.
Tasks are grouped by type: **fixes** (broken or degraded behaviour) and **features** (new capabilities).

---

## Fixes

### ~~F1 — Render markdown in chat bubbles and leito cards~~ ✅ DONE
`react-markdown` + `remark-gfm` implemented in both `MessageBubble.tsx` and `LeitoCard.tsx`.

### ~~F2 — Replace audio filename user bubble with a styled chip~~ ✅ DONE
`MessageBubble` renders an audio chip (Mic icon + `<audio>` player if URL available, else "Áudio gravado" pill).

### ~~F3 — Add visual section dividers inside LeitoCard~~ ✅ DONE
Section headers (Quadro Clínico / Pendências / Condutas) rendered with uppercase tracking labels in `LeitoCard.tsx`.

### ~~F7 — "Pendência" badge fires for all leitos~~ ✅ DONE
Removed `summary.pendencias.length > 0` from `hasAlert` in `LeitoCard.tsx`. Badge now only triggers on missing patient name or explicit `🔴 PENDENTE` markers.

### ~~F8 — Patient name placeholder not triggering alert badge~~ ✅ DONE
Enforced `""` (empty string) in `LEITO_EXTRACTION_TOOL` schema description and prompt template (`prompts.py`). Frontend check `!summary.nome_paciente` now reliably catches unknown patients.

### F4 — Clean up untitled "Nova Sessão" sidebar entries (LOW)
Sessions that never received audio get persisted with the title "Nova Sessão" and pollute the sidebar.
Options: auto-delete on creation if no message is sent within a short window, or mark them dimmed/italic.
- Files: `apps/web/src/components/layout/Sidebar.tsx`, `apps/web/src/components/layout/SidebarThreadItem.tsx`, possibly `apps/api/app/routers/threads.py`

### F5 — Verify TranscriptionAccordion is rendering (LOW)
The transcription accordion is expected to appear in the chat after each audio submission but was not visible in the screenshot. Confirm whether it is rendering and, if not, fix the SSE `onTranscription` callback path.
- Files: `apps/web/src/hooks/useStreamingSummary.ts`, `apps/web/src/components/session/MessageList.tsx`, `apps/web/src/components/session/TranscriptionAccordion.tsx`

### F6 — Thread title reflects only the first/last extracted leito (LOW)
`update_thread_from_response` picks the first regex match in the response. When multiple leitos are extracted, the title may not reflect the most clinically relevant one.
- File: `apps/api/app/repositories/thread.py` (`update_thread_from_response`)

---

## Features

### FT1 — Leitos panel section header includes patient name
Right-panel section labels currently show only "LEITO 1". Add the patient name: "LEITO 1 — José Rodrigues" to improve navigation when many leitos are present.
- File: `apps/web/src/components/session/LeitosPanel.tsx`

### FT2 — Delete thread from sidebar
There is no UI to delete a thread. The `DELETE /api/v1/threads/{id}` endpoint exists but is not wired up.
- Files: `apps/web/src/components/layout/SidebarThreadItem.tsx`, `apps/web/src/hooks/useThreads.ts` (`useDeleteThread` already exists)

### ~~FT3 — Mobile sidebar (drawer/overlay)~~ ✅ DONE
Sidebar is now a fixed overlay on mobile (z-30, slide-in/out with backdrop). Hamburger button in `TopBar` triggers it via `SidebarContext`. Desktop gets a collapse-to-icon-strip toggle (ChevronLeft/Right). `SidebarProvider` wraps the app layout.
- Files: `SidebarContext.tsx` (new), `Sidebar.tsx`, `TopBar.tsx`, `app/app/layout.tsx`

### FT3b — Right panel (Leitos) mobile accessibility (MEDIUM)
On small screens the two-column layout (chat + leitos panel) is too cramped. The leitos panel should stack below the chat or be accessible via a tab/toggle.
- File: `apps/web/src/app/app/sessao/[id]/page.tsx` or `SessionView.tsx`

### FT3c — Touch-friendly tap targets (LOW)
Several buttons (LeitoCard toggle, SidebarThreadItem, audio chip) are smaller than 44×44px, making them hard to tap on mobile.
- Files: `LeitoCard.tsx`, `SidebarThreadItem.tsx`, `MessageBubble.tsx`

### FT3d — Viewport meta / overflow on mobile (LOW)
Verify `<meta name="viewport">` is set and no horizontal overflow occurs on small screens (e.g. wide code blocks, PDF button row).
- Files: `apps/web/src/app/layout.tsx`

### FT4 — Loading skeletons
No loading state is shown while the thread list or session data is fetching. Add skeleton placeholders.
- Files: `apps/web/src/components/layout/Sidebar.tsx`, `apps/web/src/app/app/sessao/[id]/page.tsx`

### FT5 — Copy leito summary to clipboard
Add a copy button on each `LeitoCard` to copy the formatted summary text for pasting into other systems.
- File: `apps/web/src/components/session/LeitoCard.tsx`

### FT6 — Railway deployment configuration
The `apps/api/railway.json` and web Dockerfile exist but the Railway project (two services + Postgres add-on, env vars, internal networking) has not been set up and smoke-tested in production.
- Files: `apps/api/railway.json`, `apps/web/Dockerfile`, `docker-compose.yml`

---

## Verification checklist (e2e)

From the original migration plan — to be confirmed once deployed:

- [ ] Google OAuth login → user row in `users` table
- [ ] Create session → thread row created, redirect to `/app/sessao/{id}`
- [ ] Record audio → transcription SSE event appears → Claude streams → all leitos shown
- [ ] Upload multiple audio files → processed sequentially, leitos accumulate
- [ ] Send text edit referencing "leito N" → only that leito updated
- [ ] Refresh page → session fully restored from DB
- [ ] Thread list grouped by day in PT-BR labels
- [ ] PDF download → file opens correctly with all leitos
- [ ] Dark/light theme toggle persists across sessions
- [ ] `docker compose up` starts all three services cleanly from scratch
- [ ] Railway deploy: both services healthy, DB migrations run on startup
