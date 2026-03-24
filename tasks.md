# DocScribe — Task Backlog

Reference file for pending work on the Next.js + FastAPI migration.
Tasks are grouped by type: **fixes** (broken or degraded behaviour) and **features** (new capabilities).

---

## Fixes

### F1 — Render markdown in chat bubbles and leito cards (HIGH)
Raw markdown (`**bold**`, `1. item`, `- item`) is displayed as literal text in `MessageBubble` and `LeitoCard`.
- Add `react-markdown` (or similar) to `apps/web/package.json`
- Replace `<pre>` in `MessageBubble` with a markdown renderer
- Replace raw text in `LeitoCard` with the same renderer
- Files: `apps/web/src/components/session/MessageBubble.tsx`, `apps/web/src/components/session/LeitoCard.tsx`

### F2 — Replace audio filename user bubble with a styled chip (MEDIUM)
User messages show raw filenames like `[Arquivo: WhatsApp Audio 2026-03-23 at 07.26.25.opus]`.
Should show a compact audio chip: mic/waveform icon + short label (e.g. "Áudio gravado" or truncated filename).
- File: `apps/web/src/components/session/SessionView.tsx` (`handleSendAudio` label), `MessageBubble.tsx`

### F3 — Add visual section dividers inside LeitoCard (MEDIUM)
Quadro Clínico / Pendências / Condutas sections blend together inside each card.
Even after F1 (markdown rendering), a subtle separator or coloured section label would improve scannability.
- File: `apps/web/src/components/session/LeitoCard.tsx`

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

### FT3 — Mobile sidebar (drawer/overlay)
The sidebar is always visible and takes fixed width. On small screens it collapses the main area. Needs a hamburger toggle + overlay drawer for mobile.
- Files: `apps/web/src/components/layout/Sidebar.tsx`, `apps/web/src/app/app/layout.tsx`

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
