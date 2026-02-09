"""
Custom Chainlit data layer for DocScribe.

Uses asyncpg to interact with the PascalCase Chainlit tables
(User, Thread, Step, Element, Feedback) for sidebar history support.
"""

import json
import logging
import traceback
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import asyncpg

from chainlit.data import BaseDataLayer
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)

try:
    from chainlit.user import PersistedUser
except ImportError:
    from chainlit.types import PersistedUser  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


def _to_json(val) -> Optional[str]:
    """Serialize a value to a JSON string for JSONB columns via ::jsonb cast.

    asyncpg does NOT auto-serialize Python dicts for JSONB — when using
    explicit ``::jsonb`` casts we must pass a JSON string.
    """
    if val is None:
        return None
    if isinstance(val, str):
        return val  # already serialized
    return json.dumps(val)


def _from_json(val, default=None):
    """Read a value coming back from a JSONB column.

    asyncpg auto-deserializes JSONB to Python dicts/lists, but some
    values may arrive as strings (depends on asyncpg codec setup).
    """
    if val is None:
        return default
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return default
    return val


class DocScribeDataLayer(BaseDataLayer):
    """PostgreSQL data layer using asyncpg for Chainlit sidebar history."""

    def __init__(self, conninfo: str):
        # Normalize to plain postgresql:// for asyncpg
        ci = conninfo
        for prefix in ("postgresql+psycopg2://", "postgresql+asyncpg://"):
            if ci.startswith(prefix):
                ci = "postgresql://" + ci[len(prefix):]
        if ci.startswith("postgres://"):
            ci = "postgresql://" + ci[len("postgres://"):]
        self.conninfo = ci
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.conninfo, min_size=1, max_size=5
            )
        return self._pool

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_uuid(val: Optional[str]) -> Optional[uuid.UUID]:
        """Convert string to UUID, returning None on failure."""
        if not val:
            return None
        try:
            return uuid.UUID(val)
        except (ValueError, AttributeError):
            return None

    # ── User ─────────────────────────────────────────────────────────

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    'SELECT "id", "identifier", "metadata", "createdAt" '
                    'FROM "User" WHERE "identifier" = $1',
                    identifier,
                )
            if not row:
                return None
            return PersistedUser(
                id=str(row["id"]),
                identifier=row["identifier"],
                metadata=_from_json(row["metadata"], {}),
                createdAt=row["createdAt"] or "",
            )
        except Exception:
            logger.error(f"get_user error:\n{traceback.format_exc()}")
            return None

    async def create_user(self, user) -> Optional[PersistedUser]:
        try:
            pool = await self._get_pool()
            user_id = uuid.uuid4()
            now = self._now_iso()
            meta = json.dumps(user.metadata or {})

            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    '''
                    INSERT INTO "User" ("id", "identifier", "metadata", "createdAt")
                    VALUES ($1, $2, $3::jsonb, $4)
                    ON CONFLICT ("identifier") DO UPDATE
                        SET "metadata" = $3::jsonb
                    RETURNING "id", "identifier", "metadata", "createdAt"
                    ''',
                    user_id, user.identifier, meta, now,
                )
            return PersistedUser(
                id=str(row["id"]),
                identifier=row["identifier"],
                metadata=_from_json(row["metadata"], {}),
                createdAt=row["createdAt"] or now,
            )
        except Exception:
            logger.error(f"create_user error:\n{traceback.format_exc()}")
            return None

    # ── Feedback ─────────────────────────────────────────────────────

    async def upsert_feedback(self, feedback: Feedback) -> str:
        fb_id = uuid.UUID(feedback.id) if feedback.id else uuid.uuid4()
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    '''
                    INSERT INTO "Feedback" ("id", "forId", "threadId", "value", "comment")
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT ("id") DO UPDATE
                        SET "value" = $4, "comment" = $5
                    ''',
                    fb_id,
                    self._safe_uuid(feedback.forId),
                    self._safe_uuid(feedback.threadId),
                    feedback.value,
                    feedback.comment,
                )
        except Exception:
            logger.error(f"upsert_feedback error:\n{traceback.format_exc()}")
        return str(fb_id)

    async def delete_feedback(self, feedback_id: str) -> bool:
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    'DELETE FROM "Feedback" WHERE "id" = $1',
                    self._safe_uuid(feedback_id),
                )
            return "DELETE 1" in result
        except Exception:
            logger.error(f"delete_feedback error:\n{traceback.format_exc()}")
            return False

    # ── Element ──────────────────────────────────────────────────────

    async def create_element(self, element):
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    '''
                    INSERT INTO "Element" (
                        "id", "threadId", "type", "url", "chainlitKey",
                        "name", "display", "objectKey", "size", "page",
                        "language", "forId", "mime", "props"
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14::jsonb)
                    ON CONFLICT ("id") DO NOTHING
                    ''',
                    self._safe_uuid(element.id),
                    self._safe_uuid(getattr(element, "thread_id", None)),
                    getattr(element, "type", None),
                    getattr(element, "url", None),
                    getattr(element, "chainlit_key", None),
                    element.name,
                    getattr(element, "display", None),
                    getattr(element, "object_key", None),
                    getattr(element, "size", None),
                    getattr(element, "page", None),
                    getattr(element, "language", None),
                    self._safe_uuid(getattr(element, "for_id", None)),
                    getattr(element, "mime", None),
                    _to_json(getattr(element, "props", None)),
                )
        except Exception:
            logger.error(f"create_element error:\n{traceback.format_exc()}")

    async def get_element(
        self, thread_id: str, element_id: str
    ) -> Optional[dict]:
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    'SELECT * FROM "Element" WHERE "id" = $1 AND "threadId" = $2',
                    self._safe_uuid(element_id),
                    self._safe_uuid(thread_id),
                )
            if not row:
                return None
            return {
                "id": str(row["id"]),
                "threadId": str(row["threadId"]) if row["threadId"] else None,
                "type": row["type"],
                "url": row["url"],
                "chainlitKey": row["chainlitKey"],
                "name": row["name"],
                "display": row["display"],
                "objectKey": row["objectKey"],
                "size": row["size"],
                "page": row["page"],
                "language": row["language"],
                "forId": str(row["forId"]) if row["forId"] else None,
                "mime": row["mime"],
                "props": row["props"],
            }
        except Exception:
            logger.error(f"get_element error:\n{traceback.format_exc()}")
            return None

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    'DELETE FROM "Element" WHERE "id" = $1',
                    self._safe_uuid(element_id),
                )
        except Exception:
            logger.error(f"delete_element error:\n{traceback.format_exc()}")

    # ── Step ─────────────────────────────────────────────────────────

    async def create_step(self, step_dict: dict):
        try:
            pool = await self._get_pool()
            thread_id = self._safe_uuid(step_dict.get("threadId"))
            step_id = self._safe_uuid(step_dict.get("id")) or uuid.uuid4()

            if not thread_id:
                logger.warning("create_step: no threadId, skipping")
                return

            # Normalize metadata/generation to JSON strings for ::jsonb cast
            meta = step_dict.get("metadata", {})
            gen = step_dict.get("generation")

            async with pool.acquire() as conn:
                # Ensure thread row exists
                await conn.execute(
                    '''
                    INSERT INTO "Thread" ("id", "createdAt")
                    VALUES ($1, $2)
                    ON CONFLICT ("id") DO NOTHING
                    ''',
                    thread_id,
                    self._now_iso(),
                )

                await conn.execute(
                    '''
                    INSERT INTO "Step" (
                        "id", "name", "type", "threadId", "parentId",
                        "streaming", "waitForAnswer", "isError",
                        "metadata", "tags", "input", "output",
                        "createdAt", "start", "end", "generation",
                        "showInput", "language", "indent"
                    ) VALUES (
                        $1,$2,$3,$4,$5,$6,$7,$8,
                        $9::jsonb,$10,$11,$12,$13,$14,$15,
                        $16::jsonb,$17,$18,$19
                    )
                    ON CONFLICT ("id") DO UPDATE SET
                        "output" = EXCLUDED."output",
                        "end" = EXCLUDED."end",
                        "metadata" = EXCLUDED."metadata"
                    ''',
                    step_id,
                    step_dict.get("name", ""),
                    step_dict.get("type", "undefined"),
                    thread_id,
                    self._safe_uuid(step_dict.get("parentId")),
                    step_dict.get("streaming", False),
                    step_dict.get("waitForAnswer"),
                    step_dict.get("isError"),
                    _to_json(meta),
                    step_dict.get("tags"),
                    step_dict.get("input", ""),
                    step_dict.get("output", ""),
                    step_dict.get("createdAt"),
                    step_dict.get("start"),
                    step_dict.get("end"),
                    _to_json(gen),
                    str(step_dict["showInput"]) if step_dict.get("showInput") is not None else None,
                    step_dict.get("language"),
                    step_dict.get("indent"),
                )
        except Exception:
            logger.error(f"create_step error:\n{traceback.format_exc()}")

    async def update_step(self, step_dict: dict):
        await self.create_step(step_dict)

    async def delete_step(self, step_id: str):
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    'DELETE FROM "Step" WHERE "id" = $1',
                    self._safe_uuid(step_id),
                )
        except Exception:
            logger.error(f"delete_step error:\n{traceback.format_exc()}")

    # ── Thread ───────────────────────────────────────────────────────

    async def get_thread_author(self, thread_id: str) -> str:
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    'SELECT "userId", "userIdentifier" FROM "Thread" WHERE "id" = $1',
                    self._safe_uuid(thread_id),
                )
            if not row:
                return ""
            # Primary: use userIdentifier directly
            if row["userIdentifier"]:
                return row["userIdentifier"]
            # Fallback: resolve identifier from PascalCase User table via userId
            if row["userId"]:
                async with pool.acquire() as conn:
                    user_row = await conn.fetchrow(
                        'SELECT "identifier" FROM "User" WHERE "id" = $1',
                        row["userId"],
                    )
                if user_row and user_row["identifier"]:
                    return user_row["identifier"]
            return ""
        except Exception:
            logger.error(f"get_thread_author error:\n{traceback.format_exc()}")
            return ""

    async def delete_thread(self, thread_id: str):
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    'DELETE FROM "Thread" WHERE "id" = $1',
                    self._safe_uuid(thread_id),
                )
        except Exception:
            logger.error(f"delete_thread error:\n{traceback.format_exc()}")

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        try:
            return await self._list_threads_impl(pagination, filters)
        except Exception:
            logger.error(f"list_threads error:\n{traceback.format_exc()}")
            return PaginatedResponse(
                pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None),
                data=[],
            )

    async def _list_threads_impl(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        pool = await self._get_pool()
        limit = pagination.first
        cursor = pagination.cursor
        user_id = filters.userId
        search = filters.search

        conditions: list[str] = []
        params: list = []
        idx = 1

        if user_id:
            conditions.append(f'"userIdentifier" = ${idx}')
            params.append(user_id)
            idx += 1

        if search:
            conditions.append(f'"name" ILIKE ${idx}')
            params.append(f"%{search}%")
            idx += 1

        if cursor:
            conditions.append(f'"createdAt" < ${idx}')
            params.append(cursor)
            idx += 1

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        query = f'''
            SELECT "id", "createdAt", "name", "userId",
                   "userIdentifier", "tags", "metadata"
            FROM "Thread"
            {where}
            ORDER BY "createdAt" DESC
            LIMIT ${idx}
        '''
        params.append(limit + 1)

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        has_next = len(rows) > limit
        rows = rows[:limit]

        threads: List[ThreadDict] = []
        for row in rows:
            tid = str(row["id"])

            async with pool.acquire() as conn:
                step_rows = await conn.fetch(
                    '''
                    SELECT "id", "name", "type", "threadId", "parentId",
                           "streaming", "waitForAnswer", "isError",
                           "metadata", "tags", "input", "output",
                           "createdAt", "start", "end", "generation",
                           "showInput", "language"
                    FROM "Step"
                    WHERE "threadId" = $1
                    ORDER BY "createdAt" ASC
                    ''',
                    row["id"],
                )

            steps = self._rows_to_step_dicts(step_rows)

            threads.append({
                "id": tid,
                "createdAt": row["createdAt"] or "",
                "name": row["name"],
                "userId": str(row["userId"]) if row["userId"] else None,
                "userIdentifier": row["userIdentifier"],
                "tags": row["tags"],
                "metadata": _from_json(row["metadata"], {}),
                "steps": steps,
                "elements": [],
            })

        return PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=has_next,
                startCursor=threads[0]["createdAt"] if threads else None,
                endCursor=threads[-1]["createdAt"] if threads else None,
            ),
            data=threads,
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        try:
            return await self._get_thread_impl(thread_id)
        except Exception:
            logger.error(f"get_thread error:\n{traceback.format_exc()}")
            return None

    async def _get_thread_impl(self, thread_id: str) -> Optional[ThreadDict]:
        pool = await self._get_pool()
        tid = self._safe_uuid(thread_id)
        if not tid:
            return None

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                '''
                SELECT "id", "createdAt", "name", "userId",
                       "userIdentifier", "tags", "metadata"
                FROM "Thread" WHERE "id" = $1
                ''',
                tid,
            )
        if not row:
            # Return a minimal thread object instead of failing hard
            return {
                "id": str(tid),
                "createdAt": self._now_iso(),
                "name": "Sessão",
                "userId": None,
                "userIdentifier": None,
                "tags": None,
                "metadata": {},
                "steps": [],
                "elements": [],
            }

        async with pool.acquire() as conn:
            step_rows = await conn.fetch(
                '''
                SELECT "id", "name", "type", "threadId", "parentId",
                       "streaming", "waitForAnswer", "isError",
                       "metadata", "tags", "input", "output",
                       "createdAt", "start", "end", "generation",
                       "showInput", "language"
                FROM "Step"
                WHERE "threadId" = $1
                ORDER BY "createdAt" ASC
                ''',
                tid,
            )
            element_rows = await conn.fetch(
                'SELECT * FROM "Element" WHERE "threadId" = $1',
                tid,
            )

        steps = self._rows_to_step_dicts(step_rows)

        elements = []
        for e in element_rows:
            try:
                elements.append({
                    "id": str(e["id"]),
                    "threadId": str(e["threadId"]) if e["threadId"] else None,
                    "type": e.get("type"),
                    "url": e.get("url"),
                    "chainlitKey": e.get("chainlitKey"),
                    "name": e.get("name", ""),
                    "display": e.get("display"),
                    "objectKey": e.get("objectKey"),
                    "size": e.get("size"),
                    "page": e.get("page"),
                    "language": e.get("language"),
                    "forId": str(e["forId"]) if e.get("forId") else None,
                    "mime": e.get("mime"),
                    "props": e.get("props"),
                })
            except Exception:
                logger.warning(f"Skipping malformed element: {traceback.format_exc()}")

        return {
            "id": str(row["id"]),
            "createdAt": row["createdAt"] or "",
            "name": row["name"],
            "userId": str(row["userId"]) if row["userId"] else None,
            "userIdentifier": row["userIdentifier"],
            "tags": row["tags"],
            "metadata": _from_json(row["metadata"], {}),
            "steps": steps,
            "elements": elements,
        }

    def _rows_to_step_dicts(self, step_rows) -> list:
        """Convert asyncpg step rows to StepDict list."""
        steps = []
        for s in step_rows:
            try:
                steps.append({
                    "id": str(s["id"]),
                    "name": s["name"] or "",
                    "type": s["type"] or "undefined",
                    "threadId": str(s["threadId"]),
                    "parentId": str(s["parentId"]) if s["parentId"] else None,
                    "streaming": s["streaming"] if s["streaming"] is not None else False,
                    "waitForAnswer": s.get("waitForAnswer"),
                    "isError": s.get("isError"),
                    "metadata": _from_json(s["metadata"], {}),
                    "tags": s.get("tags"),
                    "input": s["input"] or "",
                    "output": s["output"] or "",
                    "createdAt": s["createdAt"],
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "generation": _from_json(s.get("generation"), None),
                    "showInput": s.get("showInput"),
                    "language": s.get("language"),
                })
            except Exception:
                logger.warning(f"Skipping malformed step: {traceback.format_exc()}")
        return steps

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        try:
            await self._update_thread_impl(thread_id, name, user_id, metadata, tags)
        except Exception:
            logger.error(f"update_thread error:\n{traceback.format_exc()}")

    async def _update_thread_impl(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        pool = await self._get_pool()
        tid = self._safe_uuid(thread_id)
        if not tid:
            return
        now = self._now_iso()

        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                'SELECT "id" FROM "Thread" WHERE "id" = $1', tid
            )

            if existing:
                sets: list[str] = []
                params: list = []
                idx = 1

                if name is not None:
                    sets.append(f'"name" = ${idx}')
                    params.append(name)
                    idx += 1
                if user_id is not None:
                    user_row = await conn.fetchrow(
                        'SELECT "id" FROM "User" WHERE "identifier" = $1',
                        user_id,
                    )
                    if user_row:
                        sets.append(f'"userId" = ${idx}')
                        params.append(user_row["id"])
                        idx += 1
                    sets.append(f'"userIdentifier" = ${idx}')
                    params.append(user_id)
                    idx += 1
                if metadata is not None:
                    sets.append(f'"metadata" = ${idx}::jsonb')
                    params.append(_to_json(metadata))
                    idx += 1
                if tags is not None:
                    sets.append(f'"tags" = ${idx}')
                    params.append(tags)
                    idx += 1

                if sets:
                    params.append(tid)
                    query = f'UPDATE "Thread" SET {", ".join(sets)} WHERE "id" = ${idx}'
                    await conn.execute(query, *params)
            else:
                user_uuid = None
                if user_id:
                    user_row = await conn.fetchrow(
                        'SELECT "id" FROM "User" WHERE "identifier" = $1',
                        user_id,
                    )
                    if user_row:
                        user_uuid = user_row["id"]

                await conn.execute(
                    '''
                    INSERT INTO "Thread"
                        ("id", "createdAt", "name", "userId",
                         "userIdentifier", "tags", "metadata")
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                    ''',
                    tid,
                    now,
                    name,
                    user_uuid,
                    user_id,
                    tags,
                    _to_json(metadata or {}),
                )

    # ── Utility ──────────────────────────────────────────────────────

    async def build_debug_url(self) -> str:
        return ""

    async def get_favorite_steps(self, user_id: str) -> List[dict]:
        return []

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
