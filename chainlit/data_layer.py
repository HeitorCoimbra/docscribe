"""
Custom Chainlit data layer for DocScribe.

Uses asyncpg to interact with the PascalCase Chainlit tables
(User, Thread, Step, Element, Feedback) for sidebar history support.
"""

import json
import logging
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
from chainlit.user import PersistedUser

logger = logging.getLogger(__name__)


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
    def _parse_json(val):
        """Parse a JSON value that may be a string or already a dict."""
        if val is None:
            return {}
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, ValueError):
                return {}
        return val

    # ── User ─────────────────────────────────────────────────────────

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
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
            metadata=self._parse_json(row["metadata"]),
            createdAt=row["createdAt"] or "",
        )

    async def create_user(self, user) -> Optional[PersistedUser]:
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
            metadata=self._parse_json(row["metadata"]),
            createdAt=row["createdAt"] or now,
        )

    # ── Feedback ─────────────────────────────────────────────────────

    async def upsert_feedback(self, feedback: Feedback) -> str:
        pool = await self._get_pool()
        fb_id = uuid.UUID(feedback.id) if feedback.id else uuid.uuid4()
        async with pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO "Feedback" ("id", "forId", "threadId", "value", "comment")
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT ("id") DO UPDATE
                    SET "value" = $4, "comment" = $5
                ''',
                fb_id,
                uuid.UUID(feedback.forId),
                uuid.UUID(feedback.threadId) if feedback.threadId else None,
                feedback.value,
                feedback.comment,
            )
        return str(fb_id)

    async def delete_feedback(self, feedback_id: str) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                'DELETE FROM "Feedback" WHERE "id" = $1',
                uuid.UUID(feedback_id),
            )
        return "DELETE 1" in result

    # ── Element ──────────────────────────────────────────────────────

    async def create_element(self, element):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO "Element" (
                    "id", "threadId", "type", "url", "chainlitKey",
                    "name", "display", "objectKey", "size", "page",
                    "language", "forId", "mime", "props"
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                ON CONFLICT ("id") DO NOTHING
                ''',
                uuid.UUID(element.id),
                uuid.UUID(element.thread_id) if getattr(element, "thread_id", None) else None,
                getattr(element, "type", None),
                getattr(element, "url", None),
                getattr(element, "chainlit_key", None),
                element.name,
                getattr(element, "display", None),
                getattr(element, "object_key", None),
                getattr(element, "size", None),
                getattr(element, "page", None),
                getattr(element, "language", None),
                uuid.UUID(element.for_id) if getattr(element, "for_id", None) else None,
                getattr(element, "mime", None),
                json.dumps(element.props) if getattr(element, "props", None) else None,
            )

    async def get_element(
        self, thread_id: str, element_id: str
    ) -> Optional[dict]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                'SELECT * FROM "Element" WHERE "id" = $1 AND "threadId" = $2',
                uuid.UUID(element_id),
                uuid.UUID(thread_id),
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

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                'DELETE FROM "Element" WHERE "id" = $1',
                uuid.UUID(element_id),
            )

    # ── Step ─────────────────────────────────────────────────────────

    async def create_step(self, step_dict: dict):
        pool = await self._get_pool()
        thread_id = uuid.UUID(step_dict["threadId"])
        step_id = uuid.UUID(step_dict.get("id", str(uuid.uuid4())))

        async with pool.acquire() as conn:
            # Ensure thread row exists (Chainlit may create steps
            # before explicitly calling update_thread)
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
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)
                ON CONFLICT ("id") DO UPDATE SET
                    "output" = EXCLUDED."output",
                    "end" = EXCLUDED."end",
                    "metadata" = EXCLUDED."metadata"
                ''',
                step_id,
                step_dict.get("name", ""),
                step_dict.get("type", "undefined"),
                thread_id,
                uuid.UUID(step_dict["parentId"]) if step_dict.get("parentId") else None,
                step_dict.get("streaming", False),
                step_dict.get("waitForAnswer"),
                step_dict.get("isError"),
                json.dumps(step_dict.get("metadata", {})),
                step_dict.get("tags"),
                step_dict.get("input", ""),
                step_dict.get("output", ""),
                step_dict.get("createdAt"),
                step_dict.get("start"),
                step_dict.get("end"),
                json.dumps(step_dict["generation"]) if step_dict.get("generation") else None,
                str(step_dict["showInput"]) if step_dict.get("showInput") is not None else None,
                step_dict.get("language"),
                step_dict.get("indent"),
            )

    async def update_step(self, step_dict: dict):
        await self.create_step(step_dict)

    async def delete_step(self, step_id: str):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                'DELETE FROM "Step" WHERE "id" = $1',
                uuid.UUID(step_id),
            )

    # ── Thread ───────────────────────────────────────────────────────

    async def get_thread_author(self, thread_id: str) -> str:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                'SELECT "userIdentifier" FROM "Thread" WHERE "id" = $1',
                uuid.UUID(thread_id),
            )
        return (row["userIdentifier"] if row and row["userIdentifier"] else "")

    async def delete_thread(self, thread_id: str):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                'DELETE FROM "Thread" WHERE "id" = $1',
                uuid.UUID(thread_id),
            )

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        pool = await self._get_pool()
        limit = pagination.first
        cursor = pagination.cursor  # createdAt ISO of last seen thread
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

            # Fetch steps for this thread
            async with pool.acquire() as conn:
                step_rows = await conn.fetch(
                    '''
                    SELECT "id", "name", "type", "threadId", "parentId",
                           "streaming", "waitForAnswer", "isError",
                           "metadata", "tags", "input", "output",
                           "createdAt", "start", "end", "generation",
                           "showInput", "language", "indent"
                    FROM "Step"
                    WHERE "threadId" = $1
                    ORDER BY "createdAt" ASC
                    ''',
                    row["id"],
                )

            steps = []
            for s in step_rows:
                steps.append({
                    "id": str(s["id"]),
                    "name": s["name"],
                    "type": s["type"],
                    "threadId": str(s["threadId"]),
                    "parentId": str(s["parentId"]) if s["parentId"] else None,
                    "streaming": s["streaming"],
                    "waitForAnswer": s["waitForAnswer"],
                    "isError": s["isError"],
                    "metadata": self._parse_json(s["metadata"]),
                    "tags": s["tags"],
                    "input": s["input"] or "",
                    "output": s["output"] or "",
                    "createdAt": s["createdAt"],
                    "start": s["start"],
                    "end": s["end"],
                    "generation": self._parse_json(s["generation"]),
                    "showInput": s["showInput"],
                    "language": s["language"],
                })

            threads.append({
                "id": tid,
                "createdAt": row["createdAt"] or "",
                "name": row["name"],
                "userId": str(row["userId"]) if row["userId"] else None,
                "userIdentifier": row["userIdentifier"],
                "tags": row["tags"],
                "metadata": self._parse_json(row["metadata"]),
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
        pool = await self._get_pool()
        tid = uuid.UUID(thread_id)

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
            return None

        async with pool.acquire() as conn:
            step_rows = await conn.fetch(
                'SELECT * FROM "Step" WHERE "threadId" = $1 ORDER BY "createdAt" ASC',
                tid,
            )
            element_rows = await conn.fetch(
                'SELECT * FROM "Element" WHERE "threadId" = $1',
                tid,
            )

        steps = []
        for s in step_rows:
            steps.append({
                "id": str(s["id"]),
                "name": s["name"],
                "type": s["type"],
                "threadId": str(s["threadId"]),
                "parentId": str(s["parentId"]) if s["parentId"] else None,
                "streaming": s["streaming"],
                "waitForAnswer": s["waitForAnswer"],
                "isError": s["isError"],
                "metadata": self._parse_json(s["metadata"]),
                "tags": s["tags"],
                "input": s["input"] or "",
                "output": s["output"] or "",
                "createdAt": s["createdAt"],
                "start": s["start"],
                "end": s["end"],
                "generation": self._parse_json(s["generation"]),
                "showInput": s["showInput"],
                "language": s["language"],
            })

        elements = []
        for e in element_rows:
            elements.append({
                "id": str(e["id"]),
                "threadId": str(e["threadId"]) if e["threadId"] else None,
                "type": e["type"],
                "url": e["url"],
                "chainlitKey": e["chainlitKey"],
                "name": e["name"],
                "display": e["display"],
                "objectKey": e["objectKey"],
                "size": e["size"],
                "page": e["page"],
                "language": e["language"],
                "forId": str(e["forId"]) if e["forId"] else None,
                "mime": e["mime"],
                "props": e["props"],
            })

        return {
            "id": str(row["id"]),
            "createdAt": row["createdAt"] or "",
            "name": row["name"],
            "userId": str(row["userId"]) if row["userId"] else None,
            "userIdentifier": row["userIdentifier"],
            "tags": row["tags"],
            "metadata": self._parse_json(row["metadata"]),
            "steps": steps,
            "elements": elements,
        }

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        pool = await self._get_pool()
        tid = uuid.UUID(thread_id)
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
                    # Look up User UUID from identifier
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
                    params.append(json.dumps(metadata))
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
                # Insert new thread
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
                    json.dumps(metadata or {}),
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
