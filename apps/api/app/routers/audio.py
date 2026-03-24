import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..config import get_settings, Settings
from ..core.prompts import (
    build_system_prompt,
    extract_leito_number,
    extract_leitos_structured,
)
from ..schemas import coerce_confirmed_leitos
from ..core.summarization import stream_summary
from ..core.transcription import transcribe_audio
from ..database import get_db
from ..models import User
from ..repositories.message import MessageRepository
from ..repositories.thread import ThreadRepository

router = APIRouter(tags=["audio"])
logger = logging.getLogger(__name__)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/threads/{thread_id}/audio")
async def upload_audio(
    thread_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """
    Accept an audio upload, transcribe it, then stream SSE events:
      - {"type": "transcription", "text": "..."}
      - {"type": "delta", "text": "..."}  (one per Claude token)
      - {"type": "done", "thread_title": "...", "confirmed_leitos": {...}}
    """
    thread_repo = ThreadRepository(db)
    thread = thread_repo.get_thread(thread_id)
    if not thread or thread.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Thread not found")

    audio_bytes = await file.read()
    filename = file.filename or "audio.wav"

    msg_repo = MessageRepository(db)
    messages_db = msg_repo.get_thread_messages(thread_id, limit=settings.max_history_messages)
    history = [{"role": m.role, "content": m.content} for m in messages_db]

    confirmed_leitos: dict = coerce_confirmed_leitos(dict(thread.confirmed_leitos or {}))

    async def event_stream() -> AsyncGenerator[str, None]:
        nonlocal confirmed_leitos

        # Transcribe
        try:
            transcription = transcribe_audio(
                audio_bytes=audio_bytes,
                filename=filename,
                groq_api_key=settings.groq_api_key,
            )
        except Exception as exc:
            logger.error(f"Transcription error: {exc}")
            yield _sse({"type": "error", "message": str(exc)})
            return

        yield _sse({"type": "transcription", "text": transcription})

        user_content = f"[Transcrição do áudio]\n\n{transcription}"

        msg_repo.add_message(
            thread_id=thread_id,
            role="user",
            content=user_content,
            has_audio=True,
            audio_filename=filename,
            transcription=transcription,
        )

        # Build API messages (stub out already-confirmed leitos to save tokens)
        confirmed_nums = set(confirmed_leitos.keys())
        api_messages = []
        for m in history:
            if m["role"] == "assistant":
                num = extract_leito_number(m["content"])
                if num and num in confirmed_nums:
                    api_messages.append({"role": "assistant", "content": f"[Leito {num} confirmado]"})
                    continue
            api_messages.append({"role": m["role"], "content": m["content"]})
        api_messages.append({"role": "user", "content": user_content})

        # Stream Claude response
        from anthropic import Anthropic
        client = Anthropic(api_key=settings.anthropic_api_key)
        system_prompt = build_system_prompt(confirmed_leitos=confirmed_leitos)

        full_response = ""
        try:
            async for chunk in stream_summary(
                client=client,
                messages=api_messages,
                system_prompt=system_prompt,
                model=settings.claude_model,
            ):
                full_response += chunk
                yield _sse({"type": "delta", "text": chunk})
        except Exception as exc:
            logger.error(f"Claude streaming error: {exc}")
            yield _sse({"type": "error", "message": str(exc)})
            return

        msg_repo.add_message(thread_id=thread_id, role="assistant", content=full_response)

        # Extract structured leito data and persist
        try:
            new_leitos_raw = extract_leitos_structured(full_response, client, settings.claude_model)
            for leito_data in new_leitos_raw:
                leito_num = leito_data["leito"]
                confirmed_leitos[leito_num] = leito_data
            if new_leitos_raw:
                thread_repo.update_confirmed_leitos(thread_id, confirmed_leitos)
        except Exception as exc:
            logger.error(f"Structured extraction error: {exc}")

        thread_repo.update_thread_from_response(thread_id, full_response)
        db.refresh(thread)

        yield _sse({
            "type": "done",
            "thread_title": thread.title or "Nova Sessão",
            "confirmed_leitos": confirmed_leitos,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")
