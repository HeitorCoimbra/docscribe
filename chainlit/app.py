"""
DocScribe Chainlit App - Medical Audio Summarization with OAuth and Session History.

Features:
- OAuth authentication (Google, GitHub)
- Session history with PostgreSQL persistence
- Sessions titled "Leito X - Name" grouped by day
- Audio upload with Groq Whisper transcription
- Claude-powered extraction
"""

import os
import sys
import json
import logging
import uuid
import base64
import mimetypes
import io
import wave
import asyncio
from datetime import datetime, timezone
from typing import Optional

# Force UTF-8 for stdout/stderr in containers with ASCII locale
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure the app directory is in sys.path so lazy imports (database, core)
# work correctly when Chainlit loads the module via importlib.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chainlit as cl
from chainlit.types import ThreadDict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable audio recording and file upload in Chainlit config
# (auto-generated config.toml has these disabled by default)
try:
    from chainlit.config import config as chainlit_config
    chainlit_config.features.audio.enabled = True
    chainlit_config.features.audio.sample_rate = 24000
    chainlit_config.features.spontaneous_file_upload.enabled = True
    chainlit_config.features.spontaneous_file_upload.accept = ["audio/*"]
    chainlit_config.ui.custom_js = "/public/group-threads-by-date.js"
    chainlit_config.ui.custom_css = "/public/group-threads-by-date.css"
except Exception as e:
    logger.warning(f"Could not configure audio/upload features: {e}")

# =============================================================================
# CONFIGURATION
# =============================================================================

def _clean_key(val: str | None) -> str | None:
    """Remove invisible Unicode chars that sneak in from copy-paste."""
    if not val:
        return None
    # Remove common invisible characters: line/paragraph separators, ZWS, BOM, etc.
    for ch in "\u2028\u2029\u200b\ufeff\u00a0":
        val = val.replace(ch, "")
    return val.strip() or None

GROQ_API_KEY = _clean_key(os.environ.get("GROQ_API_KEY"))
ANTHROPIC_API_KEY = _clean_key(os.environ.get("ANTHROPIC_API_KEY"))
DATABASE_URL = os.environ.get("DOCSCRIBE_DB_URL", os.environ.get("DATABASE_URL"))
MAX_HISTORY_MESSAGES = 50

# Log configuration status
logger.info(f"GROQ_API_KEY configured: {bool(GROQ_API_KEY)}")
logger.info(f"ANTHROPIC_API_KEY configured: {bool(ANTHROPIC_API_KEY)}")
logger.info(f"DOCSCRIBE_DB_URL configured: {bool(DATABASE_URL)}")

# =============================================================================
# LAZY INITIALIZATION
# =============================================================================

_anthropic_client = None
_db_initialized = False


def get_anthropic_client():
    """Get or create Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None and ANTHROPIC_API_KEY:
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def ensure_db():
    """Initialize database if not already done."""
    global _db_initialized
    if not _db_initialized and DATABASE_URL:
        try:
            from database import init_db
            init_db()
            _db_initialized = True
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    return _db_initialized


def get_db_session():
    """Get a database session if available."""
    if ensure_db():
        try:
            from database import SessionLocal
            return SessionLocal()
        except Exception as e:
            logger.error(f"Failed to create DB session: {e}")
    return None


# Initialize DB eagerly so Chainlit's built-in data layer tables exist
# before its first query (which happens during authentication).
ensure_db()

# Register Chainlit data layer for sidebar history
if DATABASE_URL:
    try:
        import chainlit.data as cl_data
        from data_layer import DocScribeDataLayer
        cl_data._data_layer = DocScribeDataLayer(conninfo=DATABASE_URL)
        logger.info("Chainlit data layer registered for sidebar history")
    except Exception as e:
        logger.warning(f"Could not register data layer (sidebar history disabled): {e}")


# =============================================================================
# IMPORT CORE MODULE
# =============================================================================

from core import (
    SumarioPaciente,
    SYSTEM_PROMPT,
    transcribe_audio,
    CLAUDE_MODEL,
    generate_pdf_report,
)

# =============================================================================
# CHAT SYSTEM PROMPT
# =============================================================================

import re


def extract_summary_from_response(response: str) -> str | None:
    """Extract the structured summary block from Claude's response."""
    pattern = r'(\*\*Leito\s+.+?)(?:\n---|\Z)'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_leito_number(summary: str) -> str | None:
    """Extract leito number from a formatted summary string."""
    match = re.search(r'\*\*Leito\s+([^\s*-]+)', summary)
    return match.group(1).strip() if match else None


def detect_referenced_leito(text: str, confirmed_leitos: dict) -> str | None:
    """Return leito number if text explicitly references a confirmed leito."""
    for leito_num in confirmed_leitos:
        if re.search(rf'\bleito\s+{re.escape(leito_num)}\b', text, re.IGNORECASE):
            return leito_num
    return None


def build_system_prompt(
    confirmed_leitos: dict | None = None,
    current_summary: str | None = None,
    edit_target_leito: str | None = None
) -> str:
    """Build the system prompt with confirmed leitos context and current editable summary."""
    confirmed_context = ""
    if confirmed_leitos:
        if edit_target_leito and edit_target_leito in confirmed_leitos:
            others = {k: v for k, v in confirmed_leitos.items() if k != edit_target_leito}
            target_summary = confirmed_leitos[edit_target_leito]
            if others:
                block = "\n\n".join(others.values())
                confirmed_context = f"""
=== LEITOS CONFIRMADOS (NÃO REPRODUZA NA RESPOSTA) ===
{block}
=== FIM ===\n"""
            confirmed_context += f"""
=== LEITO {edit_target_leito} EM REEDIÇÃO (atualize com as novas informações) ===
{target_summary}
=== FIM ==="""
        else:
            block = "\n\n".join(confirmed_leitos.values())
            confirmed_context = f"""
=== LEITOS JÁ CONFIRMADOS (NÃO REPRODUZA ESTES NA RESPOSTA) ===
{block}
=== FIM DOS LEITOS CONFIRMADOS ===

IMPORTANTE: Não repita os leitos acima. Sua resposta deve conter SOMENTE o novo leito da transcrição atual."""

    current_context = ""
    if current_summary and not edit_target_leito:
        current_context = f"""
=== SUMÁRIO DO LEITO EM EDIÇÃO (use como base para atualizações) ===
{current_summary}
=== FIM DO SUMÁRIO EM EDIÇÃO ==="""

    current_date = datetime.now().strftime("%d/%m/%Y")

    return f"""Data de hoje: {current_date}

{SYSTEM_PROMPT}

=== INSTRUCOES DE COMPORTAMENTO ===

Quando receber uma transcrição de áudio, responda IMEDIATAMENTE com o sumário do NOVO leito apenas.
Não repita leitos já confirmados. Não faça perguntas antes de mostrar o sumário.

Para campos não preenchíveis, use: 🔴 PENDENTE

FORMATO OBRIGATÓRIO DE RESPOSTA (UM LEITO POR VEZ):

**Leito [X] - [Nome do Paciente]**

**Quadro Clínico:**
1. [problema clínico extraído ou 🔴 PENDENTE]

**Pendências:**
1. [pendência extraída ou 🔴 PENDENTE]

**Condutas:**
- [conduta com verbo no infinitivo ou 🔴 PENDENTE]

---
Deseja corrigir ou adicionar algo?

=== REGRAS DE ATUALIZAÇÃO ===
- Quando o usuário enviar correções, mostre o sumário COMPLETO do leito atual atualizado (não todos os leitos)
- Condutas SEMPRE começam com verbo no INFINITIVO
- Seja conciso. Não repita instruções ao usuário.

{confirmed_context}
{current_context}"""


# =============================================================================
# OAUTH AUTHENTICATION
# =============================================================================

# Check if any OAuth provider is configured
_has_oauth = any(
    os.environ.get(var)
    for var in [
        "OAUTH_GOOGLE_CLIENT_ID",
        "OAUTH_GITHUB_CLIENT_ID",
    ]
)

if _has_oauth:
    @cl.oauth_callback
    def oauth_callback(
        provider_id: str,
        token: str,
        raw_user_data: dict,
        default_user: cl.User,
    ) -> Optional[cl.User]:
        """Handle OAuth callback and create/update user in database."""
        email = raw_user_data.get("email")
        name = raw_user_data.get("name") or raw_user_data.get("login")
        avatar = raw_user_data.get("picture") or raw_user_data.get("avatar_url")

        if not email:
            logger.warning("OAuth callback: no email provided")
            return default_user

        # Use a stable identifier (email) for Chainlit auth/data layer
        stable_identifier = email

        # Try to save to database if available
        db = get_db_session()
        db_user_id = None
        if db:
            try:
                from database import UserRepository
                user_repo = UserRepository(db)
                db_user = user_repo.get_or_create_user(
                    email=email,
                    name=name,
                    avatar_url=avatar,
                    provider=provider_id
                )
                db_user_id = db_user.id
            except Exception as e:
                logger.error(f"Database error in oauth_callback: {e}")
            finally:
                db.close()

        return cl.User(
            identifier=stable_identifier,
            metadata={
                "email": email,
                "name": name,
                "avatar": avatar,
                "provider": provider_id,
                "db_user_id": db_user_id,
            }
        )
else:
    logger.warning("No OAuth provider configured — running without authentication")


# =============================================================================
# BACKGROUND CLEANUP
# =============================================================================

_cleanup_task_started = False


async def _cleanup_loop(days: int = 60, interval_hours: int = 24):
    """Periodically delete threads and all related data older than `days` days."""
    await asyncio.sleep(60)  # Let the app finish initializing first
    while True:
        try:
            import chainlit.data as cl_data
            if cl_data._data_layer:
                count = await cl_data._data_layer.cleanup_old_threads(days=days)
                if count:
                    logger.info(f"Cleanup: deleted {count} thread(s) older than {days} days")
                else:
                    logger.info("Cleanup: no old threads to delete")
        except Exception as e:
            logger.error(f"Cleanup loop error: {e}")
        await asyncio.sleep(interval_hours * 3600)


# =============================================================================
# PDF EXPORT
# =============================================================================

@cl.action_callback("gerar_pdf")
async def on_gerar_pdf(action: cl.Action):
    """Generate and send a PDF of all session summaries."""
    import tempfile
    import os

    confirmed_leitos = cl.user_session.get("confirmed_leitos", {})
    current_summary = cl.user_session.get("current_summary")

    all_summaries = list(confirmed_leitos.values())
    if current_summary and current_summary not in all_summaries:
        all_summaries.append(current_summary)

    if not all_summaries:
        await cl.Message(content="⚠️ Nenhum sumário disponível para gerar o PDF.").send()
        return

    try:
        pdf_bytes = generate_pdf_report(all_summaries)
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        await cl.Message(content=f"❌ Erro ao gerar PDF: {str(e)}").send()
        return

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="docscribe_")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(pdf_bytes)
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        elements = [
            cl.File(
                name=f"sumarios_plantao_{date_str}.pdf",
                path=tmp_path,
                display="inline",
            )
        ]
        await cl.Message(
            content=f"📄 **PDF gerado** — {len(all_summaries)} leito(s):",
            elements=elements,
        ).send()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# =============================================================================
# CHAT START
# =============================================================================

@cl.on_chat_start
async def on_chat_start():
    """Initialize new chat session."""
    global _cleanup_task_started
    if not _cleanup_task_started and DATABASE_URL:
        _cleanup_task_started = True
        asyncio.create_task(_cleanup_loop())

    # Get thread ID from Chainlit context (set by data layer) or generate one
    thread_id = None
    try:
        thread_id = cl.context.session.thread_id
    except (AttributeError, RuntimeError):
        pass
    if not thread_id:
        thread_id = str(uuid.uuid4())

    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("message_history", [])
    cl.user_session.set("current_summary", None)
    cl.user_session.set("confirmed_leitos", {})
    cl.user_session.set("edit_target_leito", None)
    cl.user_session.set("thread_persisted", False)

    # Notify frontend JS to inject thread into sidebar immediately
    await cl.send_window_message({
        "type": "new_thread",
        "threadId": thread_id,
        "name": "Nova Sessão",
        "createdAt": datetime.now(timezone.utc).isoformat(),
    })

    # Welcome message
    await cl.Message(
        content="""👋 **Bem-vindo ao DocScribe!**

Sou seu assistente para criar sumários de pacientes de UTI.

**Como usar:**
1. 🎤 **Grave um áudio** usando o botão de microfone
2. 📎 Ou faça upload de um arquivo de áudio
3. 💬 Ou cole a transcrição diretamente no chat
4. ✏️ Corrija ou adicione informações conforme necessário
5. ✅ Confirme para salvar o sumário

*Dica: Você pode enviar múltiplos áudios e eu consolidarei as informações.*
"""
    ).send()


async def ensure_thread_persisted(thread_id: str):
    """Create the thread in both custom DB and data layer on first message.

    Uses a session flag so this only runs once per chat session.
    """
    if cl.user_session.get("thread_persisted"):
        return
    cl.user_session.set("thread_persisted", True)

    user = cl.user_session.get("user")
    if not user:
        return

    # Custom DB table (domain-specific data)
    db = get_db_session()
    if db:
        try:
            from database import ThreadRepository, UserRepository
            user_repo = UserRepository(db)
            metadata = user.metadata or {}
            db_user = user_repo.get_or_create_user(
                email=metadata.get("email", user.identifier),
                name=metadata.get("name"),
                avatar_url=metadata.get("avatar"),
                provider=metadata.get("provider"),
            )
            thread_repo = ThreadRepository(db)
            thread_repo.create_thread_with_id(
                thread_id=thread_id,
                user_id=db_user.id,
                title="Nova Sessão"
            )
        except Exception as e:
            logger.error(f"Failed to create thread in DB: {e}")
        finally:
            db.close()

    # Data layer (powers the sidebar)
    try:
        import chainlit.data as cl_data
        if cl_data._data_layer:
            await cl_data._data_layer.update_thread(
                thread_id=thread_id,
                user_id=user.identifier,
                name="Nova Sessão",
            )
    except Exception as e:
        logger.error(f"Failed to register thread in data layer: {e}")


# =============================================================================
# AUDIO PROCESSING HELPER
# =============================================================================

async def process_audio_and_transcribe(
    audio_bytes: bytes,
    filename: str,
    source_label: str = "áudio"
) -> Optional[str]:
    """
    Process audio bytes, transcribe it, and display the result.
    Returns the transcription text if successful, None otherwise.
    """
    processing_msg = cl.Message(content=f"🎤 Transcrevendo {source_label}...")
    await processing_msg.send()
    
    try:
        # Transcribe with Groq Whisper
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY não configurada")
        
        transcription = transcribe_audio(
            audio_bytes=audio_bytes,
            filename=filename,
            groq_api_key=GROQ_API_KEY
        )
        
        await processing_msg.remove()
        
        # Display transcription in expandable accordion
        transcription_element = cl.CustomElement(
            name="TranscriptionAccordion",
            props={
                "transcription": transcription,
                "characterCount": len(transcription),
                "audioBase64": base64.b64encode(audio_bytes).decode("utf-8"),
                "audioMime": mimetypes.guess_type(filename)[0] or "audio/wav",
                "label": source_label.capitalize(),
            }
        )
        await cl.Message(
            content=f"✅ **{source_label.capitalize()} transcrito:**",
            elements=[transcription_element]
        ).send()
        
        return transcription
        
    except Exception as e:
        import traceback
        logger.error(f"Transcription error:\n{traceback.format_exc()}")
        await processing_msg.remove()
        await cl.Message(content=f"❌ Erro na transcrição do {source_label}: {str(e)}").send()
        return None


def pcm_to_wav_bytes(
    pcm_bytes: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2
) -> bytes:
    """Wrap raw PCM bytes into a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buffer.getvalue()


# =============================================================================
# AUDIO RECORDING HANDLERS
# =============================================================================

@cl.on_audio_start
async def on_audio_start():
    """Allow audio recording to begin."""
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    """Collect audio chunks during recording."""
    if chunk.isStart:
        # Initialize audio buffer for new recording
        cl.user_session.set("audio_chunks", [])
        cl.user_session.set("audio_mime", getattr(chunk, "mimeType", None))
        cl.user_session.set("audio_sample_rate", getattr(chunk, "sampleRate", None))
        cl.user_session.set("audio_channels", getattr(chunk, "channels", None))
        cl.user_session.set("audio_sample_width", getattr(chunk, "sampleWidth", None))

    # Collect chunk data
    chunks = cl.user_session.get("audio_chunks", [])
    chunks.append(chunk.data)
    cl.user_session.set("audio_chunks", chunks)


@cl.on_audio_end
async def on_audio_end():
    """Process complete audio recording and transcribe it."""
    chunks = cl.user_session.get("audio_chunks", [])
    sample_rate = cl.user_session.get("audio_sample_rate") or 24000
    channels = cl.user_session.get("audio_channels") or 1
    sample_width = cl.user_session.get("audio_sample_width") or 2
    
    if not chunks:
        await cl.Message(content="❌ Nenhum áudio foi gravado.").send()
        return
    
    # Combine all chunks into single audio bytes
    pcm_bytes = b"".join(chunks)
    
    # Clear chunks from session
    cl.user_session.set("audio_chunks", None)
    cl.user_session.set("audio_mime", None)
    cl.user_session.set("audio_sample_rate", None)
    cl.user_session.set("audio_channels", None)
    cl.user_session.set("audio_sample_width", None)
    
    # Chainlit streams raw PCM; wrap it into a WAV container
    audio_bytes = pcm_to_wav_bytes(
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width
    )
    filename = "recorded_audio.wav"
    
    # Process the recorded audio
    transcription = await process_audio_and_transcribe(
        audio_bytes=audio_bytes,
        filename=filename,
        source_label="áudio gravado"
    )
    
    if transcription:
        # Persist a user_message step so the transcription appears on resume
        thread_id = cl.user_session.get("thread_id")
        if thread_id:
            try:
                import chainlit.data as cl_data
                if cl_data._data_layer:
                    await cl_data._data_layer.create_step({
                        "id": str(uuid.uuid4()),
                        "threadId": thread_id,
                        "type": "user_message",
                        "output": f"[Transcrição do áudio gravado]\n\n{transcription}",
                        "createdAt": datetime.now(timezone.utc).isoformat(),
                    })
            except Exception as e:
                logger.error(f"Failed to persist transcription step: {e}")

        # Process the transcription as if it came from a message
        synthetic_message = cl.Message(
            content=f"[Transcrição do áudio gravado]\n\n{transcription}"
        )
        await on_message(synthetic_message)


# =============================================================================
# MESSAGE HANDLING
# =============================================================================

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    thread_id = cl.user_session.get("thread_id")
    
    if not thread_id:
        thread_id = str(uuid.uuid4())
        cl.user_session.set("thread_id", thread_id)
    
    # Get message history
    history = cl.user_session.get("message_history", [])
    
    # Check for audio file attachments
    audio_files = []
    if message.elements:
        audio_files = [f for f in message.elements if f.mime and f.mime.startswith("audio/")]
    
    user_content = message.content or ""
    transcriptions = []

    # Process audio if present
    if audio_files:
        await ensure_thread_persisted(thread_id)
        for i, audio_file in enumerate(audio_files, 1):
            label = f"áudio {i}" if len(audio_files) > 1 else "áudio"

            if hasattr(audio_file, 'content') and audio_file.content:
                audio_bytes = audio_file.content
            elif hasattr(audio_file, 'path') and audio_file.path:
                with open(audio_file.path, 'rb') as f:
                    audio_bytes = f.read()
            else:
                await cl.Message(content=f"❌ Não foi possível ler o {label}.").send()
                continue

            t = await process_audio_and_transcribe(
                audio_bytes=audio_bytes,
                filename=audio_file.name or f"audio_{i}.mp3",
                source_label=label
            )
            if t:
                transcriptions.append(t)

        if not transcriptions:
            return

        if len(transcriptions) == 1:
            user_content = f"[Transcrição do áudio]\n\n{transcriptions[0]}"
        else:
            parts = [f"Áudio {i}: {t}" for i, t in enumerate(transcriptions, 1)]
            user_content = f"[Transcrição dos áudios]\n\n" + "\n\n".join(parts)
    
    if not user_content.strip():
        return

    # Persist thread on first real message (shows it in sidebar)
    await ensure_thread_persisted(thread_id)

    # Add user message to history
    history.append({"role": "user", "content": user_content})

    # Save message to database
    db = get_db_session()
    if db:
        try:
            from database import MessageRepository
            msg_repo = MessageRepository(db)
            msg_repo.add_message(
                thread_id=thread_id,
                role="user",
                content=user_content,
                has_audio=bool(audio_files),
                transcription="\n\n".join(transcriptions) if transcriptions else None
            )
        except Exception as e:
            logger.error(f"Failed to save user message: {e}")
        finally:
            db.close()
    
    # Check if Anthropic is configured
    client = get_anthropic_client()
    if not client:
        await cl.Message(content="❌ ANTHROPIC_API_KEY não configurada").send()
        return
    
    # Routing: confirm previous leito on new audio, or detect targeted leito edit
    if transcriptions:
        prev = cl.user_session.get("current_summary")
        if prev:
            num = extract_leito_number(prev)
            if num:
                confirmed = cl.user_session.get("confirmed_leitos", {})
                confirmed[num] = prev
                cl.user_session.set("confirmed_leitos", confirmed)
        cl.user_session.set("current_summary", None)
        cl.user_session.set("edit_target_leito", None)
    else:
        confirmed = cl.user_session.get("confirmed_leitos", {})
        ref = detect_referenced_leito(user_content, confirmed)
        cl.user_session.set("edit_target_leito", ref)

    # Generate response with Claude
    response_msg = cl.Message(content="")
    await response_msg.send()

    try:
        # Build API messages — replace full confirmed-leito responses with stubs
        confirmed_leito_nums = set(cl.user_session.get("confirmed_leitos", {}).keys())
        api_messages = []
        for m in history:
            if m["role"] == "assistant":
                num = extract_leito_number(m["content"])
                if num and num in confirmed_leito_nums:
                    api_messages.append({"role": "assistant", "content": f"[Leito {num} confirmado]"})
                    continue
            api_messages.append({"role": m["role"], "content": m["content"]})

        # Stream response
        full_response = ""
        confirmed_leitos = cl.user_session.get("confirmed_leitos", {})
        current_summary  = cl.user_session.get("current_summary")
        edit_target      = cl.user_session.get("edit_target_leito")
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=build_system_prompt(
                confirmed_leitos=confirmed_leitos,
                current_summary=current_summary,
                edit_target_leito=edit_target
            ),
            messages=api_messages
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                await response_msg.stream_token(text)

        await response_msg.update()

        # Store response in correct location
        extracted = extract_summary_from_response(full_response)
        if extracted:
            edit_target = cl.user_session.get("edit_target_leito")
            if edit_target:
                confirmed = cl.user_session.get("confirmed_leitos", {})
                confirmed[edit_target] = extracted
                cl.user_session.set("confirmed_leitos", confirmed)
                cl.user_session.set("edit_target_leito", None)
            else:
                cl.user_session.set("current_summary", extracted)

        # Attach PDF button if any summaries exist in the session
        has_summaries = (
            bool(cl.user_session.get("confirmed_leitos"))
            or bool(cl.user_session.get("current_summary"))
        )
        if has_summaries:
            response_msg.actions = [
                cl.Action(name="gerar_pdf", payload={"action": "gerar_pdf"}, label="📄 Gerar PDF da Sessão")
            ]
            await response_msg.update()

        # Add assistant response to history and cap to prevent context overflow
        history.append({"role": "assistant", "content": full_response})
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]
        cl.user_session.set("message_history", history)
        
        # Save assistant message to database
        db = get_db_session()
        if db:
            try:
                from database import MessageRepository, ThreadRepository
                msg_repo = MessageRepository(db)
                msg_repo.add_message(
                    thread_id=thread_id,
                    role="assistant",
                    content=full_response
                )
                
                # Try to extract and update thread title
                await update_thread_title(db, thread_id, full_response)
            except Exception as e:
                logger.error(f"Failed to save assistant message: {e}")
            finally:
                db.close()
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        await response_msg.update()
        await cl.Message(content=f"❌ Erro: {str(e)}").send()


async def update_thread_title(db, thread_id: str, response: str):
    """Try to extract Leito and patient name from response to update thread title."""
    import re

    # Look for patterns like "Leito 1 - Maria" or "**Leito 1 - Maria**"
    pattern = r"[*]*Leito\s+(\d+|[A-Za-z]+)\s*[-–]\s*([A-Za-zÀ-ú\s]+)[*]*"
    match = re.search(pattern, response, re.IGNORECASE)

    if match:
        leito = match.group(1).strip()
        patient_name = match.group(2).strip()

        # Clean up patient name
        patient_name = re.sub(r'[*\n].*', '', patient_name).strip()

        if leito and patient_name and len(patient_name) > 1:
            title = f"Leito {leito} - {patient_name}"

            # Update custom table
            try:
                from database import ThreadRepository
                thread_repo = ThreadRepository(db)
                thread_repo.update_thread_title(thread_id, leito, patient_name)
                logger.info(f"Updated thread title: {title}")
            except Exception as e:
                logger.error(f"Failed to update thread title: {e}")

            # Update Chainlit data layer thread name (for sidebar display)
            try:
                import chainlit.data as cl_data
                if cl_data._data_layer:
                    await cl_data._data_layer.update_thread(
                        thread_id=thread_id, name=title
                    )
            except Exception as e:
                logger.error(f"Failed to update data layer thread name: {e}")


# =============================================================================
# CHAT RESUME (for session history)
# =============================================================================

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Resume a previous chat session from sidebar."""
    try:
        if isinstance(thread, dict):
            thread_id = thread.get("id")
            steps = thread.get("steps") or []
        else:
            thread_id = getattr(thread, "id", None)
            steps = getattr(thread, "steps", []) or []

        if not thread_id:
            return

        cl.user_session.set("thread_id", thread_id)
        cl.user_session.set("thread_persisted", True)
        cl.user_session.set("current_summary", None)
        cl.user_session.set("confirmed_leitos", {})
        cl.user_session.set("edit_target_leito", None)

        # Reconstruct message history — prefer custom messages table (has correct
        # user/assistant alternation for both text and audio inputs), fall back
        # to data layer steps if no custom messages exist.
        history: list[dict] = []
        db = get_db_session()
        if db:
            try:
                from database import MessageRepository
                msg_repo = MessageRepository(db)
                messages = msg_repo.get_thread_messages(thread_id)
                history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ]
            except Exception as e:
                logger.error(f"Failed to load messages from DB: {e}")
            finally:
                db.close()

        # Fall back to data layer steps if custom table had nothing
        if not history:
            for step in steps:
                if not isinstance(step, dict):
                    continue
                step_type = step.get("type", "")
                if step_type == "user_message":
                    content = step.get("output") or step.get("input") or ""
                    if content:
                        history.append({"role": "user", "content": content})
                elif step_type in ("assistant_message", "run"):
                    content = step.get("output", "")
                    if content:
                        history.append({"role": "assistant", "content": content})

        cl.user_session.set("message_history", history)

        # Re-populate session file store for custom elements (TranscriptionAccordion).
        # Chainlit serves element content via /project/file/{chainlitKey}?session_id=...
        # which looks up session.files — a session-scoped in-memory dict cleared on resume.
        # We restore each entry so the frontend can fetch the props JSON on the new session.
        elements = thread.get("elements", []) if isinstance(thread, dict) else []
        if elements:
            from chainlit.context import context
            session = context.session
            session.files_dir.mkdir(exist_ok=True)
            for elem in elements:
                chainlit_key = elem.get("chainlitKey")
                props = elem.get("props")
                if not chainlit_key or not props:
                    continue
                mime = elem.get("mime") or "application/json"
                content_bytes = json.dumps(props).encode("utf-8")
                ext = mimetypes.guess_extension(mime) or ""
                file_path = session.files_dir / f"{chainlit_key}{ext}"
                with open(file_path, "wb") as fh:
                    fh.write(content_bytes)
                session.files[chainlit_key] = {
                    "id": chainlit_key,
                    "path": file_path,
                    "name": elem.get("name", "element"),
                    "type": mime,
                    "size": len(content_bytes),
                }

        # Restore confirmed_leitos and current_summary from all assistant messages
        confirmed_leitos = {}
        last_summary = None
        for msg in history:
            if msg.get("role") == "assistant":
                extracted = extract_summary_from_response(msg.get("content", ""))
                if extracted:
                    num = extract_leito_number(extracted)
                    if num:
                        confirmed_leitos[num] = extracted
                    last_summary = extracted

        # Last summary stays editable; remove it from confirmed
        if last_summary:
            last_num = extract_leito_number(last_summary)
            if last_num and last_num in confirmed_leitos:
                del confirmed_leitos[last_num]
            cl.user_session.set("current_summary", last_summary)

        cl.user_session.set("confirmed_leitos", confirmed_leitos)
    except Exception as e:
        logger.error(f"Failed to resume chat: {e}")
        await cl.Message(content="❌ Não foi possível carregar esta conversa.").send()
