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
from datetime import datetime
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
    CLAUDE_MODEL
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


def build_system_prompt(current_summary: str | None = None) -> str:
    """Build the system prompt with optional current summary context."""
    summary_context = ""
    if current_summary:
        summary_context = f"""
=== SUMARIO ATUAL DO PACIENTE (use como base para atualizacoes) ===
{current_summary}
=== FIM DO SUMARIO ATUAL ==="""

    return f"""{SYSTEM_PROMPT}

=== INSTRUCOES DE COMPORTAMENTO ===

REGRA PRINCIPAL: Quando receber uma transcricao de audio ou texto de passagem de plantao,
voce DEVE SEMPRE responder IMEDIATAMENTE com o sumario estruturado abaixo.
NAO faca perguntas antes de mostrar o sumario. NAO peca confirmacao antes de mostrar o sumario.

Para campos que NAO podem ser preenchidos com base na transcricao, use exatamente:
🔴 PENDENTE

FORMATO OBRIGATORIO DE RESPOSTA:

**Leito [X] - [Nome do Paciente]**

**Diagnosticos:**
1. [diagnostico extraido ou 🔴 PENDENTE]

**Pendencias:**
1. [pendencia extraida ou 🔴 PENDENTE]

**Condutas:**
- [conduta com verbo no infinitivo ou 🔴 PENDENTE]

---
Deseja corrigir ou adicionar algo?

=== REGRAS DE ATUALIZACAO ===
- Quando o usuario enviar correcoes, mostre o sumario COMPLETO atualizado (nao apenas o campo alterado)
- Condutas SEMPRE comecam com verbo no INFINITIVO
- Seja conciso. Nao repita instrucoes ao usuario, apenas mostre o sumario atualizado.

{summary_context}"""


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

        user_id = email  # Default to email as ID

        # Try to save to database if available
        db = get_db_session()
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
                user_id = db_user.id
            except Exception as e:
                logger.error(f"Database error in oauth_callback: {e}")
            finally:
                db.close()

        return cl.User(
            identifier=user_id,
            metadata={
                "email": email,
                "name": name,
                "avatar": avatar,
                "provider": provider_id
            }
        )
else:
    logger.warning("No OAuth provider configured — running without authentication")


# =============================================================================
# CHAT START
# =============================================================================

@cl.on_chat_start
async def on_chat_start():
    """Initialize new chat session."""
    user = cl.user_session.get("user")

    # Get thread ID from Chainlit context (set by data layer) or generate one
    thread_id = None
    try:
        thread_id = cl.context.session.thread_id
    except (AttributeError, RuntimeError):
        pass
    if not thread_id:
        thread_id = str(uuid.uuid4())

    # Try to create thread in custom database table for domain-specific data
    if user:
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

    # Ensure thread is registered in data layer with user info (for sidebar)
    if user:
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

    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("message_history", [])
    cl.user_session.set("current_summary", None)

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
                "characterCount": len(transcription)
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


# =============================================================================
# AUDIO RECORDING HANDLERS
# =============================================================================

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    """Collect audio chunks during recording."""
    if chunk.isStart:
        # Initialize audio buffer for new recording
        cl.user_session.set("audio_chunks", [])
        cl.user_session.set("audio_mime", chunk.mime)
    
    # Collect chunk data
    chunks = cl.user_session.get("audio_chunks", [])
    chunks.append(chunk.chunk)
    cl.user_session.set("audio_chunks", chunks)


@cl.on_audio_end
async def on_audio_end():
    """Process complete audio recording and transcribe it."""
    chunks = cl.user_session.get("audio_chunks", [])
    mime = cl.user_session.get("audio_mime", "audio/webm")
    
    if not chunks:
        await cl.Message(content="❌ Nenhum áudio foi gravado.").send()
        return
    
    # Combine all chunks into single audio bytes
    audio_bytes = b"".join(chunks)
    
    # Clear chunks from session
    cl.user_session.set("audio_chunks", None)
    cl.user_session.set("audio_mime", None)
    
    # Determine file extension from MIME type
    mime_to_ext = {
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/mp3": ".mp3",
        "audio/m4a": ".m4a",
    }
    ext = mime_to_ext.get(mime, ".webm")
    filename = f"recorded_audio{ext}"
    
    # Process the recorded audio
    transcription = await process_audio_and_transcribe(
        audio_bytes=audio_bytes,
        filename=filename,
        source_label="áudio gravado"
    )
    
    if transcription:
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
    transcription = None
    
    # Process audio if present
    if audio_files:
        audio_file = audio_files[0]
        
        # Read audio bytes
        if hasattr(audio_file, 'content') and audio_file.content:
            audio_bytes = audio_file.content
        elif hasattr(audio_file, 'path') and audio_file.path:
            with open(audio_file.path, 'rb') as f:
                audio_bytes = f.read()
        else:
            await cl.Message(content="❌ Não foi possível ler o arquivo de áudio.").send()
            return
        
        # Process and transcribe audio
        transcription = await process_audio_and_transcribe(
            audio_bytes=audio_bytes,
            filename=audio_file.name or "audio.mp3",
            source_label="áudio"
        )
        
        if transcription:
            user_content = f"[Transcrição do áudio]\n\n{transcription}"
        else:
            return
    
    if not user_content.strip():
        return
    
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
                transcription=transcription
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
    
    # Generate response with Claude
    response_msg = cl.Message(content="")
    await response_msg.send()
    
    try:
        # Build API messages
        api_messages = [{"role": m["role"], "content": m["content"]} for m in history]
        
        # Stream response
        full_response = ""
        current_summary = cl.user_session.get("current_summary")
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=build_system_prompt(current_summary),
            messages=api_messages
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                await response_msg.stream_token(text)
        
        await response_msg.update()
        
        # Update current summary state if a summary was found
        extracted = extract_summary_from_response(full_response)
        if extracted:
            cl.user_session.set("current_summary", extracted)

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
    thread_id = thread.get("id")

    if not thread_id:
        return

    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("current_summary", None)

    # Reconstruct message history from data layer steps
    history: list[dict] = []
    for step in thread.get("steps", []):
        step_type = step.get("type", "")
        if step_type == "user_message":
            content = step.get("output") or step.get("input") or ""
            if content:
                history.append({"role": "user", "content": content})
        elif step_type in ("assistant_message", "run"):
            content = step.get("output", "")
            if content:
                history.append({"role": "assistant", "content": content})

    # Fall back to custom messages table if no steps found
    if not history:
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

    cl.user_session.set("message_history", history)

    # Restore summary state from last assistant message
    for msg in reversed(history):
        if msg["role"] == "assistant":
            extracted = extract_summary_from_response(msg["content"])
            if extracted:
                cl.user_session.set("current_summary", extracted)
                break
