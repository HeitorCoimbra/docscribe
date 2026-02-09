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

# Ensure the app directory is in sys.path so lazy imports (database, core)
# work correctly when Chainlit loads the module via importlib.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chainlit as cl
from chainlit.types import ThreadDict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
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

CHAT_SYSTEM = f"""{SYSTEM_PROMPT}

INSTRUÇÕES PARA CONVERSA:
1. Quando receber uma transcrição de áudio, analise e apresente o sumário estruturado
2. Se algo não estiver claro, pergunte ao usuário para esclarecer
3. Permita correções e ajustes através da conversa
4. Condutas SEMPRE começam com verbo no INFINITIVO

Quando tiver o sumário completo e confirmado, apresente no formato:

**Leito [X] - [Nome do Paciente]**

**Diagnósticos:**
1. [diagnóstico]

**Pendências:**
1. [pendência]

**Condutas:**
• [conduta com verbo no infinitivo]

E pergunte se o usuário deseja salvar o sumário.
"""


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
    
    # Generate thread ID
    thread_id = str(uuid.uuid4())
    
    # Try to create thread in database
    if user:
        db = get_db_session()
        if db:
            try:
                from database import ThreadRepository
                thread_repo = ThreadRepository(db)
                thread = thread_repo.create_thread(
                    user_id=user.identifier,
                    title="Nova Sessão"
                )
                thread_id = thread.id
            except Exception as e:
                logger.error(f"Failed to create thread in DB: {e}")
            finally:
                db.close()
    
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("message_history", [])
    cl.user_session.set("current_summary", None)
    
    # Welcome message
    await cl.Message(
        content="""👋 **Bem-vindo ao DocScribe!**

Sou seu assistente para criar sumários de pacientes de UTI.

**Como usar:**
1. 📎 Faça upload de um áudio de passagem de plantão
2. 💬 Ou cole a transcrição diretamente no chat
3. ✏️ Corrija ou adicione informações conforme necessário
4. ✅ Confirme para salvar o sumário

*Dica: Você pode enviar múltiplos áudios e eu consolidarei as informações.*
"""
    ).send()


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
        
        processing_msg = cl.Message(content="🎤 Transcrevendo áudio...")
        await processing_msg.send()
        
        try:
            # Read audio bytes
            if hasattr(audio_file, 'content') and audio_file.content:
                audio_bytes = audio_file.content
            elif hasattr(audio_file, 'path') and audio_file.path:
                with open(audio_file.path, 'rb') as f:
                    audio_bytes = f.read()
            else:
                raise ValueError("Não foi possível ler o arquivo de áudio")
            
            # Transcribe with Groq Whisper
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY não configurada")
            
            transcription = transcribe_audio(
                audio_bytes=audio_bytes,
                filename=audio_file.name or "audio.mp3",
                groq_api_key=GROQ_API_KEY
            )
            
            user_content = f"[Transcrição do áudio]\n\n{transcription}"
            
            await processing_msg.remove()
            preview = transcription[:500] + "..." if len(transcription) > 500 else transcription
            await cl.Message(content=f"✅ **Áudio transcrito:**\n\n_{preview}_").send()
            
        except Exception as e:
            await processing_msg.remove()
            await cl.Message(content=f"❌ Erro na transcrição: {str(e)}").send()
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
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=CHAT_SYSTEM,
            messages=api_messages
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                await response_msg.stream_token(text)
        
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
            try:
                from database import ThreadRepository
                thread_repo = ThreadRepository(db)
                thread_repo.update_thread_title(thread_id, leito, patient_name)
                logger.info(f"Updated thread title: Leito {leito} - {patient_name}")
            except Exception as e:
                logger.error(f"Failed to update thread title: {e}")


# =============================================================================
# CHAT RESUME (for session history)
# =============================================================================

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Resume a previous chat session."""
    thread_id = thread.get("id")
    
    if not thread_id:
        return
    
    db = get_db_session()
    if db:
        try:
            from database import MessageRepository
            msg_repo = MessageRepository(db)
            messages = msg_repo.get_thread_messages(thread_id)
            
            cl.user_session.set("thread_id", thread_id)
            
            history = [{"role": msg.role, "content": msg.content} for msg in messages]
            cl.user_session.set("message_history", history)
            
        except Exception as e:
            logger.error(f"Failed to resume chat: {e}")
        finally:
            db.close()
