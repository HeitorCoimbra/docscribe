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
import json
from datetime import datetime
from typing import Optional

import chainlit as cl
from chainlit.types import ThreadDict
from anthropic import Anthropic

from core import (
    SumarioPaciente,
    SYSTEM_PROMPT,
    HUMAN_PROMPT_TEMPLATE,
    transcribe_audio,
    CLAUDE_MODEL
)
from database import (
    init_db,
    SessionLocal,
    ThreadRepository,
    MessageRepository,
    UserRepository
)

# Initialize database on startup
init_db()

# =============================================================================
# CONFIGURATION
# =============================================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Anthropic client
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

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

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict,
    default_user: cl.User,
) -> Optional[cl.User]:
    """
    Handle OAuth callback and create/update user in database.
    """
    db = SessionLocal()
    try:
        user_repo = UserRepository(db)
        
        # Extract user info based on provider
        email = raw_user_data.get("email")
        name = raw_user_data.get("name") or raw_user_data.get("login")
        avatar = raw_user_data.get("picture") or raw_user_data.get("avatar_url")
        
        if not email:
            return None
        
        # Create or update user in database
        db_user = user_repo.get_or_create_user(
            email=email,
            name=name,
            avatar_url=avatar,
            provider=provider_id
        )
        
        # Return Chainlit user with database ID
        return cl.User(
            identifier=db_user.id,
            metadata={
                "email": email,
                "name": name,
                "avatar": avatar,
                "provider": provider_id
            }
        )
    finally:
        db.close()


# =============================================================================
# SESSION/THREAD MANAGEMENT
# =============================================================================

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Resume a previous chat session."""
    thread_id = thread.get("id")
    
    if not thread_id:
        return
    
    db = SessionLocal()
    try:
        msg_repo = MessageRepository(db)
        messages = msg_repo.get_thread_messages(thread_id)
        
        # Restore message history
        cl.user_session.set("thread_id", thread_id)
        
        history = []
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        cl.user_session.set("message_history", history)
        
    finally:
        db.close()


@cl.set_chat_profiles
async def set_chat_profiles():
    """Define chat profiles (optional)."""
    return [
        cl.ChatProfile(
            name="DocScribe",
            markdown_description="Sumário de Pacientes de UTI",
            icon="https://api.iconify.design/mdi:hospital-building.svg"
        )
    ]


# =============================================================================
# CHAT START
# =============================================================================

@cl.on_chat_start
async def on_chat_start():
    """Initialize new chat session."""
    user = cl.user_session.get("user")
    
    if not user:
        await cl.Message(content="❌ Erro de autenticação. Por favor, faça login.").send()
        return
    
    # Create new thread in database
    db = SessionLocal()
    try:
        thread_repo = ThreadRepository(db)
        thread = thread_repo.create_thread(
            user_id=user.identifier,
            title="Nova Sessão"
        )
        
        cl.user_session.set("thread_id", thread.id)
        cl.user_session.set("message_history", [])
        cl.user_session.set("current_summary", None)
        
    finally:
        db.close()
    
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
    user = cl.user_session.get("user")
    thread_id = cl.user_session.get("thread_id")
    
    if not user or not thread_id:
        await cl.Message(content="❌ Sessão inválida. Por favor, recarregue a página.").send()
        return
    
    # Get message history
    history = cl.user_session.get("message_history", [])
    
    # Check for audio file attachments
    audio_files = [f for f in (message.elements or []) if f.mime and f.mime.startswith("audio/")]
    
    user_content = message.content
    transcription = None
    
    # Process audio if present
    if audio_files:
        audio_file = audio_files[0]
        
        processing_msg = cl.Message(content="🎤 Transcrevendo áudio...")
        await processing_msg.send()
        
        try:
            # Read audio bytes
            audio_bytes = audio_file.content if hasattr(audio_file, 'content') else open(audio_file.path, 'rb').read()
            
            # Transcribe with Groq Whisper
            transcription = transcribe_audio(
                audio_bytes=audio_bytes,
                filename=audio_file.name or "audio.mp3",
                groq_api_key=GROQ_API_KEY
            )
            
            user_content = f"[Transcrição do áudio]\n\n{transcription}"
            
            await processing_msg.remove()
            await cl.Message(content=f"✅ **Áudio transcrito:**\n\n_{transcription[:500]}{'...' if len(transcription) > 500 else ''}_").send()
            
        except Exception as e:
            await processing_msg.remove()
            await cl.Message(content=f"❌ Erro na transcrição: {str(e)}").send()
            return
    
    # Add user message to history
    history.append({"role": "user", "content": user_content})
    
    # Save message to database
    db = SessionLocal()
    try:
        msg_repo = MessageRepository(db)
        msg_repo.add_message(
            thread_id=thread_id,
            role="user",
            content=user_content,
            has_audio=bool(audio_files),
            transcription=transcription
        )
    finally:
        db.close()
    
    # Generate response with Claude
    response_msg = cl.Message(content="")
    await response_msg.send()
    
    try:
        # Build API messages
        api_messages = [{"role": m["role"], "content": m["content"]} for m in history]
        
        # Stream response
        full_response = ""
        with anthropic_client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=CHAT_SYSTEM,
            messages=api_messages
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                await response_msg.stream_token(text)
        
        await response_msg.update()
        
        # Add assistant response to history
        history.append({"role": "assistant", "content": full_response})
        cl.user_session.set("message_history", history)
        
        # Save assistant message to database
        db = SessionLocal()
        try:
            msg_repo = MessageRepository(db)
            msg_repo.add_message(
                thread_id=thread_id,
                role="assistant",
                content=full_response
            )
            
            # Try to extract summary info for thread title
            await update_thread_from_response(thread_id, full_response)
            
        finally:
            db.close()
        
    except Exception as e:
        await response_msg.update()
        await cl.Message(content=f"❌ Erro: {str(e)}").send()


async def update_thread_from_response(thread_id: str, response: str):
    """
    Try to extract Leito and patient name from response to update thread title.
    """
    import re
    
    # Look for patterns like "Leito 1 - Maria" or "**Leito 1 - Maria**"
    pattern = r"[*]*Leito\s+(\d+|[A-Za-z]+)\s*[-–]\s*([A-Za-zÀ-ú\s]+)[*]*"
    match = re.search(pattern, response, re.IGNORECASE)
    
    if match:
        leito = match.group(1).strip()
        patient_name = match.group(2).strip()
        
        # Clean up patient name (remove trailing punctuation, etc.)
        patient_name = re.sub(r'[*\n].*', '', patient_name).strip()
        
        if leito and patient_name and len(patient_name) > 1:
            db = SessionLocal()
            try:
                thread_repo = ThreadRepository(db)
                thread_repo.update_thread_title(thread_id, leito, patient_name)
            finally:
                db.close()


# =============================================================================
# SESSION HISTORY SIDEBAR
# =============================================================================

@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings updates."""
    pass


# =============================================================================
# DATA LAYER FOR THREAD HISTORY
# =============================================================================

@cl.data_layer
class PostgresDataLayer:
    """Custom data layer for PostgreSQL session persistence."""
    
    async def get_user_threads(self, user_id: str, pagination=None):
        """Get threads for sidebar display, grouped by day."""
        db = SessionLocal()
        try:
            thread_repo = ThreadRepository(db)
            threads = thread_repo.get_user_threads(user_id)
            
            return [
                {
                    "id": t.id,
                    "name": t.title or "Nova Sessão",
                    "createdAt": t.created_at.isoformat(),
                    "metadata": {
                        "leito": t.leito,
                        "patient_name": t.patient_name,
                        "is_complete": t.is_complete
                    }
                }
                for t in threads
            ]
        finally:
            db.close()
    
    async def get_thread(self, thread_id: str):
        """Get a specific thread."""
        db = SessionLocal()
        try:
            thread_repo = ThreadRepository(db)
            thread = thread_repo.get_thread(thread_id)
            
            if not thread:
                return None
            
            return {
                "id": thread.id,
                "name": thread.title,
                "createdAt": thread.created_at.isoformat(),
                "metadata": {
                    "leito": thread.leito,
                    "patient_name": thread.patient_name,
                    "summary": thread.summary_json
                }
            }
        finally:
            db.close()
    
    async def delete_thread(self, thread_id: str):
        """Delete a thread."""
        db = SessionLocal()
        try:
            thread_repo = ThreadRepository(db)
            thread_repo.delete_thread(thread_id)
        finally:
            db.close()


# Initialize data layer
# Note: This requires Chainlit data layer feature to be enabled
