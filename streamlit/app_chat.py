"""
DocScribe Chat - Conversational interface for medical audio summarization.

Features:
- Chat-based interaction with Claude
- Audio file upload with Groq Whisper transcription
- Guided conversation to fill SumarioPaciente schema
- Streaming responses
"""

import streamlit as st
from dotenv import load_dotenv
import os
import json
from anthropic import Anthropic

from core import (
    SumarioPaciente, 
    SYSTEM_PROMPT, 
    transcribe_audio,
    WHISPER_MODEL,
    CLAUDE_MODEL
)

# Load environment variables
load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="DocScribe Chat",
    page_icon="💬",
    layout="centered"
)

# =============================================================================
# CHAT SYSTEM PROMPT
# =============================================================================

CHAT_SYSTEM_PROMPT = """Você é um assistente médico especializado em extrair sumários de pacientes de UTI.

Seu objetivo é ajudar o usuário a preencher um sumário estruturado com os seguintes campos:
- **Leito**: Número do leito
- **Nome do Paciente**: Nome completo
- **Diagnósticos**: Lista de problemas médicos atuais
- **Pendências**: Tarefas/avaliações aguardando resolução
- **Condutas**: Ações tomadas ou planejadas (sempre começar com verbo no infinitivo)

REGRAS IMPORTANTES:
1. NUNCA invente informações - use apenas o que foi dito
2. Seja conciso e objetivo
3. Condutas SEMPRE começam com verbo no INFINITIVO (Manter, Iniciar, Solicitar, etc.)
4. Use terminologia médica correta (IRA, não "disfunção renal"; norepinefrina, não "noraepinefrina")

Quando receber uma transcrição de áudio, analise e extraia as informações.
Se algo não estiver claro, pergunte ao usuário.
Quando tiver todas as informações necessárias, apresente o sumário formatado.

Para finalizar, quando o usuário confirmar que o sumário está correto, responda com o JSON estruturado entre tags <sumario_json> e </sumario_json>.

Exemplo:
<sumario_json>
{"leito": "1", "nome_paciente": "Maria", "diagnosticos": ["..."], "pendencias": ["..."], "condutas": ["Manter...", "Iniciar..."]}
</sumario_json>
"""

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "transcription" not in st.session_state:
    st.session_state.transcription = None

if "sumario_final" not in st.session_state:
    st.session_state.sumario_final = None

# =============================================================================
# API KEYS
# =============================================================================

# Try Streamlit secrets first, then environment variables
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY", None)
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
except:
    groq_api_key = None
    anthropic_api_key = None

if not groq_api_key:
    groq_api_key = os.environ.get("GROQ_API_KEY", None)
if not anthropic_api_key:
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", None)

# Check for missing keys
missing_keys = []
if not groq_api_key:
    missing_keys.append("GROQ_API_KEY")
if not anthropic_api_key:
    missing_keys.append("ANTHROPIC_API_KEY")

if missing_keys:
    st.error(f"❌ API Keys não configuradas: {', '.join(missing_keys)}")
    st.info("Configure no arquivo `.env` ou em Streamlit Cloud Secrets.")
    st.stop()

# Initialize Anthropic client
client = Anthropic(api_key=anthropic_api_key)

# =============================================================================
# HEADER
# =============================================================================

st.title("💬 DocScribe Chat")
st.caption("Converse comigo para criar o sumário do paciente")

# =============================================================================
# SIDEBAR - FILE UPLOAD
# =============================================================================

with st.sidebar:
    st.header("📁 Upload de Áudio")
    
    uploaded_file = st.file_uploader(
        "Arraste um arquivo de áudio",
        type=["mp3", "wav", "m4a", "opus", "ogg", "webm", "flac"],
        help="O áudio será transcrito automaticamente"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("🎤 Transcrever Áudio", use_container_width=True):
            with st.spinner("Transcrevendo com Whisper..."):
                try:
                    audio_bytes = uploaded_file.read()
                    transcription = transcribe_audio(
                        audio_bytes=audio_bytes,
                        filename=uploaded_file.name,
                        groq_api_key=groq_api_key
                    )
                    st.session_state.transcription = transcription
                    
                    # Add transcription as user message
                    user_msg = f"Aqui está a transcrição do áudio:\n\n{transcription}"
                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_msg
                    })
                    st.success("✅ Transcrito!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")
    
    st.divider()
    
    # Show current transcription
    if st.session_state.transcription:
        with st.expander("📝 Transcrição atual"):
            st.text(st.session_state.transcription)
    
    # Show final summary if available
    if st.session_state.sumario_final:
        st.divider()
        st.header("📋 Sumário Final")
        sumario = st.session_state.sumario_final
        st.markdown(f"**Leito {sumario.leito}** - {sumario.nome_paciente}")
        st.text_area("Copiar:", value=sumario.formatar(), height=200)
        
        with st.expander("Ver JSON"):
            st.json(sumario.model_dump())
    
    st.divider()
    
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.session_state.transcription = None
        st.session_state.sumario_final = None
        st.rerun()

# =============================================================================
# CHAT DISPLAY
# =============================================================================

# Display chat messages
for message in st.session_state.messages:
    avatar = "🧑‍⚕️" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# =============================================================================
# CHAT INPUT
# =============================================================================

if prompt := st.chat_input("Digite sua mensagem ou cole uma transcrição..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Build messages for API
        api_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ]
        
        # Stream response
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=CHAT_SYSTEM_PROMPT,
            messages=api_messages
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Check if response contains final JSON
    if "<sumario_json>" in full_response and "</sumario_json>" in full_response:
        try:
            json_start = full_response.index("<sumario_json>") + len("<sumario_json>")
            json_end = full_response.index("</sumario_json>")
            json_str = full_response[json_start:json_end].strip()
            
            data = json.loads(json_str)
            sumario = SumarioPaciente(**data)
            st.session_state.sumario_final = sumario
            
            st.success("✅ Sumário extraído! Veja na barra lateral.")
            st.rerun()
        except Exception as e:
            st.warning(f"Não foi possível extrair o sumário: {e}")

# =============================================================================
# WELCOME MESSAGE
# =============================================================================

if not st.session_state.messages:
    st.info("""
    👋 **Bem-vindo ao DocScribe Chat!**
    
    **Como usar:**
    1. **Upload de áudio**: Use a barra lateral para fazer upload e transcrever um áudio
    2. **Colar transcrição**: Ou cole diretamente uma transcrição no chat
    3. **Conversar**: Tire dúvidas e refine o sumário comigo
    4. **Confirmar**: Quando o sumário estiver correto, confirme para extrair o JSON final
    
    *Dica: Você pode corrigir informações ou pedir esclarecimentos a qualquer momento!*
    """)

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption("DocScribe Chat | 🎤 Groq Whisper + 🧠 Claude")
