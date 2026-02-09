"""
DocScribe - Streamlit App for Medical Audio Summarization

Architecture:
1. Groq Whisper - Fast audio transcription  
2. Anthropic Claude - Text analysis and structured extraction
"""

import streamlit as st
from dotenv import load_dotenv
import os

from core import process_audio, SumarioPaciente

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="DocScribe - Sumário de Pacientes",
    page_icon="🏥",
    layout="centered"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .summary-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 15px;
    }
    .transcription-box {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================

st.title("🏥 DocScribe")
st.subheader("Sumário de Pacientes de UTI")

st.markdown("""
*Upload de áudio de passagem de plantão → Transcrição (Groq Whisper) → Sumário estruturado (Claude)*
""")

st.divider()

# =============================================================================
# API KEYS - Check secrets (Streamlit Cloud) or .env
# =============================================================================

# Try Streamlit secrets first (for Streamlit Cloud deployment)
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY", None)
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
except:
    groq_api_key = None
    anthropic_api_key = None

# Fall back to environment variables (from .env file)
if not groq_api_key:
    groq_api_key = os.environ.get("GROQ_API_KEY", None)
if not anthropic_api_key:
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", None)

# Show warning if keys are missing
missing_keys = []
if not groq_api_key:
    missing_keys.append("GROQ_API_KEY")
if not anthropic_api_key:
    missing_keys.append("ANTHROPIC_API_KEY")

if missing_keys:
    st.warning(f"⚠️ **API Keys não configuradas: {', '.join(missing_keys)}**")
    
    st.markdown("""
    Configure as chaves no arquivo `.env`:
    
    ```
    GROQ_API_KEY=sua-chave-groq
    ANTHROPIC_API_KEY=sua-chave-anthropic
    ```
    
    Ou, para deploy no Streamlit Cloud, configure em **Settings > Secrets**.
    """)
    
    st.markdown("---")
    st.markdown("**Ou cole as chaves aqui (apenas para teste):**")
    
    col1, col2 = st.columns(2)
    with col1:
        if not groq_api_key:
            groq_api_key = st.text_input("Groq API Key:", type="password", key="groq_input")
    with col2:
        if not anthropic_api_key:
            anthropic_api_key = st.text_input("Anthropic API Key:", type="password", key="anthropic_input")
    
    if not groq_api_key or not anthropic_api_key:
        st.stop()

# =============================================================================
# FILE UPLOAD
# =============================================================================

st.markdown("### 📁 Upload do Áudio")

uploaded_file = st.file_uploader(
    "Arraste um arquivo de áudio ou clique para selecionar",
    type=["mp3", "wav", "m4a", "opus", "ogg", "webm", "flac"],
    help="Formatos suportados: MP3, WAV, M4A, OPUS, OGG, WebM, FLAC"
)

# =============================================================================
# AUDIO PREVIEW
# =============================================================================

if uploaded_file is not None:
    st.markdown("### 🎧 Prévia do Áudio")
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    # Reset file pointer after preview
    uploaded_file.seek(0)

# =============================================================================
# PROCESS BUTTON
# =============================================================================

process_button = st.button(
    "🎯 Gerar Sumário",
    disabled=uploaded_file is None,
    use_container_width=True
)

# =============================================================================
# PROCESSING
# =============================================================================

if process_button and uploaded_file is not None:
    
    # Step 1: Transcription with Groq Whisper
    with st.spinner("🎤 Transcrevendo áudio com Whisper (Groq)..."):
        try:
            audio_bytes = uploaded_file.read()
            
            from core import transcribe_audio
            transcription = transcribe_audio(
                audio_bytes=audio_bytes,
                filename=uploaded_file.name,
                groq_api_key=groq_api_key
            )
            st.session_state["transcription"] = transcription
        except Exception as e:
            st.error(f"❌ Erro na transcrição: {str(e)}")
            st.stop()
    
    # Step 2: Analysis with Claude
    with st.spinner("🧠 Analisando transcrição com Claude..."):
        try:
            from core import analyze_transcription
            sumario = analyze_transcription(
                transcription=transcription,
                anthropic_api_key=anthropic_api_key
            )
            st.session_state["sumario"] = sumario
            st.session_state["sumario_text"] = sumario.formatar()
        except Exception as e:
            st.error(f"❌ Erro na análise: {str(e)}")
            st.stop()
    
    st.success("✅ Processamento concluído!")

# =============================================================================
# RESULTS DISPLAY
# =============================================================================

if "transcription" in st.session_state:
    with st.expander("📝 Ver transcrição do áudio", expanded=False):
        st.markdown(f'<div class="transcription-box">{st.session_state["transcription"]}</div>', 
                    unsafe_allow_html=True)

if "sumario" in st.session_state:
    st.markdown("### 📋 Sumário Gerado")
    
    sumario = st.session_state["sumario"]
    sumario_text = st.session_state["sumario_text"]
    
    # Display structured summary
    st.markdown(f"**Leito {sumario.leito}** - {sumario.nome_paciente}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🩺 Diagnósticos:**")
        for i, diag in enumerate(sumario.diagnosticos, 1):
            st.markdown(f"{i}. {diag}")
    
    with col2:
        st.markdown("**⏳ Pendências:**")
        for i, pend in enumerate(sumario.pendencias, 1):
            st.markdown(f"{i}. {pend}")
    
    st.markdown("**📌 Condutas:**")
    for conduta in sumario.condutas:
        st.markdown(f"• {conduta}")
    
    st.divider()
    
    # Copyable text area
    st.markdown("**📋 Texto para copiar:**")
    st.text_area(
        label="Sumário formatado",
        value=sumario_text,
        height=300,
        label_visibility="collapsed"
    )
    
    # JSON view
    with st.expander("🔧 Ver dados em JSON"):
        st.json(sumario.model_dump())

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "DocScribe Streamlit App | "
    "🎤 Groq Whisper + 🧠 Claude"
    "</div>",
    unsafe_allow_html=True
)
