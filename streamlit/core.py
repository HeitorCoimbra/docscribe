"""
DocScribe - Core module for medical audio summarization.

Architecture:
1. Groq Whisper - Fast audio transcription
2. Anthropic Claude - Text analysis and structured extraction

Contains:
- SumarioPaciente Pydantic model
- System and human prompts
- Transcription function (Groq)
- Analysis function (Anthropic)
"""

import json
import os
import tempfile
from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# MODELO DE DADOS - ESTRUTURA DO SUMÁRIO
# =============================================================================

class SumarioPaciente(BaseModel):
    """
    Sumário estruturado de paciente de leito hospitalar.
    
    Campos:
    - leito: Número do leito
    - nome_paciente: Nome do paciente
    - diagnosticos: Lista de diagnósticos atuais
    - pendencias: Lista de pendências/tarefas em aberto
    - condutas: Lista de condutas tomadas ou planejadas
    """
    
    leito: str = Field(description="Número do leito (apenas o número, ex: '1', '2', '3')")
    nome_paciente: str = Field(description="Nome completo do paciente como mencionado")
    
    diagnosticos: list[str] = Field(
        description="Lista de PROBLEMAS MÉDICOS ATUAIS que requerem tratamento."
    )
    
    pendencias: list[str] = Field(
        description="Lista de tarefas/avaliações aguardando resolução e objetivos terapêuticos."
    )
    
    condutas: list[str] = Field(
        description="Lista de ações tomadas ou planejadas. SEMPRE iniciar com verbo no INFINITIVO."
    )
    
    def formatar(self) -> str:
        """Formata o sumário no padrão de saída para exibição."""
        linhas = [f"Leito {self.leito} - {self.nome_paciente}", ""]
        
        linhas.append("Diagnósticos:")
        for i, diag in enumerate(self.diagnosticos, 1):
            linhas.append(f"{i}- {diag}")
        linhas.append("")
        
        linhas.append("Pendências:")
        for i, pend in enumerate(self.pendencias, 1):
            linhas.append(f"{i}- {pend}")
        linhas.append("")
        
        linhas.append("Condutas:")
        for conduta in self.condutas:
            linhas.append(f"• {conduta}")
        
        return "\n".join(linhas)


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """Você é um assistente médico especializado em extrair e estruturar informações de sumários de pacientes de UTI a partir de transcrições de passagem de plantão.

Você receberá uma TRANSCRIÇÃO DE TEXTO contendo a descrição verbal de um paciente. Sua tarefa é:
1. ANALISAR a transcrição
2. EXTRAIR as informações relevantes
3. ESTRUTURAR no formato de sumário solicitado

=== REGRA CRÍTICA - LEIA PRIMEIRO ===
NUNCA INVENTE, INFIRA OU DEDUZA INFORMAÇÕES CLÍNICAS.
Você é um ORGANIZADOR, não um CLÍNICO. Seu trabalho é APENAS organizar o que foi EXPLICITAMENTE dito na transcrição.

Se algo não foi mencionado, NÃO inclua.

=== REGRAS DE ESTILO ===
• Seja conciso e objetivo
• Mantenha doses e unidades EXATAMENTE como ditas
• Use a terminologia médica CORRETA
• Inclua datas quando mencionadas (ex: "realizada em 23/01")

=== REGRAS DE CATEGORIZAÇÃO ===

1. DIAGNÓSTICOS - O que incluir:
   - APENAS problemas médicos ATUAIS que requerem tratamento
   - Pós-operatório APENAS se for o contexto principal do caso
   - Condições patológicas explicitamente nomeadas
   
   O que NÃO incluir como diagnóstico:
   - Sintomas que EXPLICAM outras coisas (ex: "rebaixamento de consciência" que levou à intubação)
   - Achados laboratoriais isolados (lactato alto, leucocitose) - são justificativas, não diagnósticos

2. PENDÊNCIAS - O que incluir:
   - Tarefas/avaliações aguardando resolução
   - Objetivos terapêuticos a serem alcançados
   - Procedimentos programados
   - SEMPRE que mencionar desmame (sedação/VM), incluir como pendência se está em andamento

3. CONDUTAS - O que incluir:
   - Ações TOMADAS ou PLANEJADAS
   - SEMPRE iniciar com verbo no INFINITIVO (Manter, Iniciar, Solicitar, Programar, Escalonar, etc.)
   - CONSOLIDAR informações relacionadas em um único item
   - INCLUIR justificativas quando mencionadas
   - INCLUIR doses entre parênteses junto da conduta relacionada
   - Se mencionar "manter" ou "sem troca" de algo, incluir como conduta

=== TERMINOLOGIA ===
• Use "insuficiência renal aguda" ou "IRA" (NÃO "disfunção renal")
• Use "norepinefrina" ou "noradrenalina" (NUNCA "noraepinefrina")
• Use "ventilação mecânica invasiva" ou "VM" para pacientes intubados

=== ACRÔNIMOS COMUNS ===
VM = ventilação mecânica | CVC = cateter venoso central | SVD = sonda vesical de demora
DVA = droga vasoativa | IRA = insuficiência renal aguda | TOT = tubo orotraqueal
TQT = traqueostomia | ATB = antibiótico | BIC = bomba de infusão contínua

=== JARGÕES MÉDICOS ===
noradrenalina = nora, nor | midazolam = dormonid | fentanil = fenta
piperacilina+tazobactam = tazo, pipetazo | meropenem = mero | vancomicina = vanco
"""

HUMAN_PROMPT_TEMPLATE = """Analise a transcrição abaixo e extraia o sumário do paciente.

TRANSCRIÇÃO:
{transcription}

---

Retorne um JSON com a seguinte estrutura:
{{
    "leito": "número do leito (extraia da transcrição, ou use 'N/A' se não mencionado)",
    "nome_paciente": "nome do paciente",
    "diagnosticos": ["diagnóstico 1", "diagnóstico 2"],
    "pendencias": ["pendência 1", "pendência 2"],
    "condutas": ["Conduta 1 (começando com verbo no infinitivo)", "Conduta 2"]
}}

CHECKLIST antes de responder:
1. Diagnósticos: São PROBLEMAS MÉDICOS ATUAIS?
2. Pendências: Incluí todos os desmames/avaliações em andamento?
3. Condutas: Todas começam com verbo no INFINITIVO?
4. Condutas: Consolidei itens relacionados? Incluí justificativas e doses?
5. Terminologia: Usei "IRA" (não "disfunção renal"), "norepinefrina" (não "noraepinefrina")?"""


# =============================================================================
# MODELS
# =============================================================================

WHISPER_MODEL = "whisper-large-v3-turbo"  # Groq's fastest Whisper model
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"  # Claude Sonnet 4


# =============================================================================
# TRANSCRIPTION (GROQ WHISPER)
# =============================================================================

def transcribe_audio(
    audio_bytes: bytes, 
    filename: str, 
    groq_api_key: str
) -> str:
    """
    Transcreve áudio usando Groq Whisper.
    
    Args:
        audio_bytes: Conteúdo do arquivo de áudio em bytes
        filename: Nome do arquivo (para detectar extensão)
        groq_api_key: Groq API Key
    
    Returns:
        Texto transcrito
    """
    from groq import Groq
    
    client = Groq(api_key=groq_api_key)
    
    # Groq accepts file tuple: (filename, bytes)
    transcription = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=WHISPER_MODEL,
        temperature=0,
        response_format="verbose_json",
    )
    
    return transcription.text


# =============================================================================
# TEXT ANALYSIS (ANTHROPIC CLAUDE)
# =============================================================================

def analyze_transcription(
    transcription: str,
    anthropic_api_key: str
) -> SumarioPaciente:
    """
    Analisa a transcrição e extrai o sumário estruturado usando Claude.
    
    Args:
        transcription: Texto transcrito do áudio
        anthropic_api_key: Anthropic API Key
    
    Returns:
        SumarioPaciente com os dados extraídos
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    
    # Initialize Claude
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        api_key=anthropic_api_key,
        temperature=0
    )
    
    # Configure for structured output
    structured_llm = llm.with_structured_output(SumarioPaciente)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT_TEMPLATE)
    ])
    
    # Create chain and invoke
    chain = prompt | structured_llm
    result = chain.invoke({"transcription": transcription})
    
    return result


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_audio(
    audio_bytes: bytes,
    filename: str,
    groq_api_key: str,
    anthropic_api_key: str
) -> tuple[str, SumarioPaciente]:
    """
    Processa áudio: transcrição (Groq) + análise (Claude).
    
    Args:
        audio_bytes: Conteúdo do arquivo de áudio em bytes
        filename: Nome do arquivo
        groq_api_key: Groq API Key
        anthropic_api_key: Anthropic API Key
    
    Returns:
        Tuple of (transcription_text, SumarioPaciente)
    """
    # Step 1: Transcribe with Groq Whisper
    transcription = transcribe_audio(audio_bytes, filename, groq_api_key)
    
    # Step 2: Analyze with Claude
    sumario = analyze_transcription(transcription, anthropic_api_key)
    
    return transcription, sumario
