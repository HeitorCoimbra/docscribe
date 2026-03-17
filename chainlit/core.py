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
    - quadro_clinico: Lista do quadro clínico atual
    - pendencias: Lista de pendências/tarefas em aberto
    - condutas: Lista de condutas tomadas ou planejadas
    """
    
    leito: str = Field(description="Número do leito. Se mencionado explicitamente, use apenas o número (ex: '1', '2'). Se ocluído/inaudível mas inferível pela sequência, use 'N (inferido)'. Se não identificável, use 'N/A'.")
    nome_paciente: str = Field(description="Nome completo do paciente como mencionado")
    
    quadro_clinico: list[str] = Field(
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
        
        linhas.append("Quadro Clínico:")
        for i, diag in enumerate(self.quadro_clinico, 1):
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

1. QUADRO CLÍNICO - O que incluir:
   - APENAS problemas médicos ATUAIS que requerem tratamento
   - Pós-operatório APENAS se for o contexto principal do caso
   - Condições patológicas explicitamente nomeadas

   O que NÃO incluir no quadro clínico:
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

=== OCLUSÃO DE NÚMERO DE LEITO ===
Em gravações de passagem de plantão, o número do leito pode ser inaudível ou ocluído. Quando isso ocorrer:

1. IDENTIFIQUE os leitos cujos números foram mencionados claramente antes e depois do trecho ocluído.
2. ANALISE se o perfil clínico do paciente ocluído é diferente do paciente do leito anterior — se for, trata-se de um paciente distinto, portanto um leito diferente.
3. INFIRA o número do leito faltante com base na sequência numérica. Exemplo: se o leito 4 foi mencionado antes e o leito 6 depois, e há um perfil clínico distinto entre eles, infira que trata-se do leito 5.
4. Quando inferir um número de leito, indique explicitamente que foi inferido. Use o formato: "leito [N] (inferido)".
5. Se não for possível inferir com segurança (sequência não-contígua, múltiplos leitos possíveis), use "leito ? (não identificado)" e liste os leitos vizinhos conhecidos como contexto.

=== ACRÔNIMOS COMUNS ===
VM = ventilação mecânica | CVC = cateter venoso central | SVD = sonda vesical de demora
DVA = droga vasoativa | IRA = insuficiência renal aguda | TOT = tubo orotraqueal
TQT = traqueostomia | ATB = antibiótico | BIC = bomba de infusão contínua
CIHDOTT = Comissão Intra-Hospitalar de Doação de Órgãos e Tecidos para Transplante

=== JARGÕES MÉDICOS ===
noradrenalina = nora, nor | midazolam = dormonid | fentanil = fenta
piperacilina+tazobactam = tazo, pipetazo | meropenem = mero | vancomicina = vanco
ceftazidima = fortaz
"""

HUMAN_PROMPT_TEMPLATE = """Analise a transcrição abaixo e extraia o sumário do paciente.

TRANSCRIÇÃO:
{transcription}

---

Retorne um JSON com a seguinte estrutura:
{{
    "leito": "número do leito (extraia da transcrição; se ocluído/inaudível, infira pela sequência numérica e use o formato 'N (inferido)'; use 'N/A' apenas se não for possível inferir)",
    "nome_paciente": "nome do paciente",
    "quadro_clinico": ["problema clínico 1", "problema clínico 2"],
    "pendencias": ["pendência 1", "pendência 2"],
    "condutas": ["Conduta 1 (começando com verbo no infinitivo)", "Conduta 2"]
}}

CHECKLIST antes de responder:
1. Quadro Clínico: São PROBLEMAS MÉDICOS ATUAIS?
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
    import io
    from groq import Groq

    groq_max_bytes = 25 * 1024 * 1024  # 25 MB limit
    if len(audio_bytes) > groq_max_bytes:
        size_mb = len(audio_bytes) / 1024 / 1024
        raise ValueError(
            f"Áudio muito grande ({size_mb:.0f} MB). O limite é 25 MB. "
            "Grave em partes menores ou envie um arquivo comprimido (mp3, opus)."
        )

    client = Groq(api_key=groq_api_key)

    # Use BytesIO with a name attribute — avoids encoding issues
    # that can occur with the (filename, bytes) tuple format in httpx
    file_obj = io.BytesIO(audio_bytes)
    file_obj.name = filename

    transcription = client.audio.transcriptions.create(
        file=file_obj,
        model=WHISPER_MODEL,
        temperature=0,
    )

    # Sanitize Unicode line/paragraph separators
    text = transcription.text
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    return text


# =============================================================================
# PDF REPORT GENERATION
# =============================================================================

def parse_summary_text(summary: str) -> dict:
    """Parse a formatted summary markdown string into a structured dict."""
    import re

    result = {
        "leito": "N/A",
        "nome_paciente": "—",
        "quadro_clinico": [],
        "pendencias": [],
        "condutas": [],
    }

    # Extract title from the first line matching "Leito X - Name"
    for line in summary.split("\n"):
        m = re.search(
            r"Leito\s+([^\s*\n]+(?:\s+\([^)]+\))?)\s*[-–—]\s*([^*\n]+)",
            line, re.IGNORECASE,
        )
        if m:
            result["leito"] = m.group(1).strip()
            result["nome_paciente"] = re.sub(r"\*+", "", m.group(2)).strip()
            break

    # Extract section items line by line
    current_section = None
    for line in summary.split("\n"):
        clean = re.sub(r"\*+", "", line).strip()
        if re.match(r"Quadro Cl.nico\s*:", clean, re.IGNORECASE):
            current_section = "quadro_clinico"
        elif re.match(r"Pend.ncias\s*:", clean, re.IGNORECASE):
            current_section = "pendencias"
        elif re.match(r"Condutas\s*:", clean, re.IGNORECASE):
            current_section = "condutas"
        elif current_section and clean:
            item = re.sub(r"^[\d]+[.\)]\s*", "", clean)
            item = re.sub(r"^[-•*]\s*", "", item).strip()
            if item:
                result[current_section].append(item)

    return result


def generate_pdf_report(summaries: list) -> bytes:
    """Generate a PDF report with 2 leitos per page (A4)."""
    import re
    from io import BytesIO
    from datetime import datetime

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title="DocScribe — Sumários de Passagem de Plantão",
    )

    styles = getSampleStyleSheet()

    doc_header_style = ParagraphStyle(
        "DocHeader", parent=styles["Normal"],
        fontSize=12, fontName="Helvetica-Bold",
        spaceAfter=2, textColor=colors.HexColor("#1a1a2e"),
    )
    date_style = ParagraphStyle(
        "DateStyle", parent=styles["Normal"],
        fontSize=9, fontName="Helvetica",
        spaceAfter=10, textColor=colors.HexColor("#666666"),
    )
    leito_title_style = ParagraphStyle(
        "LeitoTitle", parent=styles["Normal"],
        fontSize=12, fontName="Helvetica-Bold",
        spaceBefore=4, spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e"),
    )
    section_style = ParagraphStyle(
        "SectionHeader", parent=styles["Normal"],
        fontSize=10, fontName="Helvetica-Bold",
        spaceBefore=6, spaceAfter=2,
        textColor=colors.HexColor("#2d6a4f"),
    )
    item_style = ParagraphStyle(
        "Item", parent=styles["Normal"],
        fontSize=9, fontName="Helvetica",
        leading=13, leftIndent=12, spaceAfter=1,
    )

    # Sort summaries by leito number
    parsed = [parse_summary_text(s) for s in summaries]

    def _leito_key(d):
        m = re.match(r"(\d+)", d["leito"])
        return int(m.group(1)) if m else 9999

    parsed.sort(key=_leito_key)

    story = []

    # Document header
    date_str = datetime.now().strftime("%d/%m/%Y às %H:%M")
    story.append(Paragraph("DocScribe — Sumários de Passagem de Plantão", doc_header_style))
    story.append(Paragraph(f"Gerado em {date_str}", date_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2d6a4f"), spaceAfter=10))

    for i, leito_data in enumerate(parsed):
        story.append(Paragraph(
            f"Leito {leito_data['leito']} — {leito_data['nome_paciente']}",
            leito_title_style,
        ))

        story.append(Paragraph("Quadro Clínico", section_style))
        for j, item in enumerate(leito_data["quadro_clinico"] or ["—"], 1):
            story.append(Paragraph(f"{j}. {item}", item_style))

        story.append(Paragraph("Pendências", section_style))
        for j, item in enumerate(leito_data["pendencias"] or ["—"], 1):
            story.append(Paragraph(f"{j}. {item}", item_style))

        story.append(Paragraph("Condutas", section_style))
        for item in (leito_data["condutas"] or ["—"]):
            story.append(Paragraph(f"• {item}", item_style))

        is_last = (i + 1) == len(parsed)
        is_second_on_page = (i + 1) % 2 == 0

        if not is_last:
            if is_second_on_page:
                story.append(PageBreak())
            else:
                story.append(Spacer(1, 10))
                story.append(HRFlowable(
                    width="100%", thickness=0.5,
                    color=colors.HexColor("#aaaaaa"), spaceAfter=10,
                ))

    doc.build(story)
    return buffer.getvalue()
