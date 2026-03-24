import re
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

# ---------------------------------------------------------------------------
# Prompts ported from legacy core.py
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Structured extraction tool (Claude tool_use)
# ---------------------------------------------------------------------------

LEITO_EXTRACTION_TOOL = {
    "name": "salvar_leitos",
    "description": "Salva os sumários estruturados de todos os leitos presentes na resposta.",
    "input_schema": {
        "type": "object",
        "properties": {
            "leitos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "leito": {
                            "type": "string",
                            "description": "Número do leito (ex: '1', '4A', '7 (inferido)')",
                        },
                        "nome_paciente": {"type": "string"},
                        "quadro_clinico": {"type": "array", "items": {"type": "string"}},
                        "pendencias": {"type": "array", "items": {"type": "string"}},
                        "condutas": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "leito",
                        "nome_paciente",
                        "quadro_clinico",
                        "pendencias",
                        "condutas",
                    ],
                },
            }
        },
        "required": ["leitos"],
    },
}


def extract_leitos_structured(full_response: str, client, model: str) -> list[dict]:
    """Post-stream: ask Claude to parse its own response into structured leito data."""
    result = client.messages.create(
        model=model,
        max_tokens=4096,
        tools=[LEITO_EXTRACTION_TOOL],
        tool_choice={"type": "auto"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Analise o texto abaixo e extraia TODOS os leitos em formato estruturado "
                    "usando a ferramenta salvar_leitos.\n\n"
                    f"{full_response}"
                ),
            }
        ],
    )
    for block in result.content:
        if block.type == "tool_use" and block.name == "salvar_leitos":
            return block.input.get("leitos", [])
    return []


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_leito_number(summary: str) -> Optional[str]:
    """Extract leito number from a formatted summary string (used for history collapsing)."""
    match = re.search(r'\*\*Leito\s+([^\s*-]+)', summary)
    return match.group(1).strip() if match else None


def _format_leito_for_prompt(data: dict) -> str:
    """Format a LeitoSummary dict as markdown for use in system prompt context."""
    nome = data.get("nome_paciente", "—")
    leito = data.get("leito", "?")
    lines = [f"**Leito {leito} - {nome}**"]

    lines.append("\n**Quadro Clínico:**")
    for i, item in enumerate(data.get("quadro_clinico") or [], 1):
        lines.append(f"{i}. {item}")

    lines.append("\n**Pendências:**")
    for i, item in enumerate(data.get("pendencias") or [], 1):
        lines.append(f"{i}. {item}")

    lines.append("\n**Condutas:**")
    for item in data.get("condutas") or []:
        lines.append(f"- {item}")

    return "\n".join(lines)


def build_system_prompt(confirmed_leitos: Optional[dict] = None) -> str:
    """Build the extraction-mode system prompt.

    Injects already-confirmed leitos as context so Claude only processes new ones.
    """
    current_date = datetime.now().strftime("%d/%m/%Y")

    confirmed_context = ""
    if confirmed_leitos:
        block = "\n\n".join(_format_leito_for_prompt(v) for v in confirmed_leitos.values())
        confirmed_context = (
            f"\n=== LEITOS JÁ CONFIRMADOS (NÃO REPRODUZA ESTES NA RESPOSTA) ===\n"
            f"{block}\n=== FIM DOS LEITOS CONFIRMADOS ===\n\n"
            "IMPORTANTE: Não repita os leitos acima. Extraia SOMENTE os leitos novos da transcrição atual."
        )

    return f"""Data de hoje: {current_date}

{SYSTEM_PROMPT}

=== INSTRUCOES DE COMPORTAMENTO ===

Quando receber uma transcrição de áudio ou mensagem de texto, extraia TODOS os leitos presentes e responda com TODOS eles.
Não faça perguntas antes de mostrar os sumários.

Para campos não preenchíveis, use: 🔴 PENDENTE

FORMATO OBRIGATÓRIO DE RESPOSTA (repita o bloco abaixo para CADA leito encontrado, separados por ---):

**Leito [X] - [Nome do Paciente]**

**Quadro Clínico:**
1. [problema clínico extraído ou 🔴 PENDENTE]

**Pendências:**
1. [pendência extraída ou 🔴 PENDENTE]

**Condutas:**
- [conduta com verbo no infinitivo ou 🔴 PENDENTE]

---

Se a transcrição contiver múltiplos leitos, liste TODOS em sequência com --- entre eles.
Condutas SEMPRE começam com verbo no INFINITIVO. Seja conciso.

{confirmed_context}"""
