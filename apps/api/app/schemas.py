import re
from pydantic import BaseModel, field_validator
from datetime import datetime, date
from typing import Optional


class LeitoSummary(BaseModel):
    leito: str
    nome_paciente: str
    quadro_clinico: list[str]
    pendencias: list[str]
    condutas: list[str]


def _parse_legacy_leito_string(s: str) -> dict:
    """Convert a legacy markdown summary string into a LeitoSummary-compatible dict."""
    result: dict = {
        "leito": "?",
        "nome_paciente": "—",
        "quadro_clinico": [],
        "pendencias": [],
        "condutas": [],
    }
    for line in s.split("\n"):
        m = re.search(
            r'Leito\s+([^\s*\n]+(?:\s+\([^)]+\))?)\s*[-\u2013\u2014]\s*([^*\n]+)',
            line,
            re.IGNORECASE,
        )
        if m:
            result["leito"] = m.group(1).strip()
            result["nome_paciente"] = re.sub(r"\*+", "", m.group(2)).strip()
            break
    current = None
    for line in s.split("\n"):
        clean = re.sub(r"\*+", "", line).strip()
        if re.match(r"Quadro Cl.nico\s*:", clean, re.IGNORECASE):
            current = "quadro_clinico"
        elif re.match(r"Pend.ncias\s*:", clean, re.IGNORECASE):
            current = "pendencias"
        elif re.match(r"Condutas\s*:", clean, re.IGNORECASE):
            current = "condutas"
        elif current and clean:
            item = re.sub(r"^[\d]+[.\)]\s*", "", clean)
            item = re.sub(r"^[-\u2022*]\s*", "", item).strip()
            if item:
                result[current].append(item)
    return result


def coerce_confirmed_leitos(raw: dict) -> dict:
    """Normalize confirmed_leitos: convert any legacy markdown string values to dicts."""
    result = {}
    for k, v in raw.items():
        if isinstance(v, str):
            result[k] = _parse_legacy_leito_string(v)
        else:
            result[k] = v
    return result


class ThreadCreate(BaseModel):
    title: Optional[str] = None


class ThreadUpdate(BaseModel):
    title: Optional[str] = None
    is_complete: Optional[bool] = None


class ThreadResponse(BaseModel):
    id: str
    title: Optional[str]
    leito: Optional[str]
    patient_name: Optional[str]
    created_at: datetime
    updated_at: datetime
    created_date: date
    confirmed_leitos: dict[str, LeitoSummary]
    is_complete: bool

    model_config = {"from_attributes": True}

    @field_validator("confirmed_leitos", mode="before")
    @classmethod
    def coerce_legacy_leitos(cls, v: dict) -> dict:
        if not isinstance(v, dict):
            return v
        return coerce_confirmed_leitos(v)


class MessageResponse(BaseModel):
    id: str
    thread_id: str
    role: str
    content: str
    has_audio: bool
    transcription: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class ThreadDetailResponse(ThreadResponse):
    messages: list[MessageResponse] = []


class TextMessageRequest(BaseModel):
    content: str


class ThreadGroupResponse(BaseModel):
    date: str
    label: str
    threads: list[ThreadResponse]
