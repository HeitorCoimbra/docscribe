WHISPER_MODEL = "whisper-large-v3-turbo"


def transcribe_audio(audio_bytes: bytes, filename: str, groq_api_key: str) -> str:
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
