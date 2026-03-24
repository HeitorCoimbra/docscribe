from typing import AsyncGenerator


async def stream_summary(
    client,
    messages: list[dict],
    system_prompt: str,
    model: str,
    max_tokens: int = 4096,
) -> AsyncGenerator[str, None]:
    """
    Async generator that streams text chunks from Anthropic Claude.

    Args:
        client: Anthropic client instance
        messages: List of {"role": ..., "content": ...} dicts
        system_prompt: System prompt string
        model: Claude model ID
        max_tokens: Maximum tokens to generate

    Yields:
        Text chunks as they arrive from the API
    """
    import asyncio

    # The Anthropic SDK's streaming context manager is synchronous; run it in a
    # thread executor so we don't block the event loop.
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _run_stream():
        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    loop.call_soon_threadsafe(queue.put_nowait, text)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    # Run the blocking stream in a thread pool
    loop.run_in_executor(None, _run_stream)

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk
