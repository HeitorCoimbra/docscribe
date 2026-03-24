from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import init_db
from .routers import audio, messages, pdf, text, threads


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="DocScribe API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok"}

    # Mount all routers
    prefix = "/api/v1"
    app.include_router(threads.router, prefix=prefix)
    app.include_router(audio.router, prefix=prefix)
    app.include_router(text.router, prefix=prefix)
    app.include_router(messages.router, prefix=prefix)
    app.include_router(pdf.router, prefix=prefix)

    return app


app = create_app()
