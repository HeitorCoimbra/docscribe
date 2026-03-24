from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import get_settings

Base = declarative_base()

_engine = None
_SessionLocal = None


def _get_db_url() -> str:
    url = get_settings().docscribe_db_url
    # Fix Railway PostgreSQL URL format
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_get_db_url())
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


SessionLocal = get_session_factory


def get_db():
    """FastAPI dependency that yields a database session."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables on startup."""
    from . import models  # noqa: F401 — ensures models are registered with Base
    Base.metadata.create_all(bind=get_engine())
