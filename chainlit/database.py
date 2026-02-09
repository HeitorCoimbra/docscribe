"""
Database models and session management for DocScribe Chainlit.

Uses PostgreSQL with SQLAlchemy for:
- User management (OAuth)
- Session/Thread history
- Messages persistence
"""

import os
from datetime import datetime, date, timezone
from typing import Optional, List
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Date,
    Boolean,
    Integer
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import uuid


def _utcnow():
    return datetime.now(timezone.utc)


# Database URL from environment
# Uses DOCSCRIBE_DB_URL to avoid conflict with Chainlit's built-in data layer
# which auto-detects DATABASE_URL and expects its own table schema.
DATABASE_URL = os.environ.get(
    "DOCSCRIBE_DB_URL",
    os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/docscribe")
)

# Fix for Railway PostgreSQL URL format
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

Base = declarative_base()

# Lazy engine and session factory
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL)
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


# Keep SessionLocal as a property-like accessor for backward compatibility
class _SessionLocalProxy:
    def __call__(self):
        return get_session_factory()()

SessionLocal = _SessionLocalProxy()


# =============================================================================
# MODELS
# =============================================================================

class User(Base):
    """User model for OAuth authentication."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    provider = Column(String, nullable=True)  # google, github, etc.
    created_at = Column(DateTime, default=_utcnow)
    last_login = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # Relationships
    threads = relationship("Thread", back_populates="user", cascade="all, delete-orphan")


class Thread(Base):
    """
    Thread/Session model for conversation history.
    Each thread represents one patient summary session.
    """
    __tablename__ = "threads"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)

    # Session metadata
    title = Column(String, nullable=True)  # "Leito 1 - Maria"
    leito = Column(String, nullable=True)
    patient_name = Column(String, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=_utcnow, index=True)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    created_date = Column(Date, default=date.today, index=True)  # For grouping by day

    # Final summary (JSON)
    summary_json = Column(JSON, nullable=True)
    is_complete = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="threads")
    messages = relationship("Message", back_populates="thread", cascade="all, delete-orphan")


class Message(Base):
    """Message model for conversation persistence."""
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String, ForeignKey("threads.id"), nullable=False, index=True)

    role = Column(String, nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)

    # Optional: file attachments
    has_audio = Column(Boolean, default=False)
    audio_filename = Column(String, nullable=True)
    transcription = Column(Text, nullable=True)

    created_at = Column(DateTime, default=_utcnow)

    # Relationships
    thread = relationship("Thread", back_populates="messages")


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=get_engine())


def get_db():
    """Get database session."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


class ThreadRepository:
    """Repository for Thread operations."""

    def __init__(self, db_session):
        self.db = db_session

    def create_thread(self, user_id: str, title: str = None) -> Thread:
        """Create a new thread."""
        thread = Thread(
            user_id=user_id,
            title=title or "Nova Sessão",
            created_date=date.today()
        )
        self.db.add(thread)
        self.db.commit()
        self.db.refresh(thread)
        return thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get thread by ID."""
        return self.db.query(Thread).filter(Thread.id == thread_id).first()

    def update_thread_title(self, thread_id: str, leito: str, patient_name: str):
        """Update thread title with Leito and patient name."""
        thread = self.get_thread(thread_id)
        if thread:
            thread.leito = leito
            thread.patient_name = patient_name
            thread.title = f"Leito {leito} - {patient_name}"
            thread.updated_at = _utcnow()
            self.db.commit()

    def save_summary(self, thread_id: str, summary_json: dict):
        """Save final summary to thread."""
        thread = self.get_thread(thread_id)
        if thread:
            thread.summary_json = summary_json
            thread.is_complete = True
            thread.updated_at = _utcnow()
            self.db.commit()

    def get_user_threads(self, user_id: str) -> List[Thread]:
        """Get all threads for a user, ordered by date."""
        return (
            self.db.query(Thread)
            .filter(Thread.user_id == user_id)
            .order_by(Thread.created_at.desc())
            .all()
        )

    def get_threads_grouped_by_day(self, user_id: str) -> dict:
        """Get threads grouped by creation date."""
        threads = self.get_user_threads(user_id)

        grouped = {}
        for thread in threads:
            day_key = thread.created_date.strftime("%Y-%m-%d")
            day_label = thread.created_date.strftime("%d/%m/%Y")

            if day_key not in grouped:
                grouped[day_key] = {
                    "label": day_label,
                    "threads": []
                }
            grouped[day_key]["threads"].append(thread)

        return grouped

    def delete_thread(self, thread_id: str):
        """Delete a thread and all its messages."""
        thread = self.get_thread(thread_id)
        if thread:
            self.db.delete(thread)
            self.db.commit()


class MessageRepository:
    """Repository for Message operations."""

    def __init__(self, db_session):
        self.db = db_session

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        has_audio: bool = False,
        audio_filename: str = None,
        transcription: str = None
    ) -> Message:
        """Add a message to a thread."""
        message = Message(
            thread_id=thread_id,
            role=role,
            content=content,
            has_audio=has_audio,
            audio_filename=audio_filename,
            transcription=transcription
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def get_thread_messages(self, thread_id: str) -> List[Message]:
        """Get all messages for a thread."""
        return (
            self.db.query(Message)
            .filter(Message.thread_id == thread_id)
            .order_by(Message.created_at.asc())
            .all()
        )


class UserRepository:
    """Repository for User operations."""

    def __init__(self, db_session):
        self.db = db_session

    def get_or_create_user(
        self,
        email: str,
        name: str = None,
        avatar_url: str = None,
        provider: str = None
    ) -> User:
        """Get existing user or create new one."""
        user = self.db.query(User).filter(User.email == email).first()

        if not user:
            user = User(
                email=email,
                name=name,
                avatar_url=avatar_url,
                provider=provider
            )
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
        else:
            # Update last login
            user.last_login = _utcnow()
            if name:
                user.name = name
            if avatar_url:
                user.avatar_url = avatar_url
            self.db.commit()

        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(User).filter(User.email == email).first()
