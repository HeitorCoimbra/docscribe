from sqlalchemy import Column, String, Text, DateTime, Date, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .database import Base
import uuid
from datetime import datetime, date, timezone


def _utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    provider = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    last_login = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    threads = relationship("Thread", back_populates="user", cascade="all, delete-orphan")


class Thread(Base):
    __tablename__ = "threads"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String, nullable=True)
    leito = Column(String, nullable=True)
    patient_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    created_date = Column(Date, default=date.today, index=True)
    summary_json = Column(JSON, nullable=True)
    confirmed_leitos = Column(JSON, default=dict)  # {leito_num: formatted_summary}
    is_complete = Column(Boolean, default=False)

    user = relationship("User", back_populates="threads")
    messages = relationship(
        "Message",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String, ForeignKey("threads.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    has_audio = Column(Boolean, default=False)
    audio_filename = Column(String, nullable=True)
    transcription = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    thread = relationship("Thread", back_populates="messages")
