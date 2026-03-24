from typing import Optional, List
from sqlalchemy.orm import Session
from ..models import Message


class MessageRepository:
    def __init__(self, db: Session):
        self.db = db

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        has_audio: bool = False,
        audio_filename: Optional[str] = None,
        transcription: Optional[str] = None,
    ) -> Message:
        message = Message(
            thread_id=thread_id,
            role=role,
            content=content,
            has_audio=has_audio,
            audio_filename=audio_filename,
            transcription=transcription,
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def get_thread_messages(self, thread_id: str, limit: int = 50) -> List[Message]:
        return (
            self.db.query(Message)
            .filter(Message.thread_id == thread_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
            .all()
        )
