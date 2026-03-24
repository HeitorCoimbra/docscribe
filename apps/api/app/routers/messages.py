from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import User
from ..repositories.message import MessageRepository
from ..repositories.thread import ThreadRepository
from ..schemas import MessageResponse

router = APIRouter(tags=["messages"])


@router.get("/threads/{thread_id}/messages", response_model=list[MessageResponse])
def list_messages(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all messages for a thread."""
    thread_repo = ThreadRepository(db)
    thread = thread_repo.get_thread(thread_id)
    if not thread or thread.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Thread not found")

    msg_repo = MessageRepository(db)
    messages = msg_repo.get_thread_messages(thread_id)
    return [MessageResponse.model_validate(m) for m in messages]
