from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import User
from ..repositories.thread import ThreadRepository
from ..repositories.message import MessageRepository
from ..schemas import (
    ThreadCreate,
    ThreadUpdate,
    ThreadResponse,
    ThreadDetailResponse,
    ThreadGroupResponse,
    MessageResponse,
)

router = APIRouter(prefix="/threads", tags=["threads"])


@router.get("", response_model=list[ThreadGroupResponse])
def list_threads(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List current user's threads grouped by day."""
    repo = ThreadRepository(db)
    groups = repo.get_threads_grouped_by_day(current_user.id)
    return [
        ThreadGroupResponse(
            date=g["date"],
            label=g["label"],
            threads=[ThreadResponse.model_validate(t) for t in g["threads"]],
        )
        for g in groups
    ]


@router.post("", response_model=ThreadResponse, status_code=201)
def create_thread(
    payload: ThreadCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new thread."""
    repo = ThreadRepository(db)
    thread = repo.create_thread(user_id=current_user.id, title=payload.title)
    return ThreadResponse.model_validate(thread)


@router.get("/{thread_id}", response_model=ThreadDetailResponse)
def get_thread(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a thread with its messages and confirmed leitos."""
    repo = ThreadRepository(db)
    thread = repo.get_thread(thread_id)
    if not thread or thread.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Thread not found")

    msg_repo = MessageRepository(db)
    messages = msg_repo.get_thread_messages(thread_id)

    response = ThreadDetailResponse.model_validate(thread)
    response.messages = [MessageResponse.model_validate(m) for m in messages]
    return response


@router.patch("/{thread_id}", response_model=ThreadResponse)
def update_thread(
    thread_id: str,
    payload: ThreadUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update thread title or is_complete flag."""
    repo = ThreadRepository(db)
    thread = repo.get_thread(thread_id)
    if not thread or thread.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Thread not found")

    updates = payload.model_dump(exclude_none=True)
    thread = repo.update_thread(thread_id, **updates)
    return ThreadResponse.model_validate(thread)


@router.delete("/{thread_id}", status_code=204)
def delete_thread(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a thread and all its messages."""
    repo = ThreadRepository(db)
    thread = repo.get_thread(thread_id)
    if not thread or thread.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Thread not found")
    repo.delete_thread(thread_id)
