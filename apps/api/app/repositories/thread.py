from datetime import datetime, date, timezone
from typing import Optional, List
from sqlalchemy.orm import Session
from ..models import Thread


def _utcnow():
    return datetime.now(timezone.utc)


def _ptbr_date_label(d: date) -> str:
    """Return a human-readable PT-BR label for a date relative to today."""
    today = date.today()
    delta = (today - d).days
    if delta == 0:
        return "Hoje"
    if delta == 1:
        return "Ontem"
    # Map month numbers to Portuguese names
    months = [
        "janeiro", "fevereiro", "março", "abril", "maio", "junho",
        "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
    ]
    return f"{d.day} de {months[d.month - 1]}"


class ThreadRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_thread(self, user_id: str, title: Optional[str] = None) -> Thread:
        thread = Thread(
            user_id=user_id,
            title=title or "Nova Sessão",
            created_date=date.today(),
            confirmed_leitos={},
        )
        self.db.add(thread)
        self.db.commit()
        self.db.refresh(thread)
        return thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        return self.db.query(Thread).filter(Thread.id == thread_id).first()

    def get_user_threads(self, user_id: str) -> List[Thread]:
        return (
            self.db.query(Thread)
            .filter(Thread.user_id == user_id)
            .order_by(Thread.created_at.desc())
            .all()
        )

    def get_threads_grouped_by_day(self, user_id: str) -> List[dict]:
        """Return threads grouped by creation date as a list of dicts with PT-BR labels."""
        threads = self.get_user_threads(user_id)

        groups: dict[str, dict] = {}
        for thread in threads:
            day_key = thread.created_date.strftime("%Y-%m-%d")
            if day_key not in groups:
                groups[day_key] = {
                    "date": day_key,
                    "label": _ptbr_date_label(thread.created_date),
                    "threads": [],
                }
            groups[day_key]["threads"].append(thread)

        # Return ordered by date descending
        return [groups[k] for k in sorted(groups.keys(), reverse=True)]

    def _compute_session_number(self, thread: Thread) -> int:
        threads_on_day = (
            self.db.query(Thread)
            .filter(Thread.user_id == thread.user_id, Thread.created_date == thread.created_date)
            .order_by(Thread.created_at.asc())
            .all()
        )
        for i, t in enumerate(threads_on_day, 1):
            if t.id == thread.id:
                return i
        return 1

    def _generate_title(self, thread: Thread, leito_count: int) -> str:
        session_num = self._compute_session_number(thread)
        date_str = thread.created_date.strftime("%d/%m")
        leito_label = "1 leito" if leito_count == 1 else f"{leito_count} leitos"
        return f"{date_str} · Sessão {session_num} · {leito_label}"

    def update_confirmed_leitos(self, thread_id: str, confirmed_leitos: dict):
        thread = self.get_thread(thread_id)
        if thread:
            thread.confirmed_leitos = confirmed_leitos
            if confirmed_leitos:
                thread.title = self._generate_title(thread, len(confirmed_leitos))
            thread.updated_at = _utcnow()
            self.db.commit()

    def update_thread(self, thread_id: str, **kwargs):
        """Generic update for title, is_complete, etc."""
        thread = self.get_thread(thread_id)
        if not thread:
            return None
        for key, value in kwargs.items():
            if hasattr(thread, key) and value is not None:
                setattr(thread, key, value)
        thread.updated_at = _utcnow()
        self.db.commit()
        self.db.refresh(thread)
        return thread

    def save_summary(self, thread_id: str, summary_json: dict):
        thread = self.get_thread(thread_id)
        if thread:
            thread.summary_json = summary_json
            thread.is_complete = True
            thread.updated_at = _utcnow()
            self.db.commit()

    def delete_thread(self, thread_id: str):
        thread = self.get_thread(thread_id)
        if thread:
            self.db.delete(thread)
            self.db.commit()
