from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from ..models import User


def _utcnow():
    return datetime.now(timezone.utc)


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_user(
        self,
        email: str,
        name: Optional[str] = None,
        avatar_url: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> User:
        user = self.db.query(User).filter(User.email == email).first()

        if not user:
            user = User(
                email=email,
                name=name,
                avatar_url=avatar_url,
                provider=provider,
            )
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
        else:
            user.last_login = _utcnow()
            if name:
                user.name = name
            if avatar_url:
                user.avatar_url = avatar_url
            self.db.commit()

        return user

    def get_user(self, user_id: str) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()
