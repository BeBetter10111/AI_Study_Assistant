"""Dependency Injection: DB session, Current User."""
import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import async_session
from app.models.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def get_db():
    async with async_session() as session:
        yield session


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Giải mã JWT (do /api/auth/login cấp) và trả về User tương ứng."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        raw_user_id = payload.get("sub")
        if raw_user_id is None:
            raise credentials_exception
    except JWTError as exc:
        raise credentials_exception from exc

    try:
        user_id = uuid.UUID(raw_user_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise credentials_exception from exc

    try:
        result = await db.execute(select(User).where(User.id == user_id))
    except Exception as exc:
        raise credentials_exception from exc

    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception

    return user
