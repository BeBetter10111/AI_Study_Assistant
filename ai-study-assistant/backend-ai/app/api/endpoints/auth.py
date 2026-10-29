from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.config import settings
from app.models.models import User

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

MAX_PASSWORD_LENGTH = 72

VALID_ROLES = {"Admin", "Manager", "Staff", "Viewer"}


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=150)
    full_name: Optional[str] = Field(default=None, max_length=255)
    email: EmailStr
    phone: Optional[str] = Field(default=None, max_length=30)
    password: str = Field(min_length=8, max_length=MAX_PASSWORD_LENGTH)
    role: str = Field(default="Staff", max_length=50)

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}")
        return v


class LoginRequest(BaseModel):
    # Accept username OR email in the same field
    identifier: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check uniqueness for username, email, and phone in a single query
    filters = [
        User.username == payload.username,
        User.email == payload.email,
    ]
    if payload.phone:
        filters.append(User.phone == payload.phone)

    result = await db.execute(select(User).where(or_(*filters)))
    existing = result.scalar_one_or_none()

    if existing is not None:
        if existing.username == payload.username:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username is already taken.")
        if existing.email == payload.email:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already registered.")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Phone number is already registered.")

    user = User(
        username=payload.username,
        full_name=payload.full_name,
        email=payload.email,
        phone=payload.phone or None,
        hashed_password=pwd_context.hash(payload.password),
        role=payload.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return TokenResponse(access_token=_create_access_token(str(user.id)))


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    # Allow login by either username or email
    result = await db.execute(
        select(User).where(
            or_(User.email == payload.identifier, User.username == payload.identifier)
        )
    )
    user = result.scalar_one_or_none()

    if user is None or not pwd_context.verify(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifier or password is incorrect.",
        )

    return TokenResponse(access_token=_create_access_token(str(user.id)))
