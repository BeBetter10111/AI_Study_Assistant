import hashlib
import hmac
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.api.security.rate_limiter import limiter
from app.config import settings
from app.core.email import send_password_reset_email
from app.core.login_throttle import login_throttle
from app.models.models import PasswordResetToken, User

logger = logging.getLogger(__name__)

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Dummy hash to ensure login always takes bcrypt time, even when the user does not exist -> do not reveal
# whether an account exists through response timing.
_DUMMY_HASH = pwd_context.hash("timing-equalisation-only")

MAX_PASSWORD_LENGTH = 72

PUBLIC_SELF_REGISTER_ROLE = "Student"


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=150)
    full_name: Optional[str] = Field(default=None, max_length=255)
    email: EmailStr
    phone: Optional[str] = Field(default=None, max_length=30)
    password: str = Field(min_length=8, max_length=MAX_PASSWORD_LENGTH)


class LoginRequest(BaseModel):
    identifier: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _create_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": user_id, "exp": expire, "iat": now.timestamp()}  # Float: distinguishes tokens issued within the same second.
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def register(request: Request, payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
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

    # Security: role is ALWAYS forced to the default unprivileged role at public
    # registration — never read from the client payload.  An Admin must use the
    # dedicated admin endpoint (future work) to elevate a user's role.
    # bcrypt takes ~100-300ms of CPU time: run it in a thread pool so it does not block the event loop.
    hashed_password = await run_in_threadpool(pwd_context.hash, payload.password)
    user = User(
        username=payload.username,
        full_name=payload.full_name,
        email=payload.email,
        phone=payload.phone or None,
        hashed_password=hashed_password,
        role=PUBLIC_SELF_REGISTER_ROLE,
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:
        # If Sign up is clicked twice / two requests run concurrently, both can pass the duplicate check
        # above; the second then violates UNIQUE -> previously this returned HTTP 500.
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username, email or phone number is already registered.",
        ) from exc
    await db.refresh(user)

    return TokenResponse(access_token=_create_access_token(str(user.id)))


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(request: Request, payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    # Soft-lock by ACCOUNT (IP-based rate limiting cannot stop password guessing from multiple IPs). Check BEFORE
    # verifying the password: a blocked attempt does not reveal whether it was correct, and the same behavior
    # applies to nonexistent accounts so real accounts are not exposed.
    wait = await login_throttle.check(payload.identifier)
    if wait > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed attempts for this account. Please wait {wait} seconds and try again, "
                   "or reset your password.",
            headers={"Retry-After": str(wait)},
        )

    # Allow login by either username or email
    result = await db.execute(
        select(User).where(
            or_(User.email == payload.identifier, User.username == payload.identifier)
        )
    )
    user = result.scalar_one_or_none()

    stored_hash = user.hashed_password if user is not None else _DUMMY_HASH
    password_ok = await run_in_threadpool(pwd_context.verify, payload.password, stored_hash)
    if user is None or not password_ok:
        await login_throttle.record_failure(payload.identifier)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifier or password is incorrect.",
        )

    await login_throttle.reset(payload.identifier)
    return TokenResponse(access_token=_create_access_token(str(user.id)))


# ---------------------------------------------------------------------------
# User profile & password change (authenticated users)
# ---------------------------------------------------------------------------
class UserProfile(BaseModel):
    id: uuid.UUID
    username: str
    full_name: Optional[str] = None
    email: str
    phone: Optional[str] = None
    role: str
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class UpdateProfileRequest(BaseModel):
    # Only submitted fields are changed. Email intentionally cannot be edited here: the password-reset
    # code is sent to the email address, so changing it without verification could enable account takeover
    # if a token is exposed.
    username: Optional[str] = Field(default=None, max_length=150)
    full_name: Optional[str] = Field(default=None, max_length=255)
    phone: Optional[str] = Field(default=None, max_length=30)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)
    new_password: str = Field(min_length=8, max_length=MAX_PASSWORD_LENGTH)
    confirm_password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)

    @field_validator("new_password")
    @classmethod
    def validate_bytes(cls, v: str) -> str:
        if len(v.encode("utf-8")) > MAX_PASSWORD_LENGTH:
            raise ValueError(f"password must be at most {MAX_PASSWORD_LENGTH} bytes")
        return v


@router.get("/me", response_model=UserProfile)
async def read_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.patch("/me", response_model=UserProfile)
@limiter.limit("20/minute")
async def update_me(
    request: Request,
    payload: UpdateProfileRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sent = payload.model_fields_set
    changes: dict = {}

    if "username" in sent:
        username = (payload.username or "").strip()
        if len(username) < 3:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Username must be at least 3 characters.",
            )
        changes["username"] = username
    if "full_name" in sent:
        changes["full_name"] = (payload.full_name or "").strip() or None
    if "phone" in sent:
        changes["phone"] = (payload.phone or "").strip() or None

    if changes.get("username") and changes["username"] != current_user.username:
        taken = await db.execute(
            select(User.id).where(User.username == changes["username"], User.id != current_user.id)
        )
        if taken.first() is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username is already taken.")
    if changes.get("phone") and changes["phone"] != current_user.phone:
        taken = await db.execute(
            select(User.id).where(User.phone == changes["phone"], User.id != current_user.id)
        )
        if taken.first() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Phone number is already registered."
            )

    for field, value in changes.items():
        setattr(current_user, field, value)
    # current_user belongs to the get_current_user dependency's session, not the `db` session above.
    # merge() attaches it to `db` so everything can be committed together.
    user = await db.merge(current_user)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or phone number is already registered.",
        ) from exc
    await db.refresh(user)
    return user


@router.post("/change-password", response_model=TokenResponse)
@limiter.limit("5/minute")
async def change_password(
    request: Request,
    payload: ChangePasswordRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if payload.new_password != payload.confirm_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Passwords do not match.")

    if not await run_in_threadpool(pwd_context.verify, payload.current_password, current_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect.")

    if payload.new_password == payload.current_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from your current password.",
        )

    user = await db.merge(current_user)
    user.hashed_password = await run_in_threadpool(pwd_context.hash, payload.new_password)
    # All JWTs issued before this point (including the current token) become invalid -> return a new token.
    user.password_changed_at = _now()
    await db.commit()

    logger.info("Password changed for user %s", user.id)
    return TokenResponse(access_token=_create_access_token(str(user.id)))


RESET_PURPOSE = "password_reset"
GENERIC_FORGOT_MESSAGE = "If an account exists for that email, a verification code has been sent."
INVALID_CODE_MESSAGE = "Invalid or expired code. Please check the code or request a new one."
INVALID_SESSION_MESSAGE = "Reset session is invalid or has expired. Please start over."


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class VerifyResetCodeRequest(BaseModel):
    email: EmailStr
    code: str = Field(min_length=1, max_length=64)


class VerifyResetCodeResponse(BaseModel):
    reset_token: str
    expires_in: int  # seconds


class ResetPasswordRequest(BaseModel):
    reset_token: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=MAX_PASSWORD_LENGTH)
    confirm_password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)

    @field_validator("new_password")
    @classmethod
    def validate_bytes(cls, v: str) -> str:
        # bcrypt only reads the first 72 bytes; multibyte characters (Vietnamese, emoji) can exceed this limit.
        if len(v.encode("utf-8")) > MAX_PASSWORD_LENGTH:
            raise ValueError(f"password must be at most {MAX_PASSWORD_LENGTH} bytes")
        return v


class MessageResponse(BaseModel):
    message: str


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    # SQLite (tests) returns naive datetimes; Postgres returns aware datetimes.
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.strip().lower().encode("utf-8")).hexdigest()


async def _find_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    result = await db.execute(select(User).where(func.lower(User.email) == email.lower()))
    return result.scalar_one_or_none()


@router.post("/forgot-password", response_model=MessageResponse)
@limiter.limit("5/minute")
async def forgot_password(
    request: Request,
    payload: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    user = await _find_user_by_email(db, payload.email)
    if user is None:
        return MessageResponse(message=GENERIC_FORGOT_MESSAGE)

    now = _now()
    latest = (
        await db.execute(
            select(PasswordResetToken)
            .where(PasswordResetToken.user_id == user.id)
            .order_by(PasswordResetToken.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if latest is not None and latest.created_at is not None:
        age = (now - _aware(latest.created_at)).total_seconds()
        if age < settings.password_reset_resend_cooldown_s:
            # Prevent email spam: silently ignore the request while still returning a generic message.
            return MessageResponse(message=GENERIC_FORGOT_MESSAGE)

    await db.execute(
        update(PasswordResetToken)
        .where(PasswordResetToken.user_id == user.id, PasswordResetToken.used.is_(False))
        .values(used=True)
    )
    code = str(uuid.uuid4())
    db.add(
        PasswordResetToken(
            user_id=user.id,
            token_hash=_hash_code(code),
            expires_at=now + timedelta(minutes=settings.password_reset_expire_minutes),
        )
    )
    await db.commit()

    background_tasks.add_task(send_password_reset_email, user.email, code)
    return MessageResponse(message=GENERIC_FORGOT_MESSAGE)


@router.post("/verify-reset-code", response_model=VerifyResetCodeResponse)
@limiter.limit("10/minute")
async def verify_reset_code(request: Request, payload: VerifyResetCodeRequest, db: AsyncSession = Depends(get_db)):
    invalid = HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=INVALID_CODE_MESSAGE)

    user = await _find_user_by_email(db, payload.email)
    if user is None:
        raise invalid

    token = (
        await db.execute(
            select(PasswordResetToken)
            .where(PasswordResetToken.user_id == user.id, PasswordResetToken.used.is_(False))
            .order_by(PasswordResetToken.created_at.desc())
            .limit(1)
            .with_for_update()
        )
    ).scalar_one_or_none()
    now = _now()
    if (
        token is None
        or _aware(token.expires_at) <= now
        or token.failed_attempts >= settings.password_reset_max_attempts
    ):
        raise invalid

    if not hmac.compare_digest(token.token_hash, _hash_code(payload.code)):
        token.failed_attempts += 1
        await db.commit()
        raise invalid

    session_expires = min(
        now + timedelta(minutes=settings.password_reset_session_minutes),
        _aware(token.expires_at),
    )
    token.verified_at = now
    await db.commit()

    reset_token = jwt.encode(
        {
            "sub": str(user.id),
            "jti": str(token.id),
            "purpose": RESET_PURPOSE,
            "exp": session_expires,
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    return VerifyResetCodeResponse(
        reset_token=reset_token,
        expires_in=max(int((session_expires - now).total_seconds()), 0),
    )


@router.post("/reset-password", response_model=MessageResponse)
@limiter.limit("10/minute")
async def reset_password(request: Request, payload: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    session_error = HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=INVALID_SESSION_MESSAGE)

    try:
        claims = jwt.decode(payload.reset_token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if claims.get("purpose") != RESET_PURPOSE:
            raise session_error
        user_id = uuid.UUID(claims["sub"])
        token_id = uuid.UUID(claims["jti"])
    except (JWTError, KeyError, ValueError, TypeError) as exc:
        raise session_error from exc

    if payload.new_password != payload.confirm_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Passwords do not match.")

    token = (
        await db.execute(
            select(PasswordResetToken).where(PasswordResetToken.id == token_id).with_for_update()
        )
    ).scalar_one_or_none()
    if (
        token is None
        or token.user_id != user_id
        or token.used
        or token.verified_at is None
        or _aware(token.expires_at) <= _now()
    ):
        raise session_error

    user = (await db.execute(select(User).where(User.id == user_id).with_for_update())).scalar_one_or_none()
    if user is None:
        raise session_error

    if await run_in_threadpool(pwd_context.verify, payload.new_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from your current password.",
        )

    now = _now()
    user.hashed_password = await run_in_threadpool(pwd_context.hash, payload.new_password)
    user.password_changed_at = now  
    token.used = True
    await db.execute(
        update(PasswordResetToken)
        .where(PasswordResetToken.user_id == user.id, PasswordResetToken.used.is_(False))
        .values(used=True)
    )
    await db.commit()

    logger.info("Password reset completed for user %s", user.id)
    return MessageResponse(message="Password updated successfully. Please log in with your new password.")
