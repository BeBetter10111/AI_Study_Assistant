import hashlib
import hmac
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.config import settings
from app.core.email import send_password_reset_email
from app.models.models import PasswordResetToken, User

logger = logging.getLogger(__name__)

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
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.access_token_expire_minutes)
    # "iat" cho phép vô hiệu hoá token cũ sau khi user đổi mật khẩu (xem get_current_user).
    payload = {"sub": user_id, "exp": expire, "iat": now.timestamp()}  # float: phân biệt được token cấp trước/sau trong cùng một giây
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


# ---------------------------------------------------------------------------
# Quên mật khẩu: forgot-password -> verify-reset-code -> reset-password
# ---------------------------------------------------------------------------
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
        # bcrypt chỉ đọc 72 byte đầu; ký tự đa byte (tiếng Việt, emoji) có thể vượt giới hạn.
        if len(v.encode("utf-8")) > MAX_PASSWORD_LENGTH:
            raise ValueError(f"password must be at most {MAX_PASSWORD_LENGTH} bytes")
        return v


class MessageResponse(BaseModel):
    message: str


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    # SQLite (test) trả về datetime naive; Postgres trả về aware.
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.strip().lower().encode("utf-8")).hexdigest()


async def _find_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    result = await db.execute(select(User).where(func.lower(User.email) == email.lower()))
    return result.scalar_one_or_none()


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(
    payload: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Bước 1: nhận email, gửi mã UUID. Luôn trả cùng một thông báo để không lộ email nào đã đăng ký."""
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
            # Chống spam email: bỏ qua im lặng, vẫn trả thông báo chung.
            return MessageResponse(message=GENERIC_FORGOT_MESSAGE)

    # Mã mới làm mọi mã cũ chưa dùng mất hiệu lực -> chỉ còn đúng một mã hợp lệ mỗi user.
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

    # Gửi nền: thời gian phản hồi không phụ thuộc email có tồn tại hay không.
    background_tasks.add_task(send_password_reset_email, user.email, code)
    return MessageResponse(message=GENERIC_FORGOT_MESSAGE)


@router.post("/verify-reset-code", response_model=VerifyResetCodeResponse)
async def verify_reset_code(payload: VerifyResetCodeRequest, db: AsyncSession = Depends(get_db)):
    """Bước 2: kiểm tra mã trong email. Đúng -> cấp reset_token ngắn hạn để sang bước đặt mật khẩu."""
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
async def reset_password(payload: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    """Bước 3: đặt mật khẩu mới. Cập nhật user + tiêu thụ mã trong cùng một transaction."""
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

    # Khoá hàng token: hai request song song cùng dùng một mã thì chỉ một request thắng.
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

    if pwd_context.verify(payload.new_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from your current password.",
        )

    now = _now()
    user.hashed_password = pwd_context.hash(payload.new_password)
    user.password_changed_at = now  # đăng xuất mọi phiên cũ
    token.used = True
    # Huỷ mọi mã reset khác còn hiệu lực của user này.
    await db.execute(
        update(PasswordResetToken)
        .where(PasswordResetToken.user_id == user.id, PasswordResetToken.used.is_(False))
        .values(used=True)
    )
    await db.commit()

    logger.info("Password reset completed for user %s", user.id)
    return MessageResponse(message="Password updated successfully. Please log in with your new password.")
