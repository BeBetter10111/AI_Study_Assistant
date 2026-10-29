import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username        = Column(String(150), unique=True, index=True, nullable=False)
    full_name       = Column(String(255), nullable=True)
    email           = Column(String(255), unique=True, index=True, nullable=False)
    phone           = Column(String(30),  unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    role            = Column(String(50),  nullable=False, default="Staff")
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    documents = relationship("Document", back_populates="owner")
    quiz_attempts = relationship("QuizAttempt", back_populates="user")


class UserRLState(Base):
    """Trạng thái RL bền vững theo user — thay thế dict in-memory.

    Mỗi user có đúng một row, được upsert sau mỗi lần submit quiz.
    Đây là ground truth; Redis là cache nhanh phía trên.
    """

    __tablename__ = "user_rl_states"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), primary_key=True, nullable=False)
    difficulty_index = Column(Integer, default=0, nullable=False)
    # Lịch sử reward (float list) lưu dưới dạng JSON text
    reward_history_json = Column(Text, default="[]", nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    filename = Column(String(512), nullable=False)
    mime_type = Column(String(128))
    status = Column(String(32), default="pending", nullable=False)
    storage_path = Column(String(512))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="documents")
    quiz_attempts = relationship("QuizAttempt", back_populates="document")


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    quiz_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    is_correct = Column(Boolean, nullable=False)
    response_time_s = Column(Float, nullable=False)
    hints_used = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="quiz_attempts")
    document = relationship("Document", back_populates="quiz_attempts")
