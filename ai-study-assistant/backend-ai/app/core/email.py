"""Gửi email (SMTP) cho luồng quên mật khẩu."""
import logging
import smtplib
import ssl
from email.message import EmailMessage

from app.config import settings

logger = logging.getLogger(__name__)


def _send(message: EmailMessage) -> None:
    context = ssl.create_default_context()
    if settings.smtp_use_ssl:
        with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, context=context, timeout=15) as smtp:
            if settings.smtp_username:
                smtp.login(settings.smtp_username, settings.smtp_password)
            smtp.send_message(message)
        return
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as smtp:
        if settings.smtp_use_starttls:
            smtp.starttls(context=context)
        if settings.smtp_username:
            smtp.login(settings.smtp_username, settings.smtp_password)
        smtp.send_message(message)


def send_password_reset_email(to_email: str, code: str) -> None:
    """Gửi mã reset. Chạy trong BackgroundTasks nên không bao giờ được ném lỗi ra ngoài."""
    minutes = settings.password_reset_expire_minutes

    if not settings.smtp_host:
        # Môi trường dev chưa cấu hình SMTP: in mã ra log của backend để test luồng.
        logger.warning("SMTP_HOST is empty - password reset code for %s: %s", to_email, code)
        return

    message = EmailMessage()
    message["Subject"] = "Your password reset code"
    message["From"] = settings.smtp_from or settings.smtp_username
    message["To"] = to_email
    message.set_content(
        "We received a request to reset the password of your AI Study Assistant account.\n\n"
        f"Your verification code (UUID):\n\n    {code}\n\n"
        f"It expires in {minutes} minutes and can only be used once.\n"
        "If you did not request this, you can ignore this email - your password stays unchanged."
    )
    try:
        _send(message)
    except Exception:  # noqa: BLE001 - không để lộ lỗi SMTP cho client
        logger.exception("Failed to send password reset email to %s", to_email)
