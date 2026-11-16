"""Chặn request upload QUÁ LỚN trước khi bị đọc.

Starlette phân tích multipart và ghi TOÀN BỘ body ra file tạm TRƯỚC khi hàm endpoint được gọi, nên kiểm tra
kích thước trong endpoint là quá muộn: một request 20 GB đã ngốn xong 20 GB ổ đĩa. Middleware này đứng ngoài
cùng và từ chối sớm — theo header ``Content-Length`` nếu có, còn nếu client gửi kiểu chunked (không có header)
thì đếm byte lúc nhận và cắt khi vượt ngưỡng.
"""
from __future__ import annotations

from starlette.responses import JSONResponse


class _TooLarge(Exception):
    pass


class UploadSizeLimitMiddleware:
    def __init__(self, app, path: str, max_body_bytes: int):
        self.app = app
        self.path = path.rstrip("/")
        self.max_body_bytes = max_body_bytes

    def _detail(self) -> str:
        return f"Request body exceeds the maximum upload size of {self.max_body_bytes // (1024 * 1024)}MB."

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path", "").rstrip("/") != self.path:
            await self.app(scope, receive, send)
            return

        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
            await JSONResponse({"detail": self._detail()}, status_code=413)(scope, receive, send)
            return

        received = 0
        started = False

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _TooLarge
            return message

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _TooLarge:
            if not started:
                await JSONResponse({"detail": self._detail()}, status_code=413)(scope, receive, send)
