from celery import Celery

from app.config import settings

celery_app = Celery("ai_study_assistant", broker=settings.redis_url, backend=settings.redis_url)
celery_app.autodiscover_tasks(["app.worker"])
