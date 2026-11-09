from celery import Celery

from signature_project.settings import JOB_TIME_LIMIT_SECONDS, JOB_TTL_SECONDS, REDIS_URL

celery_app = Celery(
    "signature_project",
    broker=REDIS_URL,       # where jobs wait in line
    backend=REDIS_URL,      # where finished results are kept
    include=["signature_project.tasks"],   # the worker imports the tasks from here
)

celery_app.conf.update(
    # Only plain JSON travels through the queue (safer than Python pickles).
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Results disappear after a while, so Redis does not fill up.
    result_expires=JOB_TTL_SECONDS,
    # Lets us tell "waiting in line" apart from "a worker is busy with it".
    task_track_started=True,
    # Take a job off the line only AFTER it finished. If a worker dies halfway, the
    # job is not lost: it goes back in line once the "visibility timeout" below passes.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Our jobs are heavy, so each worker takes one at a time (fair sharing between workers).
    worker_prefetch_multiplier=1,
    task_time_limit=JOB_TIME_LIMIT_SECONDS,
    # If Redis is down, fail fast instead of hanging the request.
    broker_connection_retry_on_startup=True,
    # visibility_timeout must be longer than the longest job (the time limit above),
    # otherwise a slow job would be handed to a second worker while still running.
    broker_transport_options={"socket_connect_timeout": 3, "visibility_timeout": 300},
    task_publish_retry_policy={"max_retries": 1, "interval_start": 0, "interval_step": 0.2, "interval_max": 0.5},
)
