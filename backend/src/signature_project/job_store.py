import json
from datetime import datetime, timezone
from functools import lru_cache

import redis

from signature_project.settings import JOB_TTL_SECONDS, REDIS_URL


# A verification job has two uploads, so each one gets its own name.
REFERENCE_SLOT = "reference"
CANDIDATE_SLOT = "candidate"


class JobStore:
    """Keeps each job's record and uploaded file in Redis.

    Why Redis and not a folder on disk? The web server and the workers may run as
    separate containers that do not share a disk, but they can all reach Redis.
    Everything expires by itself after `ttl_seconds`.
    """

    def __init__(self, client, ttl_seconds: int):
        self._client = client
        self._ttl = ttl_seconds

    @staticmethod
    def _meta_key(job_id: str) -> str:
        return f"job:{job_id}:meta"

    @staticmethod
    def _upload_key(job_id: str, name: str | None = None) -> str:
        # No name = the single upload of a detection job (the original key).
        base = f"job:{job_id}:upload"
        return base if name is None else f"{base}:{name}"

    def register(self, job_id: str, kind: str, filename: str | None) -> None:
        meta = {
            "kind": kind,
            "filename": filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._client.set(self._meta_key(job_id), json.dumps(meta), ex=self._ttl)

    def get_meta(self, job_id: str) -> dict | None:
        raw = self._client.get(self._meta_key(job_id))
        return json.loads(raw) if raw is not None else None

    def save_upload(self, job_id: str, contents: bytes, name: str | None = None) -> None:
        self._client.set(self._upload_key(job_id, name), contents, ex=self._ttl)

    def load_upload(self, job_id: str, name: str | None = None) -> bytes | None:
        return self._client.get(self._upload_key(job_id, name))

    def delete_upload(self, job_id: str, name: str | None = None) -> None:
        self._client.delete(self._upload_key(job_id, name))


@lru_cache
def _redis_client() -> redis.Redis:
    # One shared connection pool for the whole process.
    return redis.Redis.from_url(REDIS_URL, socket_connect_timeout=3, socket_timeout=10)


def get_job_store() -> JobStore:
    """FastAPI dependency (and used by the worker): the store backed by the real Redis."""
    return JobStore(_redis_client(), JOB_TTL_SECONDS)
