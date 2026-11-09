import os

import pytest
import redis

from signature_project.job_store import JobStore
from signature_project.settings import REDIS_URL

JOB_ID = "a" * 32


def test_upload_round_trip(job_store):
    job_store.save_upload(JOB_ID, b"file bytes")

    assert job_store.load_upload(JOB_ID) == b"file bytes"


def test_missing_upload_gives_none(job_store):
    assert job_store.load_upload(JOB_ID) is None


def test_upload_can_be_deleted(job_store):
    job_store.save_upload(JOB_ID, b"file bytes")

    job_store.delete_upload(JOB_ID)

    assert job_store.load_upload(JOB_ID) is None


def test_job_record_round_trip(job_store):
    job_store.register(JOB_ID, "detect", "contract.pdf")

    meta = job_store.get_meta(JOB_ID)

    assert meta["kind"] == "detect"
    assert meta["filename"] == "contract.pdf"
    assert "created_at" in meta


def test_unknown_job_gives_none(job_store):
    assert job_store.get_meta(JOB_ID) is None


def test_everything_expires_by_itself(job_store):
    job_store.register(JOB_ID, "detect", "contract.pdf")
    job_store.save_upload(JOB_ID, b"file bytes")

    assert 0 < job_store._client.ttl(f"job:{JOB_ID}:meta") <= 60
    assert 0 < job_store._client.ttl(f"job:{JOB_ID}:upload") <= 60


def _real_redis_available() -> bool:
    try:
        redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.5).ping()
        return True
    except redis.RedisError:
        return False


@pytest.mark.skipif(not _real_redis_available(), reason=f"no Redis server reachable at {REDIS_URL}")
def test_real_redis_round_trip():
    """Only runs when a real Redis is running (for example the Docker one)."""
    client = redis.Redis.from_url(REDIS_URL)
    store = JobStore(client, ttl_seconds=30)
    job_id = os.urandom(16).hex()
    try:
        store.register(job_id, "detect", "x.png")
        store.save_upload(job_id, b"\x00\x01\x02")
        assert store.get_meta(job_id)["filename"] == "x.png"
        assert store.load_upload(job_id) == b"\x00\x01\x02"
    finally:
        client.delete(f"job:{job_id}:meta", f"job:{job_id}:upload")
