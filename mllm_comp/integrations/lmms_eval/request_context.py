"""Request identity only; never carries answers or generation text."""
from contextlib import contextmanager
from contextvars import ContextVar

CURRENT_SAMPLE_ID = ContextVar('mllm_comp_eval_sample_id', default=None)


@contextmanager
def sample_request(sample_id):
    token = CURRENT_SAMPLE_ID.set(str(sample_id))
    try:
        yield
    finally:
        CURRENT_SAMPLE_ID.reset(token)
