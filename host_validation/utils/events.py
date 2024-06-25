from contextlib import contextmanager
from threading import Event
from threading import Thread
from typing import Iterator
from uuid import uuid4


def _set_event_after_time(event: Event, seconds: float) -> None:
    event.wait(seconds)
    event.set()


@contextmanager
def get_expiration_event(seconds: float) -> Iterator[Event]:
    event = Event()
    # Since we set `daemon=True`, we don't need to join the thread.
    Thread(target=_set_event_after_time, args=(event, seconds), name=f"timeout_thread_{uuid4()}", daemon=True).start()
    try:
        yield event

    finally:
        event.set()

