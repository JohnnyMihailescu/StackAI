"""Async Reader-Writer Lock implementation.

This module provides an async-compatible RWLock that allows:
- Multiple concurrent readers (shared access)
- Single writer with exclusive access (blocks all readers and other writers)

Design choices:
- Writer preference: When a writer is waiting, new readers queue behind it
  to prevent writer starvation
- Async context managers: Clean syntax with `async with lock.read():` and
  `async with lock.write():`
- No lock upgrades: Cannot go from read -> write without releasing first
  (prevents deadlocks)
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class AsyncRWLock:
    """Async Reader-Writer Lock.

    Allows multiple concurrent readers OR a single exclusive writer.
    Uses writer preference to prevent writer starvation.

    Usage:
        lock = AsyncRWLock()

        # For read operations
        async with lock.read():
            data = storage[key]

        # For write operations
        async with lock.write():
            storage[key] = value
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._readers = 0
        self._writer_waiting = False
        self._writer_active = False
        self._read_ready = asyncio.Condition(self._lock)
        self._write_ready = asyncio.Condition(self._lock)

    @asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        """Acquire read lock. Multiple readers can hold this simultaneously."""
        await self._acquire_read()
        try:
            yield
        finally:
            await self._release_read()

    @asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        """Acquire write lock. Exclusive access - blocks all other operations."""
        await self._acquire_write()
        try:
            yield
        finally:
            await self._release_write()

    async def _acquire_read(self) -> None:
        """Acquire read access."""
        async with self._lock:
            # Wait if a writer is active or waiting (writer preference)
            while self._writer_active or self._writer_waiting:
                await self._read_ready.wait()
            self._readers += 1

    async def _release_read(self) -> None:
        """Release read access."""
        async with self._lock:
            self._readers -= 1
            if self._readers == 0:
                # Last reader out, notify waiting writers
                self._write_ready.notify()

    async def _acquire_write(self) -> None:
        """Acquire write access."""
        async with self._lock:
            # Signal that a writer is waiting (blocks new readers)
            self._writer_waiting = True
            try:
                # Wait until no readers and no active writer
                while self._readers > 0 or self._writer_active:
                    await self._write_ready.wait()
                self._writer_active = True
            finally:
                self._writer_waiting = False

    async def _release_write(self) -> None:
        """Release write access."""
        async with self._lock:
            self._writer_active = False
            # Notify all waiting readers and one waiting writer
            self._read_ready.notify_all()
            self._write_ready.notify()

    @property
    def readers(self) -> int:
        """Current number of active readers (for debugging/monitoring)."""
        return self._readers

    @property
    def writer_active(self) -> bool:
        """Whether a writer currently holds the lock (for debugging/monitoring)."""
        return self._writer_active
