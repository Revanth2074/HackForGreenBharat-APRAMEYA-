"""
GreenRoute AI — Core Streaming Engine
Provides async streaming infrastructure: event bus, temporal windows, incremental state.
"""

import asyncio
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional


class StreamEvent:
    """Single telemetry event from a truck."""
    __slots__ = ("truck_id", "data", "timestamp")

    def __init__(self, truck_id: int, data: dict, timestamp: float):
        self.truck_id = truck_id
        self.data = data
        self.timestamp = timestamp


class TemporalWindow:
    """
    Sliding temporal window that maintains events within a configurable duration.
    Supports incremental computation — only recomputes when the window changes.
    """

    def __init__(self, window_duration_seconds: float = 300.0):
        self.window_duration = window_duration_seconds
        self._events: List[StreamEvent] = []
        self._dirty = True

    def add(self, event: StreamEvent):
        self._events.append(event)
        self._dirty = True

    def evict_old(self, current_time: float):
        cutoff = current_time - self.window_duration
        before = len(self._events)
        self._events = [e for e in self._events if e.timestamp >= cutoff]
        if len(self._events) != before:
            self._dirty = True

    @property
    def events(self) -> List[StreamEvent]:
        return self._events

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_clean(self):
        self._dirty = False

    def __len__(self):
        return len(self._events)


class StreamEngine:
    """
    Core streaming engine that orchestrates:
    - Ingestion from async generators
    - Per-truck temporal windows
    - Subscriber notifications (pub/sub)
    - Incremental state management
    """

    def __init__(self, window_duration: float = 300.0):
        self.window_duration = window_duration
        # Per-truck temporal windows
        self.windows: Dict[int, TemporalWindow] = defaultdict(
            lambda: TemporalWindow(self.window_duration)
        )
        # Subscribers receive (truck_id, window) on each update
        self._subscribers: List[Callable] = []
        # Latest state per truck (for API queries)
        self.latest_events: Dict[int, StreamEvent] = {}
        # Computed metrics (updated incrementally)
        self.metrics: Dict[int, dict] = {}
        # Running flag
        self._running = False
        # Event counter
        self.event_count = 0

    def subscribe(self, callback: Callable):
        """Register a callback: async def callback(truck_id, window, engine)"""
        self._subscribers.append(callback)

    async def ingest(self, generator):
        """Consume events from an async generator and route to windows."""
        self._running = True
        async for event in generator:
            if not self._running:
                break

            self.event_count += 1
            truck_id = event.truck_id
            
            # Update latest event
            self.latest_events[truck_id] = event
            
            # Add to temporal window
            window = self.windows[truck_id]
            window.add(event)
            window.evict_old(event.timestamp)

            # Notify all subscribers (incremental)
            for subscriber in self._subscribers:
                try:
                    await subscriber(truck_id, window, self)
                except Exception as e:
                    print(f"[StreamEngine] Subscriber error: {e}")

    def stop(self):
        self._running = False

    def get_window(self, truck_id: int) -> Optional[TemporalWindow]:
        if truck_id in self.windows:
            return self.windows[truck_id]
        return None

    def get_all_metrics(self) -> Dict[int, dict]:
        return dict(self.metrics)

    def get_truck_metrics(self, truck_id: int) -> Optional[dict]:
        return self.metrics.get(truck_id)
