"""
GreenRoute AI — State Machine & Alert System
Event-driven state transitions based on real-time truck metrics.
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional

from .stream_engine import StreamEngine, TemporalWindow
from .transforms import CARBON_ALERT_THRESHOLD, SPEED_VARIANCE_THRESHOLD

# ─── State definitions ───────────────────────────────────────────────────────
STATES = {
    "EN_ROUTE": "Vehicle is actively traveling on its assigned route.",
    "IDLE": "Vehicle engine is running but speed is below 5 km/h.",
    "DELAYED": "Vehicle exhibits erratic speed patterns (high variance).",
    "EMISSION_ALERT": "Carbon emissions exceed regulatory threshold.",
    "OPTIMIZATION_REQUIRED": "Multiple flags active - comprehensive review needed.",
}

# State priority (higher = more critical)
STATE_PRIORITY = {
    "EN_ROUTE": 0,
    "IDLE": 1,
    "DELAYED": 2,
    "EMISSION_ALERT": 3,
    "OPTIMIZATION_REQUIRED": 4,
}


class TruckState:
    """Tracks state for a single truck."""

    def __init__(self, truck_id: int):
        self.truck_id = truck_id
        self.current_state = "EN_ROUTE"
        self.previous_state = "EN_ROUTE"
        self.state_since = time.time()
        self.flags: Dict[str, bool] = {
            "idle_flag": False,
            "instability_flag": False,
            "emission_violation": False,
            "route_deviation": False,
        }

    def update(self, metrics: dict) -> Optional[dict]:
        """Update state based on metrics. Returns alert dict if state changed."""
        old_state = self.current_state

        # Update flags
        self.flags["idle_flag"] = metrics.get("idle_flag", False)
        self.flags["instability_flag"] = metrics.get("instability_flag", False)
        self.flags["emission_violation"] = metrics.get("emission_violation", False)
        self.flags["route_deviation"] = metrics.get("route_deviation_flag", False)

        # Count active flags
        active_flags = sum(1 for v in self.flags.values() if v)

        # Determine new state (priority-based)
        if active_flags >= 2:
            new_state = "OPTIMIZATION_REQUIRED"
        elif self.flags["emission_violation"]:
            new_state = "EMISSION_ALERT"
        elif self.flags["instability_flag"]:
            new_state = "DELAYED"
        elif self.flags["idle_flag"]:
            new_state = "IDLE"
        else:
            new_state = "EN_ROUTE"

        # Transition
        if new_state != old_state:
            self.previous_state = old_state
            self.current_state = new_state
            self.state_since = time.time()

            return {
                "truck_id": self.truck_id,
                "previous_state": old_state,
                "new_state": new_state,
                "timestamp": metrics.get("timestamp", time.time()),
                "reason": self._get_reason(metrics),
                "carbon_emission": metrics.get("carbon_emission", 0),
                "avg_speed": metrics.get("avg_speed", 0),
                "avg_fuel_rate": metrics.get("avg_fuel_rate", 0),
                "speed_variance": metrics.get("speed_variance", 0),
                "idle_ratio": metrics.get("idle_ratio", 0),
                "active_flags": active_flags,
            }
        return None

    def _get_reason(self, metrics: dict) -> str:
        reasons = []
        if self.flags["emission_violation"]:
            carbon = metrics.get("carbon_emission", 0)
            reasons.append(
                f"Carbon emission {carbon:.2f} kg CO2 exceeds threshold of {CARBON_ALERT_THRESHOLD} kg"
            )
        if self.flags["instability_flag"]:
            variance = metrics.get("speed_variance", 0)
            reasons.append(
                f"Speed variance {variance:.1f} exceeds threshold of {SPEED_VARIANCE_THRESHOLD}"
            )
        if self.flags["idle_flag"]:
            idle_ratio = metrics.get("idle_ratio", 0)
            reasons.append(
                f"Idle ratio {idle_ratio:.1%} - vehicle predominantly stationary"
            )
        if self.flags["route_deviation"]:
            reasons.append("Route deviation detected")

        return "; ".join(reasons) if reasons else "Returned to normal operation"

    def to_dict(self) -> dict:
        return {
            "truck_id": self.truck_id,
            "state": self.current_state,
            "state_description": STATES[self.current_state],
            "previous_state": self.previous_state,
            "flags": dict(self.flags),
            "state_since": self.state_since,
        }


class FleetStateMachine:
    """Manages state for all trucks in the fleet."""

    def __init__(self):
        self.truck_states: Dict[int, TruckState] = {}
        self.alerts: List[dict] = []
        self._max_alerts = 500  # Ring buffer

    def get_or_create(self, truck_id: int) -> TruckState:
        if truck_id not in self.truck_states:
            self.truck_states[truck_id] = TruckState(truck_id)
        return self.truck_states[truck_id]

    def process_metrics(self, metrics: dict) -> Optional[dict]:
        """Process metrics for a truck and return alert if state changed."""
        truck_id = metrics["truck_id"]
        state = self.get_or_create(truck_id)
        alert = state.update(metrics)

        if alert:
            self.alerts.append(alert)
            # Trim old alerts
            if len(self.alerts) > self._max_alerts:
                self.alerts = self.alerts[-self._max_alerts:]

        return alert

    def get_all_states(self) -> List[dict]:
        return [s.to_dict() for s in self.truck_states.values()]

    def get_active_alerts(self) -> List[dict]:
        """Return alerts for trucks currently NOT in EN_ROUTE state."""
        active = []
        for state in self.truck_states.values():
            if state.current_state != "EN_ROUTE":
                latest_alert = None
                for alert in reversed(self.alerts):
                    if alert["truck_id"] == state.truck_id:
                        latest_alert = alert
                        break
                if latest_alert:
                    active.append(latest_alert)
                else:
                    # No alert yet but state is not EN_ROUTE
                    active.append({
                        "truck_id": state.truck_id,
                        "new_state": state.current_state,
                        "previous_state": state.previous_state,
                        "reason": "State initialized from metrics",
                        "timestamp": state.state_since,
                    })
        return active

    def get_recent_alerts(self, n: int = 20) -> List[dict]:
        return self.alerts[-n:]


# ─── Global fleet state machine instance ──────────────────────────────────────
fleet_state_machine = FleetStateMachine()


async def state_machine_subscriber(truck_id: int, window: TemporalWindow, engine: StreamEngine):
    """
    Stream engine subscriber that updates truck state machine
    after metrics are computed.
    """
    metrics = engine.metrics.get(truck_id)
    if metrics:
        alert = fleet_state_machine.process_metrics(metrics)
        if alert:
            state_marker = {
                "EN_ROUTE": "[OK]",
                "IDLE": "[IDLE]",
                "DELAYED": "[DELAY]",
                "EMISSION_ALERT": "[EMISSION]",
                "OPTIMIZATION_REQUIRED": "[OPTIMIZE]",
            }
            marker = state_marker.get(alert["new_state"], "[?]")
            print(
                f"  {marker} ALERT: Truck {truck_id} -> {alert['new_state']} | {alert['reason']}"
            )
