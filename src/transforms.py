"""
GreenRoute AI -- Streaming Transformations v2
Rolling window computations: carbon, speed stats, idle detection, distance, ETA, route deviation.
All transformations are incremental -- triggered on each new event via the stream engine.
"""

import numpy as np
from typing import Dict, Optional

from .stream_engine import StreamEngine, TemporalWindow

# --- Constants ---
CO2_FACTOR = 2.68          # kg CO2 per liter of diesel
IDLE_SPEED_THRESHOLD = 5.0  # km/h
SPEED_VARIANCE_THRESHOLD = 150.0  # (km/h)^2
CARBON_ALERT_THRESHOLD = 15.0     # kg CO2 per window
ROUTE_DEVIATION_THRESHOLD = 0.5   # km off-route


def compute_window_metrics(window: TemporalWindow) -> dict:
    """
    Compute all streaming metrics for a truck's temporal window.
    Returns a dict with all computed fields.
    """
    events = window.events
    if not events:
        return _empty_metrics()

    speeds = [e.data["speed"] for e in events]
    fuel_rates = [e.data["fuel_rate"] for e in events]
    timestamps = [e.timestamp for e in events]

    # --- Time span ---
    time_span_seconds = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
    time_span_hours = time_span_seconds / 3600.0

    # --- Averages ---
    avg_speed = float(np.mean(speeds))
    avg_fuel_rate = float(np.mean(fuel_rates))
    max_speed = float(np.max(speeds))
    min_speed = float(np.min(speeds))

    # --- Total fuel consumed in window ---
    total_fuel = avg_fuel_rate * time_span_hours  # liters

    # --- Carbon emission ---
    carbon_emission = total_fuel * CO2_FACTOR  # kg CO2

    # --- Distance approximation ---
    distance_km = 0.0
    for i in range(1, len(events)):
        dt_hours = (events[i].timestamp - events[i - 1].timestamp) / 3600.0
        avg_spd = (events[i].data["speed"] + events[i - 1].data["speed"]) / 2.0
        distance_km += avg_spd * dt_hours

    # --- Idle detection ---
    idle_count = sum(1 for s in speeds if s < IDLE_SPEED_THRESHOLD)
    idle_ratio = idle_count / len(speeds) if speeds else 0
    idle_flag = idle_ratio > 0.5

    # --- Speed variance (instability) ---
    speed_variance = float(np.var(speeds)) if len(speeds) > 1 else 0.0
    instability_flag = speed_variance > SPEED_VARIANCE_THRESHOLD

    # --- Emission violation ---
    emission_violation = carbon_emission > CARBON_ALERT_THRESHOLD

    # --- Latest event data ---
    latest = events[-1].data

    # --- ETA from latest event ---
    eta_minutes = latest.get("eta_minutes", 0)
    remaining_km = latest.get("remaining_km", 0)
    cumulative_distance_km = latest.get("cumulative_distance_km", 0)

    # --- Route deviation ---
    route_deviation_km = latest.get("route_deviation_km", 0)
    route_deviation_flag = route_deviation_km > ROUTE_DEVIATION_THRESHOLD

    # --- Fuel efficiency (km per liter) ---
    fuel_efficiency = distance_km / total_fuel if total_fuel > 0 else 0

    return {
        "truck_id": latest["truck_id"],
        "driver_name": latest.get("driver_name", ""),
        "avg_speed": round(avg_speed, 2),
        "max_speed": round(max_speed, 2),
        "min_speed": round(min_speed, 2),
        "avg_fuel_rate": round(avg_fuel_rate, 2),
        "total_fuel": round(total_fuel, 4),
        "carbon_emission": round(carbon_emission, 4),
        "distance_km": round(distance_km, 4),
        "speed_variance": round(speed_variance, 2),
        "idle_ratio": round(idle_ratio, 3),
        "idle_flag": idle_flag,
        "instability_flag": instability_flag,
        "emission_violation": emission_violation,
        "route_deviation_km": round(route_deviation_km, 3),
        "route_deviation_flag": route_deviation_flag,
        "eta_minutes": round(eta_minutes, 1),
        "remaining_km": round(remaining_km, 2),
        "cumulative_distance_km": round(cumulative_distance_km, 3),
        "fuel_efficiency_km_per_l": round(fuel_efficiency, 3),
        "window_events": len(events),
        "window_span_seconds": round(time_span_seconds, 1),
        "latitude": latest["latitude"],
        "longitude": latest["longitude"],
        "cargo_weight": latest["cargo_weight"],
        "cargo_type": latest.get("cargo_type", ""),
        "route_id": latest["route_id"],
        "route_name": latest.get("route_name", ""),
        "last_speed": latest["speed"],
        "last_fuel_rate": latest["fuel_rate"],
        "timestamp": latest["timestamp"],
        "behavior": latest.get("behavior", ""),
        "trip_elapsed_minutes": latest.get("trip_elapsed_minutes", 0),
    }


def _empty_metrics() -> dict:
    return {
        "truck_id": 0, "driver_name": "",
        "avg_speed": 0, "max_speed": 0, "min_speed": 0,
        "avg_fuel_rate": 0, "total_fuel": 0, "carbon_emission": 0,
        "distance_km": 0, "speed_variance": 0, "idle_ratio": 0,
        "idle_flag": False, "instability_flag": False,
        "emission_violation": False,
        "route_deviation_km": 0, "route_deviation_flag": False,
        "eta_minutes": 0, "remaining_km": 0, "cumulative_distance_km": 0,
        "fuel_efficiency_km_per_l": 0,
        "window_events": 0, "window_span_seconds": 0,
        "latitude": 0, "longitude": 0,
        "cargo_weight": 0, "cargo_type": "", "route_id": "", "route_name": "",
        "last_speed": 0, "last_fuel_rate": 0, "timestamp": 0,
        "behavior": "", "trip_elapsed_minutes": 0,
    }


async def transform_subscriber(truck_id: int, window: TemporalWindow, engine: StreamEngine):
    """
    Stream engine subscriber that recomputes metrics incrementally
    whenever a truck's window is updated.
    """
    metrics = compute_window_metrics(window)
    engine.metrics[truck_id] = metrics
    window.mark_clean()
