"""
GreenRoute AI — Live Fleet Stream Simulator (Ingestion) v2
Simulates 50 trucks across 10 Indian routes with diverse telemetry patterns.
"""

import asyncio
import math
import random
import time
from typing import AsyncGenerator

import numpy as np

from .stream_engine import StreamEvent

# ─── Route definitions (real Indian highway corridors) ────────────────────────
ROUTES = {
    "R01": {"start": (28.6139, 77.2090), "end": (26.9124, 75.7873), "dist_km": 280, "name": "Delhi - Jaipur (NH48)"},
    "R02": {"start": (28.6139, 77.2090), "end": (27.1767, 78.0081), "dist_km": 230, "name": "Delhi - Agra (Yamuna Exp)"},
    "R03": {"start": (19.0760, 72.8777), "end": (18.5204, 73.8567), "dist_km": 150, "name": "Mumbai - Pune (Exp)"},
    "R04": {"start": (12.9716, 77.5946), "end": (13.0827, 80.2707), "dist_km": 350, "name": "Bangalore - Chennai (NH48)"},
    "R05": {"start": (22.5726, 88.3639), "end": (25.6093, 85.1376), "dist_km": 580, "name": "Kolkata - Patna (NH19)"},
    "R06": {"start": (21.1702, 72.8311), "end": (23.0225, 72.5714), "dist_km": 270, "name": "Surat - Ahmedabad (NH48)"},
    "R07": {"start": (26.8467, 80.9462), "end": (25.4358, 81.8463), "dist_km": 210, "name": "Lucknow - Prayagraj (NH30)"},
    "R08": {"start": (17.3850, 78.4867), "end": (15.4909, 78.4867), "dist_km": 220, "name": "Hyderabad - Kurnool (NH44)"},
    "R09": {"start": (11.0168, 76.9558), "end": (9.9312,  76.2673), "dist_km": 200, "name": "Coimbatore - Kochi (NH544)"},
    "R10": {"start": (30.7333, 76.7794), "end": (28.6139, 77.2090), "dist_km": 250, "name": "Chandigarh - Delhi (NH44)"},
}

# ─── Cargo types ──────────────────────────────────────────────────────────────
CARGO_TYPES = [
    {"type": "Electronics",    "weight_range": (2000, 6000),  "fragile": True},
    {"type": "FMCG",           "weight_range": (5000, 12000), "fragile": False},
    {"type": "Fuel Tanker",    "weight_range": (15000, 22000),"fragile": False},
    {"type": "Perishables",    "weight_range": (4000, 10000), "fragile": True},
    {"type": "Construction",   "weight_range": (12000, 20000),"fragile": False},
    {"type": "Pharmaceuticals","weight_range": (1000, 4000),  "fragile": True},
    {"type": "Textiles",       "weight_range": (3000, 8000),  "fragile": False},
    {"type": "Automobiles",    "weight_range": (8000, 16000), "fragile": True},
]

# ─── Behavior types for anomaly injection ─────────────────────────────────────
BEHAVIORS = {
    "normal":           "Steady driving within optimal range",
    "speed_anomaly":    "Erratic speed fluctuations (aggressive driving)",
    "high_fuel":        "Excessive fuel consumption (overloaded/engine issue)",
    "idle":             "Extended idling (stuck at depot/traffic)",
    "route_deviation":  "Deviating from planned route corridor",
    "speeding":         "Consistently exceeding speed limits",
    "eco_driver":       "Exemplary eco-driving behavior",
    "night_driver":     "Night shift - lower speeds, less traffic",
    "heavy_traffic":    "Caught in urban congestion",
    "maintenance_due":  "Degraded engine - higher fuel, erratic performance",
}

# ─── Truck configurations (50 trucks) ────────────────────────────────────────
def _generate_truck_configs():
    configs = {}
    np.random.seed(42)  # Reproducible

    for tid in range(1, 51):
        # Assign routes round-robin with some variation
        route_id = f"R{((tid - 1) % 10) + 1:02d}"
        route = ROUTES[route_id]
        cargo = random.choice(CARGO_TYPES)
        cargo_weight = random.randint(*cargo["weight_range"])

        # Assign behaviors: most normal, specific trucks get anomalies
        if tid <= 25:
            behavior = "normal"
        elif tid <= 28:
            behavior = "eco_driver"
        elif tid <= 31:
            behavior = "speed_anomaly"
        elif tid <= 34:
            behavior = "high_fuel"
        elif tid <= 37:
            behavior = "idle"
        elif tid <= 39:
            behavior = "route_deviation"
        elif tid <= 41:
            behavior = "speeding"
        elif tid <= 43:
            behavior = "night_driver"
        elif tid <= 46:
            behavior = "heavy_traffic"
        else:
            behavior = "maintenance_due"

        # Base parameters vary by truck
        base_speed = np.random.normal(58, 5)
        base_fuel = 12 + (cargo_weight / 2000)  # Heavier cargo = more fuel
        if cargo_weight > 15000:
            base_fuel += 5

        configs[tid] = {
            "route": route_id,
            "route_name": route["name"],
            "route_dist_km": route["dist_km"],
            "base_speed": round(float(base_speed), 1),
            "base_fuel": round(float(base_fuel), 1),
            "cargo_weight": cargo_weight,
            "cargo_type": cargo["type"],
            "behavior": behavior,
            "driver_name": f"Driver-{tid:03d}",
        }
    return configs


TRUCK_CONFIGS = _generate_truck_configs()


class TruckSimulator:
    """Simulates a single truck's telemetry stream."""

    def __init__(self, truck_id: int, config: dict):
        self.truck_id = truck_id
        self.config = config
        self.route = ROUTES[config["route"]]
        self.progress = random.uniform(0, 0.2)
        self.lat = self.route["start"][0]
        self.lng = self.route["start"][1]
        self.tick_count = 0
        self.trip_start_time = None
        self.cumulative_distance = 0.0

    def generate_event(self, sim_time: float) -> StreamEvent:
        self.tick_count += 1
        if self.trip_start_time is None:
            self.trip_start_time = sim_time

        behavior = self.config["behavior"]
        base_speed = self.config["base_speed"]
        base_fuel = self.config["base_fuel"]

        # ─── Speed generation by behavior ─────────────────────────────
        speed = self._generate_speed(behavior, base_speed)

        # ─── Fuel rate generation ─────────────────────────────────────
        fuel_rate = self._generate_fuel_rate(behavior, base_fuel, speed, base_speed)

        # ─── Update position ──────────────────────────────────────────
        delta_time_hrs = 0.4 / 3600
        distance_km = speed * delta_time_hrs
        self.cumulative_distance += distance_km

        if behavior == "route_deviation":
            deviation = math.sin(self.tick_count * 0.25) * 0.012
            self.progress += distance_km / max(self.config["route_dist_km"], 1)
            p = self.progress % 1.0
            self.lat = self.route["start"][0] + (self.route["end"][0] - self.route["start"][0]) * p + deviation
            self.lng = self.route["start"][1] + (self.route["end"][1] - self.route["start"][1]) * p + deviation * 0.6
        else:
            self.progress += distance_km / max(self.config["route_dist_km"], 1)
            p = self.progress % 1.0
            self.lat = self.route["start"][0] + (self.route["end"][0] - self.route["start"][0]) * p
            self.lng = self.route["start"][1] + (self.route["end"][1] - self.route["start"][1]) * p

        # ─── ETA Estimation ───────────────────────────────────────────
        remaining_km = max(0, self.config["route_dist_km"] - self.cumulative_distance)
        eta_hours = (remaining_km / max(speed, 1)) if speed > 1 else float('inf')
        eta_minutes = round(eta_hours * 60, 1)
        if eta_minutes > 9999:
            eta_minutes = 9999.0

        # ─── Route deviation distance ────────────────────────────────
        expected_lat = self.route["start"][0] + (self.route["end"][0] - self.route["start"][0]) * (self.progress % 1.0)
        expected_lng = self.route["start"][1] + (self.route["end"][1] - self.route["start"][1]) * (self.progress % 1.0)
        deviation_km = math.sqrt((self.lat - expected_lat)**2 + (self.lng - expected_lng)**2) * 111

        data = {
            "truck_id": self.truck_id,
            "driver_name": self.config["driver_name"],
            "latitude": round(self.lat, 6),
            "longitude": round(self.lng, 6),
            "speed": round(float(speed), 2),
            "fuel_rate": round(float(fuel_rate), 2),
            "cargo_weight": self.config["cargo_weight"],
            "cargo_type": self.config["cargo_type"],
            "route_id": self.config["route"],
            "route_name": self.config["route_name"],
            "behavior": behavior,
            "timestamp": sim_time,
            "cumulative_distance_km": round(self.cumulative_distance, 3),
            "remaining_km": round(remaining_km, 2),
            "eta_minutes": eta_minutes,
            "route_deviation_km": round(deviation_km, 3),
            "trip_elapsed_minutes": round((sim_time - self.trip_start_time) / 60, 1),
        }

        return StreamEvent(truck_id=self.truck_id, data=data, timestamp=sim_time)

    def _generate_speed(self, behavior: str, base_speed: float) -> float:
        if behavior == "normal":
            speed = np.random.normal(base_speed, 3.0)
            speed = max(25, min(80, speed))

        elif behavior == "eco_driver":
            speed = np.random.normal(55, 2.0)  # Very steady in optimal range
            speed = max(48, min(65, speed))

        elif behavior == "speed_anomaly":
            if self.tick_count % 5 < 2:
                speed = np.random.normal(95, 12.0)
            else:
                speed = np.random.normal(20, 8.0)
            speed = max(5, min(130, speed))

        elif behavior == "high_fuel":
            speed = np.random.normal(base_speed, 4.0)
            speed = max(30, min(75, speed))

        elif behavior == "idle":
            if self.tick_count % 25 < 18:
                speed = np.random.normal(2, 1.5)
                speed = max(0, min(5, speed))
            else:
                speed = np.random.normal(35, 5.0)
                speed = max(15, min(55, speed))

        elif behavior == "route_deviation":
            speed = np.random.normal(base_speed, 5.0)
            speed = max(20, min(80, speed))

        elif behavior == "speeding":
            speed = np.random.normal(95, 8.0)
            speed = max(75, min(130, speed))

        elif behavior == "night_driver":
            speed = np.random.normal(45, 3.0)
            speed = max(30, min(60, speed))

        elif behavior == "heavy_traffic":
            cycle = math.sin(self.tick_count * 0.15) * 0.5 + 0.5
            if cycle > 0.6:
                speed = np.random.normal(15, 5.0)
            else:
                speed = np.random.normal(35, 5.0)
            speed = max(0, min(50, speed))

        elif behavior == "maintenance_due":
            # Engine misfires cause erratic performance
            if self.tick_count % 8 == 0:
                speed = np.random.normal(15, 10.0)
            else:
                speed = np.random.normal(base_speed * 0.8, 6.0)
            speed = max(5, min(70, speed))

        else:
            speed = base_speed

        return float(speed)

    def _generate_fuel_rate(self, behavior: str, base_fuel: float, speed: float, base_speed: float) -> float:
        if behavior == "high_fuel":
            fuel_rate = np.random.normal(base_fuel * 2.2, 3.0)
            fuel_rate = max(25, min(50, fuel_rate))

        elif behavior == "eco_driver":
            fuel_rate = np.random.normal(base_fuel * 0.8, 0.8)
            fuel_rate = max(8, min(18, fuel_rate))

        elif behavior == "idle" and speed < 5:
            fuel_rate = np.random.normal(2.5, 0.5)
            fuel_rate = max(1.5, min(4.0, fuel_rate))

        elif behavior == "speeding":
            # High speed = high fuel
            fuel_rate = np.random.normal(base_fuel * 1.5, 2.0)
            fuel_rate = max(18, min(40, fuel_rate))

        elif behavior == "maintenance_due":
            # Engine inefficiency adds 30%
            fuel_rate = np.random.normal(base_fuel * 1.35, 2.5)
            fuel_rate = max(12, min(35, fuel_rate))

        elif behavior == "heavy_traffic":
            if speed < 10:
                fuel_rate = np.random.normal(3.5, 1.0)
            else:
                fuel_rate = np.random.normal(base_fuel * 1.1, 1.5)
            fuel_rate = max(2, min(25, fuel_rate))

        else:
            fuel_rate = base_fuel + (speed - base_speed) * 0.12
            fuel_rate = np.random.normal(fuel_rate, 1.0)
            fuel_rate = max(8, min(30, fuel_rate))

        return float(fuel_rate)


async def fleet_stream(
    num_trucks: int = 50,
    interval: float = 0.4,
    time_acceleration: float = 30.0,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Infinite async generator that yields truck telemetry events.
    50 trucks across 10 Indian highway routes.
    """
    actual_trucks = min(num_trucks, 50)
    simulators = {
        tid: TruckSimulator(tid, TRUCK_CONFIGS[tid])
        for tid in range(1, actual_trucks + 1)
    }

    start_real = time.time()
    start_sim = time.time()

    while True:
        for truck_id in range(1, actual_trucks + 1):
            elapsed_real = time.time() - start_real
            sim_time = start_sim + elapsed_real * time_acceleration
            event = simulators[truck_id].generate_event(sim_time)
            yield event

        await asyncio.sleep(interval)
