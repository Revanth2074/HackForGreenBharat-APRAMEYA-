"""
GreenRoute AI — Eco-Score Ranking Engine
Computes and ranks trucks by eco-efficiency in real time.
"""

from typing import Dict, List, Optional

from .stream_engine import StreamEngine


def compute_eco_score(metrics: dict) -> float:
    """
    Compute eco-score for a truck.
    
    eco_score = avg_speed / avg_fuel_rate - (carbon_emission / 10)
    
    Higher score = more eco-efficient.
    """
    avg_speed = metrics.get("avg_speed", 0)
    avg_fuel_rate = metrics.get("avg_fuel_rate", 1)  # Avoid division by zero
    carbon = metrics.get("carbon_emission", 0)

    if avg_fuel_rate <= 0:
        avg_fuel_rate = 1.0

    base_score = avg_speed / avg_fuel_rate
    carbon_penalty = carbon / 10.0

    # Additional penalties
    idle_penalty = 1.0 if metrics.get("idle_flag", False) else 0.0
    instability_penalty = 0.5 if metrics.get("instability_flag", False) else 0.0

    eco_score = base_score - carbon_penalty - idle_penalty - instability_penalty
    return round(eco_score, 3)


def get_fleet_ranking(engine: StreamEngine) -> List[dict]:
    """
    Compute eco-scores for all trucks and return ranked list.
    """
    rankings = []

    for truck_id, metrics in engine.metrics.items():
        eco_score = compute_eco_score(metrics)
        
        # Determine rating
        if eco_score > 8.0:
            rating = "Excellent"
            rating_emoji = "[*]"
        elif eco_score > 5.0:
            rating = "Good"
            rating_emoji = "[+]"
        elif eco_score > 2.0:
            rating = "Fair"
            rating_emoji = "[!]"
        else:
            rating = "Poor"
            rating_emoji = "[X]"

        rankings.append({
            "truck_id": truck_id,
            "eco_score": eco_score,
            "rating": rating,
            "rating_emoji": rating_emoji,
            "avg_speed": metrics.get("avg_speed", 0),
            "avg_fuel_rate": metrics.get("avg_fuel_rate", 0),
            "carbon_emission": metrics.get("carbon_emission", 0),
            "distance_km": metrics.get("distance_km", 0),
            "idle_flag": metrics.get("idle_flag", False),
            "instability_flag": metrics.get("instability_flag", False),
            "emission_violation": metrics.get("emission_violation", False),
            "route_id": metrics.get("route_id", ""),
        })

    # Sort by eco_score descending (best first)
    rankings.sort(key=lambda x: x["eco_score"], reverse=True)

    # Add rank
    for i, r in enumerate(rankings):
        r["rank"] = i + 1

    return rankings


def get_top_eco_trucks(engine: StreamEngine, n: int = 3) -> List[dict]:
    """Get top N eco-efficient trucks."""
    rankings = get_fleet_ranking(engine)
    return rankings[:n]


def get_worst_emitters(engine: StreamEngine, n: int = 3) -> List[dict]:
    """Get worst N emitting trucks."""
    rankings = get_fleet_ranking(engine)
    return rankings[-n:] if len(rankings) >= n else rankings
