"""
GreenRoute AI — Optimization Engine
Rule-based corrective recommendations for trucks in alert states.
"""

from typing import Dict, List, Optional


def generate_recommendations(metrics: dict, state: str) -> List[str]:
    """
    Generate rule-based optimization recommendations based on truck metrics and state.
    """
    recommendations = []
    truck_id = metrics.get("truck_id", "?")
    avg_speed = metrics.get("avg_speed", 0)
    avg_fuel_rate = metrics.get("avg_fuel_rate", 0)
    carbon = metrics.get("carbon_emission", 0)
    speed_variance = metrics.get("speed_variance", 0)
    idle_flag = metrics.get("idle_flag", False)
    idle_ratio = metrics.get("idle_ratio", 0)

    # ─── Emission-based recommendations ───────────────────────────
    if state in ("EMISSION_ALERT", "OPTIMIZATION_REQUIRED"):
        recommendations.append(
            f"[CRITICAL] Truck {truck_id} carbon emission is {carbon:.2f} kg CO2. "
            f"Exceeds 15 kg threshold. Immediate action required."
        )
        
        if avg_fuel_rate > 25:
            recommendations.append(
                "[FUEL] Reduce fuel consumption: Current fuel rate is "
                f"{avg_fuel_rate:.1f} L/hr (expected < 20 L/hr). "
                "Check for engine issues, overloading, or tire pressure."
            )
        
        if avg_speed > 70:
            recommendations.append(
                f"[SPEED] Reduce speed: Current avg speed {avg_speed:.1f} km/h exceeds "
                "optimal range (50–70 km/h). High speed increases aerodynamic drag "
                "and fuel consumption by 20–40%."
            )
        
        recommendations.append(
            "[ROUTE] Consider route optimization: Evaluate alternative routes with "
            "lower traffic, fewer stops, and less elevation change."
        )

    # ─── Speed instability recommendations ────────────────────────
    if state in ("DELAYED", "OPTIMIZATION_REQUIRED") and speed_variance > 150:
        recommendations.append(
            f"[VARIANCE] Speed instability detected: Variance = {speed_variance:.1f}. "
            "Maintain steady speed using cruise control. Anticipate traffic flow "
            "to reduce sudden acceleration/deceleration cycles."
        )
        recommendations.append(
            "[BRAKE] Reduce hard braking: Each hard brake wastes 0.05-0.15 liters of "
            "diesel. Use engine braking and increase following distance."
        )

    # ─── Idle recommendations ─────────────────────────────────────
    if state in ("IDLE", "OPTIMIZATION_REQUIRED") and idle_flag:
        recommendations.append(
            f"[IDLE] Excessive idling detected: Idle ratio = {idle_ratio:.1%}. "
            "An idling truck consumes 1.5–3.0 L/hr producing 4–8 kg CO₂/hr. "
            "Turn off engine during stops exceeding 2 minutes."
        )
        recommendations.append(
            "[LOCATION] Check if truck is stuck at loading dock, traffic, or unauthorized stop. "
            "Coordinate with dispatcher to minimize wait times."
        )

    # ─── General optimization ─────────────────────────────────────
    if avg_speed < 40 and not idle_flag:
        recommendations.append(
            f"[SLOW] Low average speed ({avg_speed:.1f} km/h) with engine running. "
            "Low gear operation increases fuel consumption by ~35%. "
            "Consider re-routing to avoid congested areas."
        )

    if not recommendations:
        recommendations.append(
            f"[OK] Truck {truck_id} is operating within normal parameters. "
            "Continue current driving pattern."
        )

    return recommendations


def get_fleet_recommendations(metrics_dict: Dict[int, dict], states_dict: Dict[int, str]) -> Dict[int, List[str]]:
    """
    Generate recommendations for all trucks.
    
    Args:
        metrics_dict: {truck_id: metrics}
        states_dict: {truck_id: state_string}
    
    Returns:
        {truck_id: [recommendation_strings]}
    """
    result = {}
    for truck_id, metrics in metrics_dict.items():
        state = states_dict.get(truck_id, "EN_ROUTE")
        result[truck_id] = generate_recommendations(metrics, state)
    return result
