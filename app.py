"""
GreenRoute AI -- FastAPI Application v2
Real-time fleet monitoring API with RAG copilot and web chat interface.
50 trucks across 10 Indian highway routes.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from src.ingestion import fleet_stream
from src.optimization import generate_recommendations, get_fleet_recommendations
from src.ranking import get_fleet_ranking, get_top_eco_trucks, get_worst_emitters
from src.rag_engine import initialize_rag, rag_copilot
from src.state_machine import fleet_state_machine, state_machine_subscriber
from src.stream_engine import StreamEngine
from src.transforms import transform_subscriber

load_dotenv()

# --- Global state ---
engine = StreamEngine(window_duration=300.0)  # 5-minute window
stream_task: Optional[asyncio.Task] = None


async def start_streaming():
    """Start the fleet stream ingestion."""
    print("=" * 60)
    print("  [GREENROUTE AI] Starting Fleet Stream (50 Trucks)")
    print("=" * 60)

    # Register subscribers (order matters: transforms -> state machine)
    engine.subscribe(transform_subscriber)
    engine.subscribe(state_machine_subscriber)

    # Start ingesting
    print("[Stream] Ingesting fleet telemetry (50 trucks, 10 routes, 400ms interval)...")
    generator = fleet_stream(num_trucks=50, interval=0.4, time_acceleration=30.0)
    await engine.ingest(generator)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: start stream + RAG on startup, cleanup on shutdown."""
    global stream_task

    print("\n[+] Starting GreenRoute AI server...\n")

    # Initialize RAG copilot
    try:
        await initialize_rag()
    except Exception as e:
        print(f"[WARNING] RAG initialization error: {e}")
        print("  (RAG copilot may not be available)")

    # Start streaming in background
    stream_task = asyncio.create_task(start_streaming())

    # Wait a moment for initial data
    print("[Server] Waiting 5s for initial data accumulation...")
    await asyncio.sleep(5)
    print(f"[Server] [OK] {engine.event_count} events processed. Server ready!\n")

    yield

    # Cleanup
    print("\n[-] Shutting down GreenRoute AI...")
    engine.stop()
    if stream_task:
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass


# --- FastAPI App ---
app = FastAPI(
    title="GreenRoute AI",
    description=(
        "Real-Time Carbon-Aware Fleet Optimization + Live RAG Copilot.\n\n"
        "## Features\n"
        "- **50 Trucks** across 10 Indian highway routes\n"
        "- **Live Fleet Telemetry**: GPS, speed, fuel, cargo\n"
        "- **Carbon Emissions**: Rolling window CO2 calculations\n"
        "- **ETA Estimation**: Real-time arrival predictions\n"
        "- **Route Deviation**: GPS-based off-route detection\n"
        "- **Anomaly Detection**: Speed variance, idle, emission violations\n"
        "- **State Machine**: Automatic state transitions with alerts\n"
        "- **Eco Ranking**: Trucks ranked by sustainability score\n"
        "- **RAG Copilot**: AI-powered Q&A with 6 policy documents\n"
        "- **Chat Interface**: Web-based fleet command center\n"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Request/Response Models ---
class AskRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "question": "Why is truck 8 violating emission standards?"
        }
    })
    question: str


# --- Endpoints ---

@app.get("/", tags=["System"])
async def root():
    """Serve the web chat interface."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "system": "GreenRoute AI",
        "version": "2.0.0",
        "status": "running",
        "event_count": engine.event_count,
        "active_trucks": len(engine.metrics),
        "ui": "/static/index.html",
        "docs": "/docs",
    }


@app.get("/fleet_status", tags=["Fleet Monitoring"])
async def fleet_status():
    """Get live fleet status -- real-time telemetry + window metrics for all 50 trucks."""
    metrics = engine.get_all_metrics()
    states = fleet_state_machine.get_all_states()
    state_lookup = {s["truck_id"]: s for s in states}

    fleet = []
    for truck_id in sorted(metrics.keys()):
        m = metrics[truck_id]
        s = state_lookup.get(truck_id, {})

        fleet.append({
            "truck_id": truck_id,
            "driver_name": m.get("driver_name", ""),
            "position": {
                "latitude": m.get("latitude"),
                "longitude": m.get("longitude"),
            },
            "current": {
                "speed": m.get("last_speed"),
                "fuel_rate": m.get("last_fuel_rate"),
                "cargo_weight": m.get("cargo_weight"),
                "cargo_type": m.get("cargo_type", ""),
                "route_id": m.get("route_id"),
                "route_name": m.get("route_name", ""),
            },
            "window_metrics": {
                "avg_speed": m.get("avg_speed"),
                "max_speed": m.get("max_speed"),
                "min_speed": m.get("min_speed"),
                "avg_fuel_rate": m.get("avg_fuel_rate"),
                "total_fuel": m.get("total_fuel"),
                "carbon_emission": m.get("carbon_emission"),
                "distance_km": m.get("distance_km"),
                "speed_variance": m.get("speed_variance"),
                "idle_ratio": m.get("idle_ratio"),
                "fuel_efficiency_km_per_l": m.get("fuel_efficiency_km_per_l"),
                "eta_minutes": m.get("eta_minutes"),
                "remaining_km": m.get("remaining_km"),
                "cumulative_distance_km": m.get("cumulative_distance_km"),
                "route_deviation_km": m.get("route_deviation_km"),
            },
            "flags": {
                "idle_flag": m.get("idle_flag"),
                "instability_flag": m.get("instability_flag"),
                "emission_violation": m.get("emission_violation"),
                "route_deviation_flag": m.get("route_deviation_flag"),
            },
            "state": s.get("state", "UNKNOWN"),
            "state_description": s.get("state_description", ""),
            "behavior": m.get("behavior", ""),
            "window_events": m.get("window_events"),
            "window_span_seconds": m.get("window_span_seconds"),
            "timestamp": m.get("timestamp"),
        })

    return {
        "total_trucks": len(fleet),
        "event_count": engine.event_count,
        "fleet": fleet,
    }


@app.get("/alerts", tags=["Alert System"])
async def alerts():
    """Get current alerts -- active state violations and recent alert history."""
    active = fleet_state_machine.get_active_alerts()
    recent = fleet_state_machine.get_recent_alerts(n=30)
    states = fleet_state_machine.get_all_states()

    alerts_with_recs = []
    for alert in active:
        truck_id = alert["truck_id"]
        metrics = engine.metrics.get(truck_id, {})
        state = alert.get("new_state", "EN_ROUTE")
        recs = generate_recommendations(metrics, state)
        alerts_with_recs.append({
            **alert,
            "recommendations": recs,
        })

    return {
        "active_alerts": alerts_with_recs,
        "recent_alerts": recent,
        "truck_states": states,
        "summary": {
            "total_trucks": len(states),
            "en_route": sum(1 for s in states if s["state"] == "EN_ROUTE"),
            "idle": sum(1 for s in states if s["state"] == "IDLE"),
            "delayed": sum(1 for s in states if s["state"] == "DELAYED"),
            "emission_alert": sum(1 for s in states if s["state"] == "EMISSION_ALERT"),
            "optimization_required": sum(1 for s in states if s["state"] == "OPTIMIZATION_REQUIRED"),
        },
    }


@app.get("/ranking", tags=["Eco Ranking"])
async def ranking():
    """Get fleet eco-score rankings -- trucks ranked by sustainability performance."""
    rankings = get_fleet_ranking(engine)
    top_3 = get_top_eco_trucks(engine, n=3)
    worst_3 = get_worst_emitters(engine, n=3)

    return {
        "rankings": rankings,
        "top_eco_trucks": [
            {"rank": t["rank"], "truck_id": t["truck_id"], "eco_score": t["eco_score"], "rating": t["rating"]}
            for t in top_3
        ],
        "worst_emitters": [
            {"rank": t["rank"], "truck_id": t["truck_id"], "eco_score": t["eco_score"], "rating": t["rating"]}
            for t in worst_3
        ],
    }


@app.post("/ask", tags=["RAG Copilot"])
async def ask(request: AskRequest):
    """
    Ask the GreenRoute AI Copilot a question.
    
    The copilot uses RAG (Retrieval-Augmented Generation) to:
    1. Retrieve relevant policy documents (6 knowledge base docs)
    2. Fetch live truck metrics from all 50 trucks
    3. Generate an AI-powered response with citations
    
    Example questions:
    - "Why is truck 8 violating emission standards?"
    - "Which truck exceeded the carbon threshold?"
    - "Suggest greener alternatives for the fleet"
    - "What Indian emission regulations apply to our fleet?"
    """
    from src.rag_engine import rag_copilot as copilot

    if copilot is None:
        raise HTTPException(
            status_code=503,
            detail="RAG Copilot is not initialized. Please wait for startup to complete.",
        )

    fleet_metrics = engine.get_all_metrics()
    fleet_states = fleet_state_machine.get_all_states()
    fleet_alerts = fleet_state_machine.get_recent_alerts(n=10)

    result = await copilot.ask(
        question=request.question,
        fleet_metrics=fleet_metrics,
        fleet_states=fleet_states,
        fleet_alerts=fleet_alerts,
    )

    return result


# --- Run directly ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
