# 🚛 GreenRoute AI

**Real-Time Carbon-Aware Fleet Optimization + Live RAG Copilot**

A production-style streaming AI system that ingests simulated live fleet telemetry, computes carbon emissions in real time, detects anomalies, ranks trucks by eco-efficiency, and uses live RAG to explain violations and suggest greener actions.

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Fleet Simulator  │────▶│  Stream Engine    │────▶│  Transforms     │
│ (10 trucks)      │     │  (Temporal Windows)│     │  (Carbon, etc.) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
          ┌────────────────────────────────────────────────┤
          ▼                                                ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  State Machine   │     │   Ranking Engine  │     │   RAG Copilot   │
│  (5 States)      │     │   (Eco-Score)     │     │   (Gemini AI)   │
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │                        │                        │
         └────────────┬───────────┘────────────────────────┘
                      ▼
              ┌──────────────┐
              │   FastAPI    │
              │  /fleet_status│
              │  /alerts      │
              │  /ranking     │
              │  /ask (RAG)   │
              └──────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Gemini API Key
Create `.env` file:
```
GEMINI_API_KEY=your_key_here
```

### 3. Run Server
```bash
python app.py
```

### 4. Open Swagger UI
Navigate to: **http://localhost:8000/docs**

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/fleet_status` | GET | Live telemetry + window metrics for all 10 trucks |
| `/alerts` | GET | Active alerts, state violations, recommendations |
| `/ranking` | GET | Eco-score rankings (best & worst trucks) |
| `/ask` | POST | RAG Copilot — AI-powered Q&A with policy citations |

### Example RAG Query
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Why is truck 8 violating emission standards?"}'
```

---

## 🚛 Simulated Fleet

| Truck | Behavior | Expected State |
|-------|----------|----------------|
| 1–6 | Normal driving | EN_ROUTE |
| 7 | Speed fluctuation anomaly | DELAYED |
| 8 | High fuel consumption | EMISSION_ALERT |
| 9 | Idle for long periods | IDLE |
| 10 | Route deviation | EN_ROUTE / OPTIMIZATION_REQUIRED |

---

## 🔧 Key Features Demonstrated

- ✅ **Streaming Connectors** — Async generators + event bus
- ✅ **Temporal Windows** — 5-min sliding windows with eviction
- ✅ **Incremental Computation** — Metrics recomputed only on changes
- ✅ **Document Store** — Chunked, embedded, indexed knowledge base
- ✅ **LLM Integration** — Gemini 2.0 Flash for RAG responses
- ✅ **Modular Architecture** — Clean separation of concerns

---

## 📁 Project Structure

```
├── data/
│   ├── emission_policy.txt          # CO₂ regulations & thresholds
│   ├── sustainability_guidelines.txt # Eco-driving best practices
│   └── fleet_operations_manual.txt  # Fleet states & protocols
├── src/
│   ├── stream_engine.py     # Core streaming engine
│   ├── ingestion.py         # Fleet telemetry simulator
│   ├── transforms.py        # Rolling window computations
│   ├── state_machine.py     # Event-driven state transitions
│   ├── ranking.py           # Eco-score computation
│   ├── optimization.py      # Rule-based recommendations
│   └── rag_engine.py        # RAG copilot (Gemini)
├── app.py                   # FastAPI server
├── requirements.txt
├── .env
└── README.md
```
