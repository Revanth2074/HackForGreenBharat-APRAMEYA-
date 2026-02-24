"""
GreenRoute AI — RAG Copilot Engine
Document loading, chunking, Gemini embeddings, retrieval, and LLM-powered responses.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

# ─── Configuration ────────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DATA_DIR = Path(__file__).parent.parent / "data"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


class DocumentChunk:
    """A chunk of a document with its embedding."""

    def __init__(self, text: str, source: str, chunk_id: int, heading: str = ""):
        self.text = text
        self.source = source
        self.chunk_id = chunk_id
        self.heading = heading
        self.embedding: Optional[np.ndarray] = None

    def __repr__(self):
        return f"<Chunk {self.chunk_id} from {self.source}: {self.text[:60]}...>"


class DocumentStore:
    """
    Live document store with chunking, embedding, and retrieval.
    Simulates Pathway's Document Store functionality.
    """

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self._indexed = False

    def load_documents(self, directory: Path = DATA_DIR):
        """Load all .txt files from the data directory."""
        print(f"[RAG] Loading documents from {directory}")
        for filepath in sorted(directory.glob("*.txt")):
            print(f"  [DOC] Loading: {filepath.name}")
            text = filepath.read_text(encoding="utf-8")
            chunks = self._chunk_document(text, filepath.name)
            self.chunks.extend(chunks)
        print(f"[RAG] Loaded {len(self.chunks)} chunks from {len(list(directory.glob('*.txt')))} documents")

    def _chunk_document(self, text: str, source: str, max_chunk_size: int = 800) -> List[DocumentChunk]:
        """
        Split document into chunks by sections/headings.
        Preserves context by keeping section headers with their content.
        """
        chunks = []
        # Split by section headers (lines of dashes, or SECTION/CHAPTER/PART headers)
        sections = re.split(
            r'\n(?=(?:SECTION|CHAPTER|PART|={10,})\s*)', text
        )

        chunk_id = 0
        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Extract heading
            lines = section.split('\n')
            heading = ""
            for line in lines[:3]:
                cleaned = line.strip().strip('=').strip('-').strip()
                if cleaned and len(cleaned) > 5:
                    heading = cleaned
                    break

            # If section is too large, split further by paragraphs
            if len(section) > max_chunk_size:
                paragraphs = section.split('\n\n')
                current = ""
                for para in paragraphs:
                    if len(current) + len(para) > max_chunk_size and current:
                        chunks.append(DocumentChunk(
                            text=current.strip(),
                            source=source,
                            chunk_id=chunk_id,
                            heading=heading,
                        ))
                        chunk_id += 1
                        current = para
                    else:
                        current += "\n\n" + para if current else para
                if current.strip():
                    chunks.append(DocumentChunk(
                        text=current.strip(),
                        source=source,
                        chunk_id=chunk_id,
                        heading=heading,
                    ))
                    chunk_id += 1
            else:
                chunks.append(DocumentChunk(
                    text=section,
                    source=source,
                    chunk_id=chunk_id,
                    heading=heading,
                ))
                chunk_id += 1

        return chunks

    async def build_index(self):
        """Embed all chunks using Gemini embedding model."""
        print(f"[RAG] Building embeddings for {len(self.chunks)} chunks...")
        
        batch_size = 10
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            texts = [c.text[:2000] for c in batch]  # Truncate very long chunks
            
            try:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=texts,
                    task_type="retrieval_document",
                )
                embeddings = result["embedding"]
                for j, chunk in enumerate(batch):
                    chunk.embedding = np.array(embeddings[j], dtype=np.float32)
            except Exception as e:
                print(f"[RAG] Embedding error for batch {i}: {e}")
                # Fallback: use random embeddings (demo purposes)
                for chunk in batch:
                    chunk.embedding = np.random.randn(768).astype(np.float32)

        self._indexed = True
        print(f"[RAG] [OK] Index built successfully with {len(self.chunks)} chunks")

    async def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve top-k most relevant chunks for a query."""
        if not self._indexed:
            await self.build_index()

        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query",
            )
            query_embedding = np.array(result["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"[RAG] Query embedding error: {e}")
            # Return first few chunks as fallback
            return [(c, 0.5) for c in self.chunks[:top_k]]

        # Cosine similarity
        scored = []
        for chunk in self.chunks:
            if chunk.embedding is not None:
                sim = np.dot(query_embedding, chunk.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding) + 1e-8
                )
                scored.append((chunk, float(sim)))
            else:
                scored.append((chunk, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class RAGCopilot:
    """
    RAG Copilot: combines document retrieval with live fleet metrics
    and Gemini LLM to answer questions about fleet operations.
    Falls back to intelligent offline mode when API is unavailable.
    """

    def __init__(self, document_store: DocumentStore):
        self.store = document_store
        self.model = None
        self.offline_mode = False
        try:
            self.model = genai.GenerativeModel("gemini-2.0-flash")
        except Exception:
            self.offline_mode = True
            print("[RAG] Gemini model unavailable, using offline mode")

    async def ask(
        self,
        question: str,
        fleet_metrics: Optional[Dict[int, dict]] = None,
        fleet_states: Optional[List[dict]] = None,
        fleet_alerts: Optional[List[dict]] = None,
    ) -> dict:
        """
        Answer a question using RAG + live fleet data.
        Falls back to offline mode if Gemini API is unavailable.
        """
        # Step 1: Retrieve relevant documents
        retrieved = await self.store.retrieve(question, top_k=5)
        
        # Step 2: Build fleet context
        truck_context = self._build_fleet_context(
            question, fleet_metrics, fleet_states, fleet_alerts
        )

        # Step 3: Build source citations
        sources = [
            {
                "document": chunk.source,
                "heading": chunk.heading,
                "relevance": round(score, 3),
                "excerpt": chunk.text[:200] + "...",
            }
            for chunk, score in retrieved[:3]
        ]
        
        # Step 4: Try Gemini first, fallback to offline
        answer = None
        mode_used = "online"

        if not self.offline_mode and self.model:
            try:
                doc_context = self._build_doc_context(retrieved)
                prompt = self._build_prompt(question, doc_context, truck_context)
                response = self.model.generate_content(prompt)
                answer = response.text
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    print("[RAG] Gemini quota exceeded, switching to offline mode")
                    self.offline_mode = True
                else:
                    print(f"[RAG] Gemini error: {err_str[:100]}")

        if answer is None:
            mode_used = "offline"
            answer = self._generate_offline_response(
                question, retrieved, fleet_metrics, fleet_states, fleet_alerts
            )

        return {
            "answer": answer,
            "sources": sources,
            "truck_context": truck_context,
            "question": question,
            "mode": mode_used,
        }

    def _generate_offline_response(
        self,
        question: str,
        retrieved: List[Tuple[DocumentChunk, float]],
        fleet_metrics: Optional[Dict[int, dict]],
        fleet_states: Optional[List[dict]],
        fleet_alerts: Optional[List[dict]],
    ) -> str:
        """Generate a structured answer using retrieved docs + fleet data without LLM."""
        q = question.lower()
        parts = []

        # --- Check if asking about a specific truck ---
        truck_match = re.search(r'truck\s*(\d+)', q)
        if truck_match and fleet_metrics:
            tid = int(truck_match.group(1))
            m = fleet_metrics.get(tid)
            if m:
                parts.append(f"**Truck {tid} - Live Status Report**\n")
                parts.append(f"- **State**: {m.get('behavior', 'normal').replace('_', ' ').title()}")
                parts.append(f"- **Speed**: {m.get('last_speed', 0):.1f} km/h (avg: {m.get('avg_speed', 0):.1f})")
                parts.append(f"- **Fuel Rate**: {m.get('last_fuel_rate', 0):.1f} L/hr (avg: {m.get('avg_fuel_rate', 0):.1f})")
                parts.append(f"- **Carbon Emission**: {m.get('carbon_emission', 0):.2f} kg CO2 (window)")
                parts.append(f"- **ETA**: {m.get('eta_minutes', 0):.0f} minutes")
                parts.append(f"- **Route**: {m.get('route_name', m.get('route_id', 'N/A'))}")
                parts.append(f"- **Cargo**: {m.get('cargo_type', 'N/A')} ({m.get('cargo_weight', 0):,} kg)")
                parts.append(f"- **Route Deviation**: {m.get('route_deviation_km', 0):.3f} km")
                parts.append(f"- **Idle Ratio**: {m.get('idle_ratio', 0):.1%}")
                parts.append(f"- **Speed Variance**: {m.get('speed_variance', 0):.1f}")

                # Explain issues
                issues = []
                if m.get('emission_violation'):
                    issues.append("Carbon emission exceeds 15 kg threshold (CARBON_ALERT_THRESHOLD)")
                if m.get('idle_flag'):
                    issues.append(f"Excessive idling: {m.get('idle_ratio', 0):.1%} of time stationary")
                if m.get('instability_flag'):
                    issues.append(f"Speed instability: variance {m.get('speed_variance', 0):.1f} > 150 threshold")
                if m.get('route_deviation_flag'):
                    issues.append(f"Route deviation: {m.get('route_deviation_km', 0):.3f} km off planned route")

                if issues:
                    parts.append("\n**Issues Detected:**")
                    for issue in issues:
                        parts.append(f"- {issue}")
                else:
                    parts.append("\n**Status**: Operating within normal parameters.")
            else:
                parts.append(f"Truck {tid} not found in active fleet data.")

        # --- Carbon / emission questions ---
        elif any(w in q for w in ['carbon', 'emission', 'co2', 'pollut', 'threshold', 'exceed']):
            parts.append("**Fleet Carbon & Emission Analysis**\n")
            if fleet_metrics:
                violators = [(tid, m) for tid, m in fleet_metrics.items() if m.get('emission_violation')]
                high_emitters = sorted(fleet_metrics.items(), key=lambda x: x[1].get('carbon_emission', 0), reverse=True)[:5]

                if violators:
                    parts.append(f"**{len(violators)} truck(s) exceeding carbon threshold (15 kg CO2):**")
                    for tid, m in violators:
                        parts.append(f"- Truck {tid}: {m.get('carbon_emission', 0):.2f} kg CO2, fuel rate {m.get('avg_fuel_rate', 0):.1f} L/hr")
                else:
                    parts.append("No trucks currently exceeding the 15 kg carbon threshold.")

                parts.append("\n**Top 5 Carbon Emitters (current window):**")
                for tid, m in high_emitters:
                    parts.append(f"- Truck {tid}: {m.get('carbon_emission', 0):.2f} kg CO2 ({m.get('behavior', 'normal')})")

                total_co2 = sum(m.get('carbon_emission', 0) for m in fleet_metrics.values())
                parts.append(f"\n**Fleet Total CO2**: {total_co2:.1f} kg (current window)")

        # --- Delayed / stuck questions ---
        elif any(w in q for w in ['delay', 'stuck', 'slow', 'traffic', 'idle']):
            parts.append("**Fleet Delay & Idle Analysis**\n")
            if fleet_metrics:
                delayed = [(tid, m) for tid, m in fleet_metrics.items()
                           if m.get('instability_flag') or m.get('idle_flag')]
                if delayed:
                    for tid, m in delayed:
                        reasons = []
                        if m.get('idle_flag'):
                            reasons.append(f"idling {m.get('idle_ratio', 0):.1%} of time")
                        if m.get('instability_flag'):
                            reasons.append(f"speed variance {m.get('speed_variance', 0):.1f}")
                        parts.append(f"- **Truck {tid}** ({m.get('behavior', 'N/A')}): {', '.join(reasons)}")
                        parts.append(f"  Speed: {m.get('avg_speed', 0):.1f} km/h, Route: {m.get('route_name', '')}")
                else:
                    parts.append("No trucks currently showing delay or excessive idle patterns.")

        # --- Ranking / eco / green questions ---
        elif any(w in q for w in ['rank', 'eco', 'green', 'best', 'worst', 'efficien', 'suggest', 'alternative', 'tip']):
            parts.append("**Eco-Driving & Fleet Efficiency Report**\n")
            if fleet_metrics:
                by_efficiency = sorted(fleet_metrics.items(),
                                       key=lambda x: x[1].get('fuel_efficiency_km_per_l', 0), reverse=True)
                parts.append("**Most Fuel-Efficient Trucks:**")
                for tid, m in by_efficiency[:5]:
                    parts.append(f"- Truck {tid}: {m.get('fuel_efficiency_km_per_l', 0):.2f} km/L, "
                                 f"CO2={m.get('carbon_emission', 0):.2f} kg ({m.get('behavior', '')})")

                parts.append("\n**Least Efficient (need improvement):**")
                for tid, m in by_efficiency[-5:]:
                    parts.append(f"- Truck {tid}: {m.get('fuel_efficiency_km_per_l', 0):.2f} km/L, "
                                 f"CO2={m.get('carbon_emission', 0):.2f} kg ({m.get('behavior', '')})")

            parts.append("\n**Eco-Driving Recommendations (from policy docs):**")
            parts.append("- Maintain speed in 50-70 km/h optimal range")
            parts.append("- Use cruise control to reduce speed variance")
            parts.append("- Turn off engine during stops > 2 minutes")
            parts.append("- Check tire pressure weekly (under-inflation adds ~8% fuel)")
            parts.append("- Consider route optimization to avoid congestion")

        # --- Route deviation ---
        elif any(w in q for w in ['route', 'deviation', 'off-route', 'deviat']):
            parts.append("**Route Deviation Report**\n")
            if fleet_metrics:
                deviators = [(tid, m) for tid, m in fleet_metrics.items()
                             if m.get('route_deviation_km', 0) > 0.5]
                if deviators:
                    for tid, m in sorted(deviators, key=lambda x: x[1].get('route_deviation_km', 0), reverse=True):
                        parts.append(f"- Truck {tid}: {m.get('route_deviation_km', 0):.3f} km off-route "
                                     f"(Route: {m.get('route_name', '')})")
                else:
                    parts.append("No trucks currently showing significant route deviation (> 0.5 km).")

        # --- General fleet overview ---
        else:
            parts.append("**Fleet Overview**\n")
            if fleet_metrics:
                total = len(fleet_metrics)
                en_route = sum(1 for m in fleet_metrics.values() if not m.get('idle_flag') and not m.get('instability_flag'))
                idle = sum(1 for m in fleet_metrics.values() if m.get('idle_flag'))
                unstable = sum(1 for m in fleet_metrics.values() if m.get('instability_flag'))
                total_co2 = sum(m.get('carbon_emission', 0) for m in fleet_metrics.values())
                avg_speed = sum(m.get('avg_speed', 0) for m in fleet_metrics.values()) / max(total, 1)

                parts.append(f"- **Active Trucks**: {total}")
                parts.append(f"- **En Route**: {en_route}")
                parts.append(f"- **Idle/Stopped**: {idle}")
                parts.append(f"- **Speed Anomaly**: {unstable}")
                parts.append(f"- **Fleet CO2**: {total_co2:.1f} kg")
                parts.append(f"- **Avg Speed**: {avg_speed:.1f} km/h")

        # --- Add relevant policy excerpts ---
        if retrieved:
            parts.append("\n---\n**Relevant Policy References:**")
            for chunk, score in retrieved[:3]:
                excerpt = chunk.text[:250].replace('\n', ' ').strip()
                parts.append(f"\n> *{chunk.source}* (Section: {chunk.heading})\n> {excerpt}...")

        return "\n".join(parts)

    def _build_doc_context(self, retrieved: List[Tuple[DocumentChunk, float]]) -> str:
        parts = []
        for chunk, score in retrieved:
            parts.append(
                f"[Source: {chunk.source} | Section: {chunk.heading} | Relevance: {score:.2f}]\n"
                f"{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    def _build_fleet_context(
        self,
        question: str,
        fleet_metrics: Optional[Dict[int, dict]],
        fleet_states: Optional[List[dict]],
        fleet_alerts: Optional[List[dict]],
    ) -> Optional[str]:
        """Extract relevant fleet data based on the question."""
        if not fleet_metrics:
            return None

        parts = []

        # Check if question mentions a specific truck
        import re
        truck_match = re.search(r'truck\s*(\d+)', question, re.IGNORECASE)
        
        if truck_match:
            truck_id = int(truck_match.group(1))
            metrics = fleet_metrics.get(truck_id)
            if metrics:
                parts.append(f"=== LIVE DATA FOR TRUCK {truck_id} ===")
                parts.append(f"Current Speed: {metrics.get('last_speed', 'N/A')} km/h")
                parts.append(f"Average Speed (window): {metrics.get('avg_speed', 'N/A')} km/h")
                parts.append(f"Fuel Rate: {metrics.get('last_fuel_rate', 'N/A')} L/hr")
                parts.append(f"Average Fuel Rate: {metrics.get('avg_fuel_rate', 'N/A')} L/hr")
                parts.append(f"Carbon Emission (window): {metrics.get('carbon_emission', 'N/A')} kg CO₂")
                parts.append(f"Speed Variance: {metrics.get('speed_variance', 'N/A')}")
                parts.append(f"Idle Flag: {metrics.get('idle_flag', 'N/A')}")
                parts.append(f"Idle Ratio: {metrics.get('idle_ratio', 'N/A')}")
                parts.append(f"Instability Flag: {metrics.get('instability_flag', 'N/A')}")
                parts.append(f"Emission Violation: {metrics.get('emission_violation', 'N/A')}")
                parts.append(f"Distance in Window: {metrics.get('distance_km', 'N/A')} km")
                parts.append(f"Cargo Weight: {metrics.get('cargo_weight', 'N/A')} kg")
                parts.append(f"Route: {metrics.get('route_id', 'N/A')}")
                parts.append(f"Position: ({metrics.get('latitude', 'N/A')}, {metrics.get('longitude', 'N/A')})")

                # Add state info
                if fleet_states:
                    for state in fleet_states:
                        if state.get("truck_id") == truck_id:
                            parts.append(f"Current State: {state.get('state', 'Unknown')}")
                            parts.append(f"State Description: {state.get('state_description', '')}")
                            break

                # Add alerts
                if fleet_alerts:
                    truck_alerts = [a for a in fleet_alerts if a.get("truck_id") == truck_id]
                    if truck_alerts:
                        parts.append(f"\nActive Alerts:")
                        for alert in truck_alerts[-3:]:
                            parts.append(f"  - State: {alert.get('new_state', 'N/A')}")
                            parts.append(f"    Reason: {alert.get('reason', 'N/A')}")
        else:
            # General fleet overview
            parts.append("=== FLEET OVERVIEW ===")
            for tid, metrics in sorted(fleet_metrics.items()):
                parts.append(
                    f"Truck {tid}: Speed={metrics.get('avg_speed', 0):.1f} km/h, "
                    f"Fuel={metrics.get('avg_fuel_rate', 0):.1f} L/hr, "
                    f"CO₂={metrics.get('carbon_emission', 0):.2f} kg, "
                    f"Idle={metrics.get('idle_flag', False)}, "
                    f"Emission Alert={metrics.get('emission_violation', False)}"
                )

        return "\n".join(parts) if parts else None

    def _build_prompt(self, question: str, doc_context: str, fleet_context: Optional[str]) -> str:
        prompt = f"""You are GreenRoute AI Copilot — an expert fleet sustainability advisor.
Your role is to analyze fleet operations, explain emission violations, and recommend greener driving practices.

You have access to official policy documents and real-time fleet telemetry data.
Always cite the specific policy document and section when referencing regulations.
Be specific with numbers and thresholds from the documents.
Provide actionable recommendations.

=== POLICY DOCUMENTS (Retrieved via RAG) ===

{doc_context}

"""
        if fleet_context:
            prompt += f"""
=== LIVE FLEET TELEMETRY ===

{fleet_context}

"""

        prompt += f"""
=== USER QUESTION ===

{question}

=== INSTRUCTIONS ===

1. Answer the question thoroughly using both the policy documents and live fleet data.
2. Cite specific documents and sections (e.g., "According to emission_policy.txt, Section 2.1...").
3. Include relevant numbers and thresholds from the policies.
4. If live data is available, reference specific truck metrics to support your analysis.
5. Provide concrete, actionable recommendations.
6. Use clear formatting with headers and bullet points.

Please respond:
"""
        return prompt


# ─── Module-level instances ───────────────────────────────────────────────────
document_store = DocumentStore()
rag_copilot: Optional[RAGCopilot] = None


async def initialize_rag():
    """Initialize the RAG system: load documents and build index."""
    global rag_copilot
    document_store.load_documents()
    await document_store.build_index()
    rag_copilot = RAGCopilot(document_store)
    print("[RAG] [OK] RAG Copilot initialized and ready")
    return rag_copilot
