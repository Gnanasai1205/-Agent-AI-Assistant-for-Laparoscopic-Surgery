from __future__ import annotations
import argparse
import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import PIL.Image

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import requests


class SentenceTransformerEmbeddings(Embeddings):
    """
    Lightweight LangChain-compatible embeddings using sentence-transformers directly.
    Uses PyTorch only — never imports TensorFlow or triggers the Keras 3 conflict.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # local import: PyTorch only
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()



def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def severity_from_score(score: float) -> str:
    if score < 0.3:
        return "LOW"
    if score < 0.7:
        return "MODERATE"
    return "HIGH"


@dataclass
class RetrievalResult:
    snippets: List[str]
    query: str


class MedicalRetriever:
    """RAG retriever over a local medical knowledge file using LangChain + FAISS."""

    def __init__(self, knowledge_path: Path) -> None:
        if not knowledge_path.exists():
            raise FileNotFoundError(f"Knowledge file not found: {knowledge_path}")

        raw_text = knowledge_path.read_text(encoding="utf-8")
        splitter = RecursiveCharacterTextSplitter(chunk_size=380, chunk_overlap=40)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(raw_text)]

        # SentenceTransformerEmbeddings: real semantic search via PyTorch only (no TF/Keras).
        # Model (~90 MB) is downloaded once and cached in ~/.cache/huggingface.
        embeddings = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        self._vectorstore = FAISS.from_documents(docs, embeddings)

    def run(self, query: str, k: int = 4) -> RetrievalResult:
        matches = self._vectorstore.similarity_search(query, k=k)
        return RetrievalResult(snippets=[m.page_content.strip() for m in matches], query=query)


class VisionAnalyzer:
    """Proxy tool for vision findings; accepts detector output from upstream systems."""

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        vision = payload.get("vision_findings", {}) or {}
        organs = vision.get("organs", [])
        instruments = vision.get("instruments", [])
        issues = vision.get("issues", [])
        return {
            "organs": organs,
            "instruments": instruments,
            "issues": issues,
            "raw": vision,
        }


class VisionAnalyzer:
    """Extracts surgical image findings using Gemini or Groq multimodal API."""

    def __init__(self, gemini_key: str = "", groq_key: str = "", model: str = "") -> None:
        self.gemini_key = gemini_key
        self.groq_key = groq_key
        self.model = model
        self.groq_endpoint = "https://api.groq.com/openai/v1/chat/completions"

        if self.gemini_key and GEMINI_AVAILABLE:
            genai.configure(api_key=self.gemini_key)

    def _image_to_data_url(self, image_path: Path) -> str:
        mime, _ = mimetypes.guess_type(str(image_path))
        if not mime:
            mime = "image/jpeg"
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}"

    def _extract_json(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict): return obj
        except json.JSONDecodeError: pass

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict): return obj
            except json.JSONDecodeError: return {}
        return {}

    def run(self, image_path: Path) -> Dict[str, Any]:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        prompt = (
            "You are a laparoscopic surgery vision assistant. Analyze the frame and return JSON only with keys: "
            "organs (array of strings), instruments (array of strings), issues (array of strings), "
            "procedure_guess (string), operative_step_guess (string), uncertainties (array of strings), "
            "image_quality (string), summary (string). Be concise and clinically relevant."
        )

        content = ""
        # 1. Try Gemini first if available
        if self.gemini_key and GEMINI_AVAILABLE:
            model_name = self.model if self.model and "gemini" in self.model.lower() else "gemini-2.5-flash"
            model = genai.GenerativeModel(model_name)
            img = PIL.Image.open(image_path)
            response = model.generate_content([prompt, img])
            content = response.text

        # 2. Fallback to Groq
        elif self.groq_key:
            model_name = self.model if self.model else "meta-llama/llama-4-scout-17b-16e-instruct"
            response = requests.post(
                self.groq_endpoint,
                headers={"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"},
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": self._image_to_data_url(image_path)}},
                            ],
                        }
                    ],
                    "temperature": 0.1,
                },
                timeout=60,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError("No valid API key provided (need Gemini or Groq).")

        extracted = self._extract_json(content)
        if not extracted:
            extracted = {
                "organs": [], "instruments": [], "issues": ["Parse error"],
                "procedure_guess": "unknown", "operative_step_guess": "unknown",
                "uncertainties": ["Model output invalid JSON."], "image_quality": "unknown", "summary": content[:200]
            }
        return extracted



class RiskEvaluator:
    """Rule-based risk model with mandated LOW/MODERATE/HIGH thresholds."""

    def run(self, payload: Dict[str, Any], vision_obs: Dict[str, Any], retrieval: RetrievalResult) -> Dict[str, Any]:
        vitals = payload.get("vitals", {}) or {}
        suspected = [s.lower() for s in payload.get("suspected_complications", [])]
        issues = [s.lower() for s in vision_obs.get("issues", [])]

        score = 0.1
        rationale: List[str] = []

        if any("bleed" in x or "hemorrhage" in x for x in issues + suspected):
            score += 0.5
            rationale.append("Potential intraoperative bleeding signal detected.")

        sbp = vitals.get("bp_systolic")
        hr = vitals.get("hr")
        spo2 = vitals.get("spo2")

        if isinstance(sbp, (int, float)) and sbp < 90:
            score += 0.2
            rationale.append("Hypotension increases immediate intraoperative risk.")

        if isinstance(hr, (int, float)) and hr > 120:
            score += 0.1
            rationale.append("Tachycardia may indicate instability or bleeding.")

        if isinstance(spo2, (int, float)) and spo2 < 92:
            score += 0.2
            rationale.append("Low oxygen saturation suggests respiratory compromise risk.")

        if any("unclear anatomy" in x or "critical view not achieved" in x for x in issues):
            score += 0.2
            rationale.append("Unclear anatomy elevates iatrogenic injury risk.")

        if any("bile leak" in x for x in issues + suspected):
            score += 0.2
            rationale.append("Bile leak suspicion requires urgent verification and source control.")

        if any("thermal" in x for x in issues + suspected):
            score += 0.15
            rationale.append("Possible thermal spread requires immediate tissue reassessment.")

        if retrieval.snippets:
            rationale.append("Risk interpretation anchored to retrieved laparoscopic safety guidance.")

        score = clamp(score, 0.0, 1.0)
        severity = severity_from_score(score)

        return {
            "risk_score": round(score, 2),
            "severity": severity,
            "rationale": rationale,
        }


class SurgicalAssistantAgent:
    def __init__(self, knowledge_path: Path) -> None:
        self.medical_retriever = MedicalRetriever(knowledge_path)
        self.vision_analyzer = VisionAnalyzer()
        self.risk_evaluator = RiskEvaluator()

    def _has_minimum_context(self, payload: Dict[str, Any]) -> bool:
        required = ["procedure", "operative_step", "vitals", "vision_findings"]
        for key in required:
            if key not in payload or payload.get(key) in (None, "", {}):
                return False
        return True

    def ask_question(self, query: str, gemini_key: str = "", groq_key: str = "") -> str:
        """Answers arbitrary clinical questions using the MedicalRetriever + Gemini/Groq LLM."""
        retrieval = self.medical_retriever.run(query, k=5)
        context = "\n\n".join(retrieval.snippets)

        prompt = (
            "You are an expert laparoscopic surgery AI assistant.\n"
            "Below are some clinical guidelines and best practices retrieved from our knowledge base:\n"
            "---\n"
            f"{context}\n"
            "---\n"
            "Based ONLY on the guidelines above, answer the user's question concisely. "
            "If the guidelines do not contain the answer, say 'I cannot answer this based on the current knowledge base.' "
            f"\n\nQuestion: {query}"
        )

        if groq_key:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        elif gemini_key and GEMINI_AVAILABLE:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        else:
            return "Error: Neither GEMINI_API_KEY nor GROQ_API_KEY is configured for chat."

    def _build_query(self, payload: Dict[str, Any]) -> str:
        procedure = payload.get("procedure", "unknown procedure")
        step = payload.get("operative_step", "unknown step")
        issues = payload.get("vision_findings", {}).get("issues", [])
        complications = payload.get("suspected_complications", [])
        return (
            f"Procedure: {procedure}. Step: {step}. "
            f"Vision issues: {', '.join(issues) if issues else 'none'}. "
            f"Suspected complications: {', '.join(complications) if complications else 'none'}."
        )

    def run(self, payload: Dict[str, Any]) -> str:
        lines: List[str] = []

        # Mandatory: call MedicalRetriever before clinical explanation.
        query = self._build_query(payload)
        lines.append("Thought: I should retrieve relevant laparoscopic safety context first before any explanation.")
        lines.append("Action: MedicalRetriever")
        lines.append(f"Action Input: {json.dumps({'query': query}, ensure_ascii=True)}")
        retrieval = self.medical_retriever.run(query)
        lines.append(
            "Observation: "
            + json.dumps({"top_snippets": retrieval.snippets[:3], "count": len(retrieval.snippets)}, ensure_ascii=True)
        )

        if not self._has_minimum_context(payload):
            lines.append("")
            lines.append("Final Answer:")
            lines.append("Insufficient medical data")
            return "\n".join(lines)

        lines.append("Thought: I should parse vision outputs to identify organs, instruments, and intraoperative issues.")
        lines.append("Action: VisionAnalyzer")
        lines.append(
            "Action Input: "
            + json.dumps({"vision_findings": payload.get("vision_findings", {})}, ensure_ascii=True)
        )
        vision_obs = self.vision_analyzer.run(payload)
        lines.append("Observation: " + json.dumps(vision_obs, ensure_ascii=True))

        lines.append("Thought: I should estimate risk score and severity using observed findings plus retrieved context.")
        lines.append("Action: RiskEvaluator")
        lines.append(
            "Action Input: "
            + json.dumps(
                {
                    "vitals": payload.get("vitals", {}),
                    "suspected_complications": payload.get("suspected_complications", []),
                    "vision_issues": vision_obs.get("issues", []),
                },
                ensure_ascii=True,
            )
        )
        risk = self.risk_evaluator.run(payload, vision_obs, retrieval)
        lines.append("Observation: " + json.dumps(risk, ensure_ascii=True))

        immediate_actions: List[str] = []
        issues_lower = [x.lower() for x in vision_obs.get("issues", [])]

        if any("bleed" in x or "hemorrhage" in x for x in issues_lower):
            immediate_actions.append("Control hemorrhage: suction, direct pressure, improve exposure, then secure source.")
            immediate_actions.append("Request senior surgical support and communicate blood loss trend to anesthesia.")

        if any("unclear anatomy" in x or "critical view not achieved" in x for x in issues_lower):
            immediate_actions.append("Pause dissection, re-establish anatomical landmarks, and avoid blind clipping/transection.")

        if any("thermal" in x for x in issues_lower):
            immediate_actions.append("Stop energy use near vulnerable structures and inspect for collateral thermal injury.")

        if any("bile leak" in x for x in issues_lower):
            immediate_actions.append("Inspect likely leak source, irrigate, and consider immediate repair/drain strategy.")

        if not immediate_actions:
            immediate_actions.append("Continue cautious dissection with continuous reassessment and team communication.")

        escalation = []
        if risk["severity"] == "HIGH":
            escalation.append("Prepare for damage-control strategy and low threshold for conversion to open surgery.")
            escalation.append("Activate additional support and blood product readiness per institutional protocol.")
        elif risk["severity"] == "MODERATE":
            escalation.append("Increase reassessment frequency and verify anatomy before irreversible steps.")
        else:
            escalation.append("Maintain standard safety checks and monitor for trend deterioration.")

        lines.append("")
        lines.append("Final Answer:")
        lines.append("Structured Surgical Insight")
        lines.append(f"- Procedure: {payload.get('procedure')}")
        lines.append(f"- Operative Step: {payload.get('operative_step')}")
        lines.append(f"- Detected Organs: {vision_obs.get('organs', [])}")
        lines.append(f"- Detected Instruments: {vision_obs.get('instruments', [])}")
        lines.append(f"- Detected Issues: {vision_obs.get('issues', [])}")
        lines.append(f"- Risk Score: {risk['risk_score']}")
        lines.append(f"- Risk Level: {risk['severity']}")
        lines.append("- Retrieved Evidence (RAG):")
        for snip in retrieval.snippets[:3]:
            lines.append(f"  - {snip}")
        lines.append("- Recommended Immediate Actions:")
        for item in immediate_actions:
            lines.append(f"  - {item}")
        lines.append("- Escalation Guidance:")
        for item in escalation:
            lines.append(f"  - {item}")
        lines.append("- Safety Note: This assistant supports clinical reasoning and does not replace surgeon judgment.")

        return "\n".join(lines)

    def run_from_image(self, image_findings: Dict[str, Any], image_path: Path, vitals: Dict[str, Any] | None = None) -> str:
        if vitals is None:
            vitals = {}
        procedure = image_findings.get("procedure_guess", "unknown procedure")
        step = image_findings.get("operative_step_guess", "unknown step")
        vision_findings = {
            "organs": image_findings.get("organs", []),
            "instruments": image_findings.get("instruments", []),
            "issues": image_findings.get("issues", []),
        }

        payload = {
            "procedure": procedure,
            "operative_step": step,
            "vitals": vitals,
            "vision_findings": vision_findings,
            "suspected_complications": [],
        }

        query = self._build_query(payload)
        retrieval = self.medical_retriever.run(query)
        risk = self.risk_evaluator.run(payload, vision_findings, retrieval)

        issues_lower = [x.lower() for x in vision_findings.get("issues", [])]
        immediate_actions: List[str] = []

        if any("bleed" in x or "hemorrhage" in x for x in issues_lower):
            immediate_actions.append("Control bleeding with suction, direct pressure, and precise source control.")
        if any("unclear anatomy" in x or "critical view not achieved" in x for x in issues_lower):
            immediate_actions.append("Pause dissection and re-establish landmarks before clipping or transection.")
        if any("thermal" in x for x in issues_lower):
            immediate_actions.append("Stop energy use near critical structures and inspect for collateral injury.")
        if any("bile leak" in x for x in issues_lower):
            immediate_actions.append("Irrigate and identify leak origin; plan repair or drainage as appropriate.")
        if not immediate_actions:
            immediate_actions.append("Continue with careful dissection and active reassessment of anatomy.")

        lines: List[str] = []
        lines.append("Thought: I should retrieve laparoscopic safety guidance before final interpretation.")
        lines.append("Action: MedicalRetriever")
        lines.append(f"Action Input: {json.dumps({'query': query}, ensure_ascii=True)}")
        lines.append(
            "Observation: "
            + json.dumps({"top_snippets": retrieval.snippets[:3], "count": len(retrieval.snippets)}, ensure_ascii=True)
        )
        lines.append("Thought: I should estimate risk using image-derived issues and safety context.")
        lines.append("Action: RiskEvaluator")
        lines.append(
            "Observation: "
            + json.dumps(
                {
                    "risk_score": risk["risk_score"],
                    "severity": risk["severity"],
                    "rationale": risk["rationale"],
                },
                ensure_ascii=True,
            )
        )

        lines.append("")
        lines.append("Final Answer:")
        lines.append("Structured Image-Based Surgical Insight")
        lines.append(f"- Source Image: {str(image_path)}")
        lines.append(f"- Likely Procedure: {procedure}")
        lines.append(f"- Likely Operative Step: {step}")
        lines.append(f"- Detected Organs: {vision_findings.get('organs', [])}")
        lines.append(f"- Detected Instruments: {vision_findings.get('instruments', [])}")
        lines.append(f"- Concerning Findings: {vision_findings.get('issues', [])}")
        lines.append(f"- Image Quality: {image_findings.get('image_quality', 'unknown')}")
        lines.append(f"- Risk Score: {risk['risk_score']}")
        lines.append(f"- Risk Level: {risk['severity']}")
        lines.append("- Immediate Recommended Actions:")
        for action in immediate_actions:
            lines.append(f"  - {action}")
        lines.append("- Missing / Uncertain:")
        for uncertainty in image_findings.get("uncertainties", []) or ["No explicit uncertainties returned by model."]:
            lines.append(f"  - {uncertainty}")
        lines.append("- Retrieved Evidence (RAG):")
        for snip in retrieval.snippets[:3]:
            lines.append(f"  - {snip}")
        lines.append(f"- Documentation Summary: {image_findings.get('summary', 'No summary provided.')}")
        lines.append("- Safety Note: This assistant supports clinical reasoning and does not replace surgeon judgment.")

        return "\n".join(lines)


def load_payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="AI Surgical Assistant Agent (LangChain + RAG)")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input",
        help="Path to JSON file containing the surgical case payload.",
    )
    source_group.add_argument(
        "--image",
        help="Path to an operative image to analyze with Groq vision model.",
    )
    parser.add_argument(
        "--knowledge",
        default="medical_knowledge_base.txt",
        help="Path to local medical knowledge text file.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        help="Groq model for image analysis mode.",
    )
    args = parser.parse_args()

    knowledge_path = Path(args.knowledge)
    agent = SurgicalAssistantAgent(knowledge_path=knowledge_path)

    groq_key = os.getenv("GROQ_API_KEY", "")

    if args.image:
        if not groq_key:
            raise ValueError("GROQ_API_KEY is required for --image mode. Add it to your .env file.")
        image_path = Path(args.image)
        vision = GroqVisionAnalyzer(api_key=groq_key, model=args.model)
        image_findings = vision.run(image_path)
        report = agent.run_from_image(image_findings=image_findings, image_path=image_path)
    else:
        input_path = Path(args.input)
        payload = load_payload(input_path)
        report = agent.run(payload)
        if not groq_key:
            print("Warning: GROQ_API_KEY is not set. Running in local-only mode.")

    print(report)


if __name__ == "__main__":
    main()
