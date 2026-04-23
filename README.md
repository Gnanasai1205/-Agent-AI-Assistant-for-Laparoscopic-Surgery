# AI Surgical Assistant for Laparoscopic Surgery

This project implements a LangChain-based surgical assistant agent using a RAG workflow.

Implemented tools:
- MedicalRetriever: retrieves relevant intraoperative safety context from a local knowledge base
- VisionAnalyzer: parses provided vision detections (organs, instruments, issues)
- RiskEvaluator: computes risk score and severity class

Safety constraints enforced:
- MedicalRetriever is always called before explanation
- If context is missing, the output is exactly: Insufficient medical data
- Risk levels follow:
  - 0.0-0.3 LOW
  - 0.3-0.7 MODERATE
  - 0.7-1.0 HIGH

## Setup

```bash
pip install -r requirements.txt
```

## Frontend (Streamlit)

Run the web UI:

```bash
streamlit run frontend_app.py
```

Frontend includes:
- Image Upload tab: upload laparoscopic frame and get full structured details
- JSON Case tab: paste/edit a case payload and generate report
- Styled risk display with procedure/step summary cards

## API key setup (Groq)

1. Copy `.env.example` to `.env`.
2. Add your Groq key in `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Notes:
- `.env` is ignored by git to avoid leaking secrets.
- If `GROQ_API_KEY` is missing, the app still runs in local-only mode.

## Run

```bash
python surgical_assistant_agent.py --input sample_case.json --knowledge medical_knowledge_base.txt
```

### Image mode (upload frame and get detailed analysis)

```bash
python surgical_assistant_agent.py --image path/to/frame.jpg --knowledge medical_knowledge_base.txt
```

Optional model override:

```bash
python surgical_assistant_agent.py --image path/to/frame.jpg --model meta-llama/llama-4-scout-17b-16e-instruct
```

Image mode returns structured details including:
- visible organs and instruments
- likely procedure and operative step
- concerning findings
- risk score and risk level
- immediate recommended actions
- uncertain/missing details
- concise documentation summary

## Expected output style

The agent prints iterative reasoning traces in this format:
- Thought
- Action
- Action Input
- Observation

Then prints:
- Final Answer with structured surgical insight and actionable recommendations.

## Input schema

```json
{
  "procedure": "string",
  "operative_step": "string",
  "vitals": {
    "bp_systolic": 0,
    "hr": 0,
    "spo2": 0
  },
  "vision_findings": {
    "organs": ["..."],
    "instruments": ["..."],
    "issues": ["..."]
  },
  "suspected_complications": ["..."]
}
```
