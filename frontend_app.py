from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from surgical_assistant_agent import VisionAnalyzer, SurgicalAssistantAgent

try:
    from report_pdf import generate_pdf_report
    PDF_SUPPORTED = True
except ImportError:
    PDF_SUPPORTED = False

import io
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

st.set_page_config(
    page_title="Laparoscopic Surgical Assistant",
    page_icon="🔬",
    layout="wide",
)


# ─── Theme ────────────────────────────────────────────────────────────────────

def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg-deep: #081c2d;
            --bg-accent: #124a66;
            --panel: rgba(255, 255, 255, 0.09);
            --text-main: #eaf6ff;
            --text-soft: #c3e5f5;
            --line: rgba(195, 229, 245, 0.25);
            --danger: #ff7f50;
            --ok: #6bd9a8;
        }

        .stApp {
            background:
                radial-gradient(1200px 600px at 8% -5%, #1d6c8c 0%, rgba(29,108,140,0.15) 38%, transparent 70%),
                radial-gradient(900px 500px at 98% 2%, #0c8e7b 0%, rgba(12,142,123,0.12) 32%, transparent 64%),
                linear-gradient(145deg, var(--bg-deep) 0%, var(--bg-accent) 100%);
            color: var(--text-main);
            font-family: 'Space Grotesk', sans-serif;
        }

        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }

        h1, h2, h3 { color: var(--text-main) !important; letter-spacing: 0.2px; }

        .caption-soft {
            color: var(--text-soft);
            font-size: 0.95rem;
            margin-top: -0.25rem;
        }

        .metric-card {
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: var(--panel);
            backdrop-filter: blur(6px);
            margin-bottom: 0.8rem;
        }

        .metric-label { color: var(--text-soft); font-size: 0.8rem; margin-bottom: 0.1rem; }
        .metric-value { font-size: 1.05rem; font-weight: 600; color: var(--text-main); }

        .severity-low      { color: var(--ok);     font-weight: 700; }
        .severity-moderate { color: #ffd166;       font-weight: 700; }
        .severity-high     { color: var(--danger); font-weight: 700; }

        div[data-testid="stSidebar"] {
            background: rgba(4, 18, 30, 0.78);
            border-right: 1px solid var(--line);
        }

        .stCodeBlock, pre, code {
            font-family: 'IBM Plex Mono', monospace !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─── Risk gauge ───────────────────────────────────────────────────────────────

def render_risk_gauge(risk_score: float, severity: str) -> None:
    color_map = {"LOW": "#6bd9a8", "MODERATE": "#ffd166", "HIGH": "#ff7f50"}
    color = color_map.get(severity.strip().upper(), "#aaaaaa")
    arc_length = 251.3
    filled = min(max(risk_score, 0.0), 1.0) * arc_length
    pct = int(min(max(risk_score, 0.0), 1.0) * 100)

    gauge_html = f"""
    <div style="display:flex;justify-content:center;padding:0.5rem 0 1rem 0;">
        <div style="text-align:center;">
            <svg width="210" height="120" viewBox="0 0 210 120">
                <!-- Background track -->
                <path d="M 25 105 A 80 80 0 0 1 185 105"
                      stroke="rgba(255,255,255,0.12)" stroke-width="18" fill="none" stroke-linecap="round"/>
                <!-- Filled arc -->
                <path d="M 25 105 A 80 80 0 0 1 185 105"
                      stroke="{color}" stroke-width="18" fill="none" stroke-linecap="round"
                      stroke-dasharray="{filled:.1f} {arc_length:.1f}"/>
                <!-- Percentage -->
                <text x="105" y="90" text-anchor="middle"
                      fill="{color}" font-family="Space Grotesk,sans-serif"
                      font-size="28" font-weight="700">{pct}%</text>
                <!-- Label -->
                <text x="105" y="110" text-anchor="middle"
                      fill="{color}" font-family="Space Grotesk,sans-serif"
                      font-size="13" font-weight="600" letter-spacing="2">{severity.upper()}</text>
            </svg>
            <div style="color:rgba(195,229,245,0.65);font-size:0.8rem;margin-top:0.1rem;">
                Risk Score: {risk_score:.2f} / 1.00
            </div>
        </div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_risk_info(report: str) -> tuple[float, str]:
    """Parse risk score and severity level from the structured report string."""
    risk_score, severity = 0.0, "UNKNOWN"
    for line in report.splitlines():
        if line.startswith("- Risk Score:"):
            try:
                risk_score = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("- Risk Level:"):
            severity = line.split(":", 1)[1].strip()
    return risk_score, severity

def render_audio_alert(report: str, severity: str) -> None:
    """Generates an MP3 voice alert of the recommended actions."""
    if not GTTS_AVAILABLE:
        return
        
    actions = []
    in_actions = False
    for line in report.splitlines():
        if line.startswith("- Recommended Immediate Actions"):
            in_actions = True
        elif line.startswith("- Escalation") or (line.startswith("- ") and not line.startswith("  -") and not line.startswith("  ")):
            if in_actions and len(actions) > 0:
                in_actions = False
        elif in_actions:
            clean_line = line.strip()
            if clean_line.startswith("- "):
                actions.append(clean_line[2:])
            elif clean_line:
                actions.append(clean_line)
                
    if not actions:
        return
        
    text_to_speak = f"Risk Level is {severity}. Immediate recommendations are: " + ". ".join(actions)
    
    with st.spinner("Generating voice alert\u2026"):
        try:
            mp3_fp = io.BytesIO()
            tts = gTTS(text=text_to_speak, lang='en', slow=False)
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            st.markdown("🔉 **Audio Alert**")
            # Autoplay if HIGH risk
            st.audio(mp3_fp.read(), format="audio/mp3", autoplay=(severity=="HIGH"))
        except Exception as e:
            st.warning(f"Voice alert failed: {e}")

def build_txt_report(report: str, mode: str, extra_meta: dict | None = None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        "  LAPAROSCOPIC SURGICAL ASSISTANT \u2014 INTRAOPERATIVE REPORT",
        "=" * 70,
        f"  Generated : {now}",
        f"  Mode      : {mode}",
    ]
    if extra_meta:
        for k, v in extra_meta.items():
            lines.append(f"  {k:<10}: {v}")
    lines += [
        "=" * 70,
        "",
        report,
        "",
        "=" * 70,
        "  DISCLAIMER: This report is intended to support clinical reasoning",
        "  only and does NOT replace the judgment of a qualified surgeon.",
        "=" * 70,
    ]
    return "\n".join(lines)


def render_report_downloads(report: str, mode: str, extra_meta: dict | None = None) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_content = build_txt_report(report, mode, extra_meta)

    dl_cols = st.columns(2) if PDF_SUPPORTED else st.columns(1)

    with dl_cols[0]:
        st.download_button(
            label="Download Report (.txt)",
            data=txt_content,
            file_name=f"surgical_report_{timestamp}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if PDF_SUPPORTED:
        pdf_bytes = generate_pdf_report(report, mode=mode, extra_meta=extra_meta)
        if pdf_bytes:
            with dl_cols[1]:
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"surgical_report_{timestamp}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


# ─── Agent cache ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RAG model (first run downloads ~90 MB)…", hash_funcs={str: str})
def get_agent(knowledge_path: str, _force_reload: int = 2) -> SurgicalAssistantAgent:
    return SurgicalAssistantAgent(Path(knowledge_path))


# ─── UI helpers ───────────────────────────────────────────────────────────────

def severity_class(level: str) -> str:
    norm = (level or "").strip().upper()
    return {"LOW": "severity-low", "MODERATE": "severity-moderate", "HIGH": "severity-high"}.get(norm, "")


def render_header() -> None:
    st.title("Laparoscopic Surgical Assistant")
    st.markdown(
        '<p class="caption-soft">Image and case-based intraoperative support with RAG-grounded risk framing.</p>',
        unsafe_allow_html=True,
    )


def render_summary_cards(image_findings: dict) -> None:
    c1, c2, c3 = st.columns(3)
    for col, label, key in [
        (c1, "Likely Procedure", "procedure_guess"),
        (c2, "Likely Step", "operative_step_guess"),
        (c3, "Image Quality", "image_quality"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{image_findings.get(key, "unknown")}</div></div>',
                unsafe_allow_html=True,
            )


# ─── Tab: Image Upload ────────────────────────────────────────────────────────

def run_image_mode(agent: SurgicalAssistantAgent, model_name: str) -> None:
    st.subheader("Analyze Uploaded Image & Vitals")
    
    col_img, col_vitals = st.columns([3, 2])
    
    with col_img:
        image_file = st.file_uploader("Upload laparoscopic frame", type=["jpg", "jpeg", "png", "webp"])
        if image_file:
            st.image(image_file, caption=image_file.name, use_container_width=True)

    with col_vitals:
        st.markdown("#### Patient Vitals")
        bp_systolic = st.slider("BP Systolic (mmHg)", 60, 200, 120, key="img_bp")
        heart_rate = st.slider("Heart Rate (bpm)", 40, 200, 78, key="img_hr")
        spo2 = st.slider("SpO₂ (%)", 75, 100, 98, key="img_spo2")
        
        vitals_ok = bp_systolic >= 90 and heart_rate <= 120 and spo2 >= 92
        status_color = "#6bd9a8" if vitals_ok else "#ff7f50"
        status_text = "Stable Vitals" if vitals_ok else "Abnormal Vitals"
        st.markdown(
            f'<div class="metric-card" style="border-color:{status_color}55;">'
            f'<div class="metric-label">Vitals Status</div>'
            f'<div class="metric-value" style="color:{status_color};">{status_text}</div>'
            f'<div style="font-size:0.8rem;color:var(--text-soft);margin-top:0.4rem;">'
            f"BP: {bp_systolic} mmHg &nbsp;|&nbsp; HR: {heart_rate} bpm &nbsp;|&nbsp; SpO₂: {spo2}%"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    if image_file is None:
        st.info("Upload an operative frame to generate a full AI-powered report.")
        return

    st.markdown("---")

    if st.button("Generate Full Image Report", type="primary", use_container_width=True):
        vitals = {"bp_systolic": bp_systolic, "hr": heart_rate, "spo2": spo2}
        groq_key = os.getenv("GROQ_API_KEY", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if not groq_key and not gemini_key:
            st.error("Neither GROQ_API_KEY nor GEMINI_API_KEY is present in your .env file.")
            return

        suffix = Path(image_file.name).suffix or ".jpg"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_file.getbuffer())
                temp_path = Path(tmp.name)

            with st.spinner("Analyzing image with Vision AI\u2026"):
                vision = VisionAnalyzer(gemini_key=gemini_key, groq_key=groq_key, model=model_name)
                image_findings = vision.run(temp_path)
                report = agent.run_from_image(
                    image_findings=image_findings, image_path=Path(image_file.name), vitals=vitals
                )

            render_summary_cards(image_findings)

            st.markdown("### Findings")
            st.write({
                "organs": image_findings.get("organs", []),
                "instruments": image_findings.get("instruments", []),
                "issues": image_findings.get("issues", []),
                "uncertainties": image_findings.get("uncertainties", []),
            })

            risk_score, severity = extract_risk_info(report)
            st.markdown("### Risk Assessment")
            render_risk_gauge(risk_score, severity)
            render_audio_alert(report, severity)

            st.markdown("### Full Structured Report")
            st.code(report)

            pdf_bytes = generate_pdf_report(report, mode="Image Analysis", extra_meta={"Image": image_file.name, "Model": model_name}) if PDF_SUPPORTED else None
            
            if "cases_history" not in st.session_state:
                st.session_state["cases_history"] = []
            st.session_state["cases_history"].append({
                "mode": "Image Analysis",
                "time": datetime.now().strftime("%H:%M:%S"),
                "severity": severity,
                "score": risk_score,
                "pdf_bytes": pdf_bytes
            })

            render_report_downloads(
                report,
                mode="Image Analysis",
                extra_meta={"Image": image_file.name, "Model": model_name},
            )

        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)


# ─── Tab: Form Input ──────────────────────────────────────────────────────────

_PROCEDURES = [
    "Laparoscopic Cholecystectomy",
    "Laparoscopic Appendectomy",
    "Laparoscopic Inguinal Hernia Repair (TAPP)",
    "Laparoscopic Inguinal Hernia Repair (TEP)",
    "Laparoscopic Sleeve Gastrectomy",
    "Laparoscopic Roux-en-Y Gastric Bypass",
    "Laparoscopic Sigmoid Colectomy",
    "Laparoscopic Right Hemicolectomy",
    "Laparoscopic Nissen Fundoplication",
    "Laparoscopic Splenectomy",
    "Diagnostic Laparoscopy",
]

_OPERATIVE_STEPS = [
    "Initial port placement and pneumoperitoneum",
    "Adhesiolysis and exposure",
    "Calot's Triangle Dissection (CVS establishment)",
    "Clipping and division of cystic duct and artery",
    "Gallbladder dissection from liver bed",
    "Mesoappendix division",
    "Appendix base ligation and division",
    "Mesh placement and fixation",
    "Stapled anastomosis creation",
    "Specimen retrieval and port closure",
    "Irrigation and hemostasis check",
]

_ORGANS = [
    "Liver", "Gallbladder", "Bile ducts", "Stomach", "Duodenum",
    "Small bowel", "Large bowel / Colon", "Appendix", "Spleen",
    "Pancreas", "Kidney", "Ovary / Uterus", "Bladder",
]

_INSTRUMENTS = [
    "Maryland dissector", "Hook cautery (monopolar)",
    "Harmonic scalpel (ultrasonic)", "Bipolar forceps",
    "Clip applier", "Laparoscopic scissors",
    "Atraumatic grasper", "Stapler (Endo-GIA)",
    "Suction-irrigation", "Needle driver", "Retrieval bag",
]

_ISSUES = [
    "Unclear anatomy / Critical View of Safety not achieved",
    "Active bleeding / hemorrhage",
    "Bile leak suspected",
    "Thermal spread risk",
    "Bowel injury suspected",
    "Port-site bleeding",
    "Dense adhesions obscuring view",
    "Subcutaneous emphysema",
    "Low visibility / smoke",
    "Instrument malfunction",
]

_COMPLICATIONS = [
    "Hemorrhage", "Bile duct injury", "Bowel perforation",
    "Vascular injury", "Bile leak", "Thermal injury",
    "CO2 gas embolism", "Port-site hernia risk",
]


def run_form_mode(agent: SurgicalAssistantAgent) -> None:
    st.subheader("Form-Based Case Input")
    st.markdown(
        '<p class="caption-soft">Fill in the surgical case details using dropdowns and sliders — no JSON required.</p>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([3, 2])

    with col_left:
        procedure = st.selectbox("Procedure", _PROCEDURES)
        operative_step = st.selectbox("Operative Step", _OPERATIVE_STEPS)
        organs = st.multiselect("Detected Organs", _ORGANS)
        instruments = st.multiselect("Detected Instruments", _INSTRUMENTS)
        issues = st.multiselect("⚠️  Detected Issues / Concerns", _ISSUES)
        complications = st.multiselect("Suspected Complications", _COMPLICATIONS)

    with col_right:
        st.markdown("#### Patient Vitals")
        bp_systolic = st.slider("BP Systolic (mmHg)", 60, 200, 120)
        heart_rate = st.slider("Heart Rate (bpm)", 40, 200, 78)
        spo2 = st.slider("SpO₂ (%)", 75, 100, 98)

        vitals_ok = bp_systolic >= 90 and heart_rate <= 120 and spo2 >= 92
        status_color = "#6bd9a8" if vitals_ok else "#ff7f50"
        status_text = "✓ Stable Vitals" if vitals_ok else "⚠ Abnormal Vitals"
        st.markdown(
            f'<div class="metric-card" style="border-color:{status_color}55;">'
            f'<div class="metric-label">Vitals Status</div>'
            f'<div class="metric-value" style="color:{status_color};">{status_text}</div>'
            f'<div style="font-size:0.8rem;color:var(--text-soft);margin-top:0.4rem;">'
            f"BP: {bp_systolic} mmHg &nbsp;|&nbsp; HR: {heart_rate} bpm &nbsp;|&nbsp; SpO₂: {spo2}%"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if st.button("Generate Surgical Report", type="primary", use_container_width=True):
        payload = {
            "procedure": procedure,
            "operative_step": operative_step,
            "vitals": {"bp_systolic": bp_systolic, "hr": heart_rate, "spo2": spo2},
            "vision_findings": {
                "organs": organs,
                "instruments": instruments,
                "issues": [i.split("/")[0].strip() for i in issues],  # normalize
            },
            "suspected_complications": complications,
        }

        try:
            with st.spinner("Running RAG retrieval and risk analysis…"):
                report = agent.run(payload)

            risk_score, severity = extract_risk_info(report)
            st.markdown("### Risk Assessment")
            render_risk_gauge(risk_score, severity)
            render_audio_alert(report, severity)

            st.markdown("### Full Structured Report")
            st.code(report)

            pdf_bytes = generate_pdf_report(report, mode="Form Input", extra_meta={"Procedure": procedure, "Step": operative_step}) if PDF_SUPPORTED else None
            
            if "cases_history" not in st.session_state:
                st.session_state["cases_history"] = []
            st.session_state["cases_history"].append({
                "mode": "Form Input",
                "time": datetime.now().strftime("%H:%M:%S"),
                "severity": severity,
                "score": risk_score,
                "pdf_bytes": pdf_bytes
            })

            render_report_downloads(
                report,
                mode="Form Input",
                extra_meta={"Procedure": procedure, "Step": operative_step},
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")


# ─── Tab: Chat Q&A ────────────────────────────────────────────────────────────


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    apply_custom_theme()
    render_header()

    with st.sidebar:
        st.header("Configuration")
        knowledge_path = st.text_input("Knowledge base file", value="medical_knowledge_base.txt")
        model_name = st.text_input(
            "Vision model Override (optional)",
            value="",
            placeholder="e.g. gemini-2.5-flash or meta-llama/llama-4-scout-17b-16e-instruct",
        )
        st.caption("Supports GROQ_API_KEY or GEMINI_API_KEY in .env.")
        st.markdown("---")
        if PDF_SUPPORTED:
            st.success("PDF reports enabled")
        else:
            st.warning("PDF reports disabled — `pip install fpdf2`")
        if st.session_state.get("cases_history"):
            st.markdown("---")
            st.subheader("Recent Cases")
            for i, case in enumerate(reversed(st.session_state["cases_history"])):
                with st.expander(f"Case {len(st.session_state['cases_history']) - i}: {case['mode']}"):
                    st.caption(f"Time: {case['time']}")
                    st.markdown(f"**Risk:** {case['severity']} ({case['score']})")
                    st.download_button(
                        label="Download PDF",
                        data=case["pdf_bytes"],
                        file_name=f"case_history_{i}.pdf",
                        mime="application/pdf",
                        key=f"hist_dl_{i}"
                    ) if case.get("pdf_bytes") else None

    try:
        agent = get_agent(knowledge_path, _force_reload=4)
    except Exception as exc:
        st.error(f"Failed to initialize agent: {exc}")
        return

    tab_form, tab_image = st.tabs(
        ["Form Input", "Image Upload"]
    )

    with tab_form:
        run_form_mode(agent)
    with tab_image:
        run_image_mode(agent, model_name)
   


if __name__ == "__main__":
    main()
