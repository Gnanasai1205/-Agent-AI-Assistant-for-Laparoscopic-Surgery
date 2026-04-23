"""
report_pdf.py — PDF report generator for Laparoscopic Surgical Assistant.
Requires: pip install fpdf2
"""
from __future__ import annotations

from datetime import datetime

try:
    from fpdf import FPDF

    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


# ─── Low-level helpers ────────────────────────────────────────────────────────

def _section_header(pdf: "FPDF", title: str) -> None:
    pdf.set_fill_color(18, 74, 102)
    pdf.set_text_color(234, 246, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)


def _bullet(pdf: "FPDF", text: str) -> None:
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(12)
    pdf.multi_cell(0, 5.5, f"-  {text}", new_x="LMARGIN", new_y="NEXT")


def _kv(pdf: "FPDF", key: str, value: str) -> None:
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(55, 6, f"{key}:", new_x="RIGHT", new_y="LAST")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")


# ─── PDF subclass ─────────────────────────────────────────────────────────────

class _SurgicalPDF(FPDF):
    def header(self):
        self.set_fill_color(8, 28, 45)
        self.rect(0, 0, 210, 28, "F")
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(234, 246, 255)
        self.set_xy(0, 7)
        self.cell(0, 8, "LAPAROSCOPIC SURGICAL ASSISTANT", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(195, 229, 245)
        self.cell(0, 6, "Intraoperative Clinical Decision Support Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(8)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(128)
        self.cell(
            0, 10,
            f"Page {self.page_no()}  |  DISCLAIMER: For clinical decision support only - does not replace surgeon judgment.",
            align="C",
        )


# ─── Report parser ────────────────────────────────────────────────────────────

def _parse_report(report: str) -> dict:
    result: dict = {
        "metadata": [],
        "actions": [],
        "escalation": [],
        "evidence": [],
        "safety_note": "",
        "raw": report,
    }
    in_final = False
    current: str | None = None

    for line in report.splitlines():
        stripped = line.strip()
        if stripped == "Final Answer:":
            in_final = True
            continue
        if not in_final or not stripped:
            continue

        # Section markers
        if stripped.startswith("- Recommended Immediate Actions"):
            current = "actions"
            continue
        elif stripped.startswith("- Escalation Guidance"):
            current = "escalation"
            continue
        elif stripped.startswith("- Retrieved Evidence"):
            current = "evidence"
            continue
        elif stripped.startswith("- Safety Note:"):
            result["safety_note"] = stripped[len("- Safety Note:"):].strip()
            current = None
            continue
        elif stripped.startswith("- ") and not line.startswith("  "):
            current = None
            result["metadata"].append(stripped[2:].strip())
            continue
        elif line.startswith("  ") and stripped.startswith("- "):
            text = stripped[2:].strip()
            if current in result:
                result[current].append(text)

    return result


# ─── Public API ───────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Replace common unsupported Unicode characters with ASCII equivalents."""
    replacements = {
        "—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "…": "...",
        "⚠️": "!", "•": "-", "✔": "v", "❌": "x", "°": " deg"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Strip any remaining non-Latin1 chars by encoding with 'ignore'
    return text.encode('latin-1', 'ignore').decode('latin-1')

def generate_pdf_report(
    report: str,
    mode: str,
    extra_meta: dict | None = None,
) -> bytes | None:
    """
    Generate a formatted PDF surgical report.
    Returns None if fpdf2 is not installed.
    """
    if not FPDF_AVAILABLE:
        return None

    report = _clean_text(report)
    parsed = _parse_report(report)
    pdf = _SurgicalPDF("P", "mm", "A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── Report info ───────────────────────────────────────────────────────────
    _section_header(pdf, "Report Information")
    _kv(pdf, "Generated", datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
    _kv(pdf, "Mode", mode)
    if extra_meta:
        for k, v in extra_meta.items():
            _kv(pdf, k, str(v))
    pdf.ln(3)

    # ── Case summary ──────────────────────────────────────────────────────────
    _section_header(pdf, "Case Summary")
    for item in parsed["metadata"]:
        if ":" in item:
            k, v = item.split(":", 1)
            _kv(pdf, k.strip(), v.strip())
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, item, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ── Risk highlight ────────────────────────────────────────────────────────
    risk_score = 0.0
    severity = "UNKNOWN"
    for item in parsed["metadata"]:
        lo = item.lower()
        if lo.startswith("risk score:"):
            try:
                risk_score = float(item.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif lo.startswith("risk level:"):
            severity = item.split(":", 1)[1].strip().upper()

    color_map = {
        "LOW": (107, 217, 168),
        "MODERATE": (255, 209, 102),
        "HIGH": (255, 127, 80),
    }
    r, g, b = color_map.get(severity, (200, 200, 200))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(
        0, 11,
        f"  RISK LEVEL: {severity}   |   Score: {risk_score:.2f} / 1.00",
        fill=True, align="C", new_x="LMARGIN", new_y="NEXT",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Recommended actions ───────────────────────────────────────────────────
    if parsed["actions"]:
        _section_header(pdf, "Recommended Immediate Actions")
        for action in parsed["actions"]:
            _bullet(pdf, action)
        pdf.ln(3)

    # ── Escalation guidance ───────────────────────────────────────────────────
    if parsed["escalation"]:
        _section_header(pdf, "Escalation Guidance")
        for g_item in parsed["escalation"]:
            _bullet(pdf, g_item)
        pdf.ln(3)

    # ── RAG evidence ──────────────────────────────────────────────────────────
    if parsed["evidence"]:
        _section_header(pdf, "Retrieved Clinical Evidence (RAG)")
        for i, snip in enumerate(parsed["evidence"], 1):
            pdf.set_x(10)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 5, f"[{i}]", new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(16)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(30, 60, 90)
            pdf.multi_cell(0, 5, snip, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        pdf.ln(2)

    # ── Full reasoning trace (new page) ──────────────────────────────────────
    pdf.add_page()
    _section_header(pdf, "Full Reasoning Trace  (ReAct Agent Log)")
    pdf.set_font("Courier", "", 7.5)
    pdf.set_text_color(40, 40, 40)
    for line in report.splitlines():
        pdf.multi_cell(0, 4, line if line.strip() else " ", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Safety note ───────────────────────────────────────────────────────────
    if parsed["safety_note"]:
        pdf.set_fill_color(255, 243, 205)
        pdf.set_text_color(100, 60, 0)
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(
            0, 8,
            f"  !  {parsed['safety_note']}",
            fill=True, new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_text_color(0, 0, 0)

    return bytes(pdf.output())
