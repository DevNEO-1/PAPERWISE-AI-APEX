"""
PAPERWISE AI APEX - Report Generator Module
Compiles structured PaperAnalysis into a publication-grade, beautifully formatted
Markdown report suitable for academic review, sharing, and assignment submission.
"""

from datetime import datetime
from typing import Optional

from config import config
from gemini_service import PaperAnalysis
from pdf_processor import ExtractedDocument


def generate_markdown_report(
    analysis: PaperAnalysis,
    document: ExtractedDocument,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate an authoritative, publication-grade Markdown document from
    the structured analysis and document metadata.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    authors_str = ", ".join(analysis.authors) if analysis.authors else "Not mentioned in the paper"
    actual_model = getattr(analysis, "model_used", None) or model_name or config.model_name
    is_fallback = getattr(analysis, "is_fallback_used", False) or (
        actual_model and actual_model != config.model_name and not str(actual_model).startswith("Omitted")
    )
    stage_models = getattr(analysis, "stage_models", {})

    def badge(is_explicit: bool) -> str:
        return "**[Explicit Author Evidence]**" if is_explicit else "**[Contextual Interpretation]**"

    overview_rows = [
        f"| **Paper Title** | **{analysis.paper_title}** |",
        f"| **Authors** | {authors_str} |",
        f"| **Source File** | `{document.filename}` |",
        f"| **Pages Analyzed** | {document.total_pages} |",
        f"| **Extracted Words** | {document.total_words:,} |",
        f"| **AI Engine Used** | `{actual_model}` |",
    ]
    if is_fallback:
        overview_rows.append(f"| **Model Status** | ⚠️ *Verified Fallback Active* (Configured: `{config.model_name}`) |")
    overview_rows.append(f"| **Generated At** | {timestamp} |")

    report_lines = [
        f"# 📑 {config.app_name} — Research Intelligence Report",
        f"",
        f"> **{config.tagline}**  ",
        f"> *{config.subtagline}*",
        f"",
        f"---",
        f"",
        f"## 📌 Document Overview",
        f"",
        f"| Field | Details |",
        f"| :--- | :--- |",
    ] + overview_rows + [
        f"",
        f"---",
        f"",
        f"## 🎯 1. Problem Statement",
        f"",
        f"{badge(analysis.problem.is_explicit)}  ",
        f"*Evidence Source: `{analysis.problem.evidence_pages}`*",
        f"",
        f"{analysis.problem.content}",
        f"",
        f"---",
        f"",
        f"## 🔬 2. Methodology & Approach",
        f"",
        f"{badge(analysis.method.is_explicit)}  ",
        f"*Evidence Source: `{analysis.method.evidence_pages}`*",
        f"",
        f"{analysis.method.content}",
        f"",
        f"---",
        f"",
        f"## 📊 3. Dataset & Experimental Setup",
        f"",
        f"{badge(analysis.dataset.is_explicit)}  ",
        f"*Evidence Source: `{analysis.dataset.evidence_pages}`*",
        f"",
        f"{analysis.dataset.content}",
        f"",
        f"---",
        f"",
        f"## 📈 4. Key Results & Findings",
        f"",
        f"{badge(analysis.results.is_explicit)}  ",
        f"*Evidence Source: `{analysis.results.evidence_pages}`*",
        f"",
        f"{analysis.results.content}",
        f"",
        f"---",
        f"",
        f"## ⚠️ 5. Limitations & Boundaries",
        f"",
        f"{badge(analysis.limitations.is_explicit)}  ",
        f"*Evidence Source: `{analysis.limitations.evidence_pages}`*",
        f"",
        f"{analysis.limitations.content}",
        f"",
        f"---",
        f"",
        f"## 💡 6. Five Key Takeaways",
        f"",
    ]

    for item in analysis.takeaways:
        card_num = f"{item.number:02d}"
        report_lines.extend([
            f"### **[{card_num}] {item.title}**",
            f"- **Summary:** {item.summary}",
            f"- **Citation:** `{item.page_reference}`",
            f"",
        ])

    report_lines.extend([
        f"---",
        f"",
        f"## 🔍 7. Academic Integrity & Disclosures",
        f"",
        f"- **Verification Policy:** All claims are strictly grounded in extracted document text without external speculation.",
        f"- **Completeness & Pipeline Disclosures:** {analysis.disclosure_notes}",
        f"",
    ])

    if stage_models and len(stage_models) > 0:
        report_lines.extend([
            f"### 🤖 Processing Pipeline & Model Attribution",
            f"",
            f"| Processing Stage | Model Utilized | Attribution Status |",
            f"| :--- | :--- | :--- |",
        ])
        for stg_name, mdl in stage_models.items():
            stg_status = (
                "Not processed (API capacity)" if str(mdl).startswith("Omitted")
                else ("Configured Primary" if mdl == config.model_name
                      else f"Verified Fallback (Configured: {config.model_name})")
            )
            report_lines.append(f"| **{stg_name}** | `{mdl}` | {stg_status} |")
        report_lines.append("")

    report_lines.extend([
        f"---",
        f"",
        f"*Report generated by **{config.app_name}** • Advanced AI Research Intelligence.*",
    ])

    return "\n".join(report_lines)
