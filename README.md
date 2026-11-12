# 🔬 PAPERWISE AI APEX — AI Research Intelligence Studio

> **"Research deserves better than a skim."**  
> *Transform complex academic knowledge into clarity.*

---

## 🌟 Overview

**PAPERWISE AI APEX** is a luxury, production-grade AI research intelligence workstation designed to synthesize dense academic papers (PDF) into verified, structured research dossiers with page-level citations and rigorous document grounding.

### Key Highlights:
- **Server-Side Zero-Leakage Security:** The frontend contains zero API key inputs or credential fields. All API keys are loaded strictly server-side via `.env` or Streamlit secrets.
- **Single Centralized Model Engine & Active Fallback Cascade:** Powered by **Google Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`) as primary, backed by an active, supported fallback cascade:
  1. `gemini-3.5-flash-lite` (Configured default primary engine)
  2. `gemini-3.8-flash` (Primary recommended high-intelligence Flash model)
  3. `gemini-3.1-flash-lite` (Lightweight, low-latency, cost-efficient fallback)
  4. `gemini-3.1-pro-preview` (Frontier Pro reasoning fallback)
  - **Project Fallback Policy:** The fallback cascade intentionally excludes Gemini 1.5, 2.0, and 2.5 model families. This is a project allowlist decision; it does not imply that every Gemini 2.5 model is officially shut down or deprecated. Check Google's current model documentation before changing the allowlist.
- **Obsidian & Electric Violet Aesthetics:** Built with a custom midnight navy/obsidian glassmorphic interface, electric violet and icy cyan accents, and scientific orbital animations.
- **One-Click Sample Paper:** Built-in sample academic research paper button allows instant demonstration without needing local PDFs.
- **Page-Level Grounding:** Injects explicit `[START PAGE X]` page markers, enabling verifiable citations across all findings.
- **Scanned PDF & OCR Detection:** Intelligently identifies image-only scanned PDFs and instructs the user on OCR pre-processing.
- **Five Mandatory Research Dimensions:** Problem Statement, Methodology & Architecture, Dataset & Setup, Key Results & Metrics, and Stated vs. Inferred Limitations.
- **Five Numbered Key Takeaways:** Crisp, punchy takeaway cards numbered `01` through `05`.
- **Publication-Grade Markdown Export:** Single-click download of the complete structured intelligence report.

---

## 🏛️ System Architecture

```
PAPERWISE-AI/
├── app.py                      # Main Streamlit APEX workstation & UI layout
├── config.py                   # Centralized configuration, boundaries & secrets
├── pdf_processor.py            # PyPDF extraction, validation & scanned detection
├── gemini_service.py           # Google Gemini API integration (google-genai SDK)
├── report_generator.py         # Markdown intelligence report compiler
├── requirements.txt            # Production dependencies (minimum-version requirements)
├── pytest.ini                  # Pytest configuration with pythonpath and test paths
├── .env.example                # Safe environment template with placeholders
├── .gitignore                  # Comprehensive secret and cache exclusion
├── README.md                   # Full documentation & assignment report
├── .streamlit/
│   └── config.toml             # Dark mode theme & telemetry configuration
├── sample_papers/
│   ├── sample_research_paper.pdf # Pre-validated 3-page synthetic CS paper
│   └── genuine_research_paper.pdf # Genuine 15-page arXiv paper (Attention Is All You Need)
├── scripts/
│   └── create_sample_paper.py  # Standalone utility to generate synthetic test research papers
└── tests/
    ├── test_pdf_processor.py   # PDF validation, boundary, header, scanned & arXiv tests
    ├── test_gemini_service.py  # Pydantic schemas, JSON healing, takeaway normalization & model cascade regression
    └── test_report_generator.py # Publication-grade Markdown report generation tests
```

---

## 📦 Dependency Version Strategy

The project specifies **minimum-version requirements** (`>=`) in `requirements.txt`:
```txt
streamlit>=1.35.0
google-genai>=2.0.0
pypdf>=4.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pytest>=8.0.0
```

### Rationale:
- **Flexibility & Security:** Using minimum versions prevents strict dependency pin conflicts when integrating into diverse Python environments while ensuring security patch availability.
- **Modern SDK Guarantee:** `google-genai>=2.0.0` ensures the official modern Google GenAI SDK is utilized (deprecating legacy `google-generativeai`).
- **Pydantic v2 Guarantee:** `pydantic>=2.0.0` guarantees robust type validation and JSON schema introspection.

**Verified Runtime Environment (Tested):**
- Python 3.14.7
- `streamlit==1.67.0`
- `google-genai==2.22.0`
- `pypdf==7.0.1`
- `python-dotenv==1.2.4`
- `pydantic==2.13.5`
- `pytest==9.2.0`

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.14)
- **Google Gemini API Key** (Free from [Google AI Studio](https://aistudio.google.com/))

### 2. Configure Your API Key (Server-Side Only)

Create or update your local `.env` file in the project root:

```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

> **Security Note:** The `.env` file is excluded from Git via `.gitignore`. The web application will never display, prompt for, or expose your API key.

### 3. Launch the Application

```bash
streamlit run app.py
```

The application will launch at:
```
http://localhost:8501
```

---

## 🧪 Automated Testing

Run the automated test suite with `pytest`:

```bash
pytest tests/ -v
```

All **16 automated unit tests** across 3 test modules execute directly against the genuine installed `google-genai` SDK without fake test shims:
1. `test_paper_analysis_pydantic_validation`: Validates Pydantic schema constraints.
2. `test_json_healing_and_takeaways_normalization`: Tests markdown code fence stripping and takeaway normalization to 5.
3. `test_stage_model_tracking_and_fallback_flag`: Verifies model telemetry across chunked stages.
4. `test_omitted_stage_is_not_counted_as_verified_fallback`: Ensures omissions are not misclassified as fallbacks.
5. `test_fallback_models_follow_project_allowlist`: **Regression test** enforcing the project's explicit exclusion policy for Gemini 1.5, 2.0, and 2.5 model families; this is not a claim that every excluded model is officially deprecated.
6. `test_empty_bytes_raises_malformed`: Ensures empty byte payloads are rejected.
7. `test_invalid_header_raises_malformed`: Ensures missing `%PDF` headers are caught.
8. `test_oversized_file_raises_error`: Enforces max file size boundaries.
9. `test_valid_header_passes`: Verifies clean PDF header acceptance.
10. `test_blank_scanned_pdf_raises_scanned_error`: Detects zero-text / scanned image PDFs.
11. `test_chunk_document_pages`: Tests multi-page chunk batching.
12. `test_document_full_text_page_markers`: Validates page marker injection (`[START PAGE X]`).
13. `test_sample_academic_paper_extraction`: Tests 3-page synthetic research paper extraction.
14. `test_genuine_arxiv_paper_extraction`: Tests 15-page genuine arXiv paper (Attention Is All You Need) extraction.
15. `test_markdown_report_generation_structure`: Validates structured Markdown report compilation.
16. `test_markdown_report_model_attribution_and_fallback_badges`: Verifies engine attribution badges and tables.

---

## 🔍 Live API Integration Status

> **Status: NOT VERIFIED**  
> *Reason:* No live `GEMINI_API_KEY` was configured in the execution environment. In accordance with strict verification and integrity guidelines, live API synthesis against the remote Gemini API was not performed and is therefore explicitly marked as **NOT VERIFIED**. Offline unit and regression tests using the genuine `google-genai` SDK and local PDF extraction verified 100% pass rate.

---

## 🛡️ Academic Integrity & Verification Guarantee

- **Strict Evidence Grounding:** Models run with low temperature (`0.2`) and strict document evidence constraints.
- **Absence Handling:** Unreported datasets, authors, or limitations are explicitly marked *"Not mentioned in the paper"*.
- **Numerical Precision:** Reported accuracy, F1-scores, latencies, and metrics are preserved exactly as stated in the text.
