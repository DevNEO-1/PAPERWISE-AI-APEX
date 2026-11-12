"""
PAPERWISE AI APEX — Research Intelligence Studio
A luxury, production-grade AI research paper synthesis workstation.
Built with Streamlit, Google Gemini (google-genai SDK), and PyPDF.
"""

import io
import html
import os
import re
import sys
from typing import Optional
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="PAPERWISE AI APEX — Research Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import config, get_secret, is_api_configured
from pdf_processor import (
    ExtractedDocument,
    PDFProcessingError,
    EncryptedPDFError,
    MalformedPDFError,
    OversizedPDFError,
    ScannedOrEmptyPDFError,
    extract_pdf_content,
)
from gemini_service import GeminiService, PaperAnalysis, check_gemini_connection
from report_generator import generate_markdown_report


def _html_text(value) -> str:
    """Escape untrusted or dynamic text before embedding it in custom HTML."""
    return html.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# High-End Visual Design: Obsidian, Violet & Cyan Glassmorphism
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Reset & Base Typography */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F8FAFC;
    -webkit-font-smoothing: antialiased;
}

/* Master Obsidian Canvas Background */
.stApp {
    background-color: #06080F;
    background-image: 
        radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.10) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(15, 23, 42, 0.8) 0px, transparent 70%);
    background-attachment: fixed;
}

/* Sidebar Refinement */
section[data-testid="stSidebar"] {
    background: #080C16;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* Hide Default Streamlit Menu & Header padding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent !important;}

/* Luxury Brand Wordmark */
.brand-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin-bottom: 1.5rem;
}

.brand-title {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 8px;
}

.brand-pill {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(6, 182, 212, 0.2));
    border: 1px solid rgba(139, 92, 246, 0.4);
    color: #C084FC;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 2px 8px;
    border-radius: 9999px;
}

/* Hero Section Typography */
.hero-wrapper {
    text-align: center;
    padding: 3.5rem 1rem 2rem 1rem;
    max-width: 900px;
    margin: 0 auto;
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.3);
    color: #A78BFA;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 4px 14px;
    border-radius: 9999px;
    margin-bottom: 1.25rem;
}

.hero-headline {
    font-size: 3.4rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1.1;
    background: linear-gradient(135deg, #FFFFFF 20%, #CBD5E1 60%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}

.hero-subheadline {
    font-size: 1.25rem;
    color: #94A3B8;
    line-height: 1.5;
    max-width: 650px;
    margin: 0 auto 2.5rem auto;
    font-weight: 400;
}

/* Glassmorphism Card System */
.apex-glass-card {
    background: rgba(13, 18, 31, 0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 18px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.apex-glass-card:hover {
    border-color: rgba(139, 92, 246, 0.35);
    box-shadow: 0 12px 35px -8px rgba(139, 92, 246, 0.15);
}

/* Document Statistics Grid */
.doc-stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin-top: 1rem;
}

.doc-stat-box {
    background: rgba(6, 8, 15, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 12px 16px;
}

.doc-stat-label {
    font-size: 0.7rem;
    color: #64748B;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.doc-stat-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-top: 2px;
}

/* Research Results Overview */
.overview-banner {
    background: linear-gradient(135deg, rgba(20, 26, 46, 0.8), rgba(11, 15, 25, 0.9));
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.overview-banner::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, #8B5CF6, #06B6D4);
}

.paper-title {
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: 0.75rem;
}

.paper-authors {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    color: #94A3B8;
    font-size: 0.95rem;
}

.author-chip {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 3px 10px;
    border-radius: 9999px;
    color: #E2E8F0;
    font-size: 0.85rem;
    font-weight: 500;
}

/* Section Header & Status Badges */
.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
    display: flex;
    align-items: center;
    gap: 10px;
}

.badge-explicit {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34D399;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.badge-inferred {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #FBBF24;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.evidence-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #38BDF8;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.2);
    padding: 3px 10px;
    border-radius: 6px;
    display: inline-block;
    margin-top: 1rem;
}

/* Luxury 5 Takeaway Cards */
.takeaway-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 1rem;
}

.takeaway-card {
    background: rgba(13, 18, 31, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 1.25rem;
    display: flex;
    gap: 18px;
    align-items: flex-start;
    transition: all 0.2s ease;
}

.takeaway-card:hover {
    border-color: rgba(139, 92, 246, 0.4);
    background: rgba(17, 24, 39, 0.7);
    transform: translateX(3px);
}

.takeaway-num {
    background: linear-gradient(135deg, #8B5CF6, #06B6D4);
    color: #FFFFFF;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 800;
    font-size: 0.85rem;
    padding: 6px 12px;
    border-radius: 10px;
    min-width: 40px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
}

.takeaway-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 4px;
}

.takeaway-desc {
    font-size: 0.92rem;
    color: #94A3B8;
    line-height: 1.5;
}

/* Sidebar Engine Status Card */
.sidebar-status-box {
    background: rgba(13, 18, 31, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 1rem;
}

.status-dot-active {
    display: inline-block;
    width: 8px;
    height: 8px;
    background-color: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 10px #10B981;
    margin-right: 6px;
}

.status-dot-inactive {
    display: inline-block;
    width: 8px;
    height: 8px;
    background-color: #EF4444;
    border-radius: 50%;
    box-shadow: 0 0 10px #EF4444;
    margin-right: 6px;
}

/* Custom Primary Button Styling */
button[kind="primary"] {
    background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.25rem !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.45) !important;
}

/* Custom Secondary Button */
button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    color: #E2E8F0 !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}

button[kind="secondary"]:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
}

/* Footer Styling */
.apex-footer {
    text-align: center;
    padding: 3rem 1rem 1.5rem 1rem;
    font-size: 0.8rem;
    color: #64748B;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 4rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Scientific Orbital Visualization (CSS + SVG)
# ---------------------------------------------------------------------------

SCIENTIFIC_ORBITAL_SVG = """
<div style="display: flex; justify-content: center; margin: 1.5rem 0;">
    <svg width="220" height="120" viewBox="0 0 220 120" fill="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="orbitGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#8B5CF6" stop-opacity="0.8"/>
                <stop offset="100%" stop-color="#06B6D4" stop-opacity="0.8"/>
            </linearGradient>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur"/>
                <feComposite in="SourceGraphic" in2="blur" operator="over"/>
            </filter>
        </defs>
        <!-- Center Core Node -->
        <circle cx="110" cy="60" r="14" fill="#0D121F" stroke="url(#orbitGrad)" stroke-width="2.5" filter="url(#glow)"/>
        <circle cx="110" cy="60" r="5" fill="#38BDF8"/>
        <!-- Elliptical Orbits -->
        <ellipse cx="110" cy="60" rx="60" ry="24" stroke="rgba(139, 92, 246, 0.35)" stroke-width="1.2" stroke-dasharray="3 3"/>
        <ellipse cx="110" cy="60" rx="95" ry="38" stroke="rgba(6, 182, 212, 0.3)" stroke-width="1.2"/>
        <!-- Orbital Satellites -->
        <circle cx="160" cy="48" r="4.5" fill="#C084FC" filter="url(#glow)"/>
        <circle cx="35" cy="72" r="3.5" fill="#22D3EE" filter="url(#glow)"/>
        <circle cx="110" cy="22" r="3" fill="#A78BFA"/>
        <circle cx="190" cy="74" r="2.5" fill="#38BDF8"/>
    </svg>
</div>
"""


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "extracted_doc" not in st.session_state:
    st.session_state.extracted_doc = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "active_filename" not in st.session_state:
    st.session_state.active_filename = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None


# ---------------------------------------------------------------------------
# Sidebar (Clean, No API Key Inputs, Only Server Status & Guardrails)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=180, show_spinner=False)
def get_verified_api_status():
    """Verify live API connectivity with an actual lightweight generation request."""
    return check_gemini_connection()

with st.sidebar:
    st.markdown("""
        <div class="brand-container">
            <div class="brand-title">
                <span>🔬</span> PAPERWISE <span style="color: #8B5CF6;">AI</span>
            </div>
            <div class="brand-pill">APEX v1.0</div>
        </div>
    """, unsafe_allow_html=True)

    # Server-Side Engine Status (Live generation verification, no keys exposed)
    api_ready, api_status_label = get_verified_api_status()
    if api_ready:
        st.markdown(f"""
            <div class="sidebar-status-box">
                <div style="font-size: 0.72rem; color: #64748B; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em;">System Status</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-top: 4px; display: flex; align-items: center;">
                    <span class="status-dot-active"></span> {_html_text(api_status_label)}
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">
                    Configured: <strong style="color: #38BDF8;">{_html_text(config.model_name)}</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="sidebar-status-box" style="border-color: rgba(239, 68, 68, 0.3);">
                <div style="font-size: 0.72rem; color: #EF4444; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em;">System Status</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #FCA5A5; margin-top: 4px; display: flex; align-items: center;">
                    <span class="status-dot-inactive"></span> {_html_text(api_status_label)}
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">
                    Configure <code>GEMINI_API_KEY</code> in server <code>.env</code> file.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Document Boundaries
    st.markdown("### ⚙️ Document Boundaries")
    st.markdown(f"""
    - **Max File Size:** `{config.max_file_size_mb} MB`
    - **Max Pages:** `{config.max_page_count} pages`
    - **Readability Threshold:** `{config.min_word_count_threshold} words`
    - **Synthesis Mode:** `> {config.synthesis_page_threshold} pages`
    """)

    st.markdown("---")

    # Reset Workspace Button
    if st.button("🔄 Reset Studio Workspace", use_container_width=True):
        st.session_state.uploader_key += 1
        st.session_state.extracted_doc = None
        st.session_state.analysis_result = None
        st.session_state.active_filename = None
        st.session_state.error_message = None
        st.rerun()

    st.markdown("---")
    st.caption("PAPERWISE AI APEX • AI Research Intelligence Workstation")


# ---------------------------------------------------------------------------
# Section A: Immersive Home Screen Hero
# ---------------------------------------------------------------------------

st.markdown(f"""
    <div class="hero-wrapper">
        <div class="hero-pill">⚡ Advanced Academic Synthesis & Grounding</div>
        <div class="hero-headline">{_html_text(config.tagline)}</div>
        <div class="hero-subheadline">{_html_text(config.subtagline)}</div>
    </div>
""", unsafe_allow_html=True)

st.markdown(SCIENTIFIC_ORBITAL_SVG, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section B: PDF Upload & Sample Document Action
# ---------------------------------------------------------------------------

col_upload, col_sample = st.columns([5, 3], gap="medium")

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Academic Research Paper (PDF)",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploader_key}",
        help=f"Accepts searchable digital PDF research papers up to {config.max_file_size_mb} MB.",
        label_visibility="collapsed",
    )

with col_sample:
    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        sample_btn = st.button("📄 Sample Paper", use_container_width=True, help="Load pre-validated 3-page CS academic paper.")
    with c_s2:
        arxiv_btn = st.button("📜 Genuine arXiv", use_container_width=True, help="Load 15-page genuine peer-reviewed paper: Attention Is All You Need (Vaswani et al.).")

# Handle Sample Paper Loading
if sample_btn:
    sample_pdf_path = os.path.join(os.path.dirname(__file__), "sample_papers", "sample_research_paper.pdf")
    if os.path.exists(sample_pdf_path):
        with open(sample_pdf_path, "rb") as f:
            sample_bytes = f.read()
        try:
            with st.spinner("Loading verified sample research paper..."):
                extracted = extract_pdf_content(sample_bytes, filename="sample_research_paper.pdf")
                st.session_state.uploader_key += 1
                st.session_state.extracted_doc = extracted
                st.session_state.active_filename = "sample_research_paper.pdf"
                st.session_state.analysis_result = None
                st.session_state.error_message = None
                st.rerun()
        except Exception as e:
            st.session_state.error_message = f"Failed to load sample: {str(e)}"
    else:
        st.session_state.error_message = "Sample research paper file not found."

# Handle Genuine arXiv Paper Loading
if arxiv_btn:
    arxiv_pdf_path = os.path.join(os.path.dirname(__file__), "sample_papers", "genuine_research_paper.pdf")
    if os.path.exists(arxiv_pdf_path):
        with open(arxiv_pdf_path, "rb") as f:
            arxiv_bytes = f.read()
        try:
            with st.spinner("Loading genuine arXiv paper (Attention Is All You Need)..."):
                extracted = extract_pdf_content(arxiv_bytes, filename="genuine_research_paper.pdf")
                st.session_state.uploader_key += 1
                st.session_state.extracted_doc = extracted
                st.session_state.active_filename = "genuine_research_paper.pdf"
                st.session_state.analysis_result = None
                st.session_state.error_message = None
                st.rerun()
        except Exception as e:
            st.session_state.error_message = f"Failed to load arXiv paper: {str(e)}"
    else:
        st.session_state.error_message = "Genuine research paper file not found."

# Handle Uploaded PDF
if uploaded_file is not None and uploaded_file.name != st.session_state.active_filename:
    st.session_state.active_filename = uploaded_file.name
    st.session_state.analysis_result = None
    st.session_state.error_message = None

    try:
        with st.spinner("Verifying digital text structure and extracting pages..."):
            file_bytes = uploaded_file.read()
            extracted = extract_pdf_content(file_bytes, filename=uploaded_file.name)
            st.session_state.extracted_doc = extracted
    except EncryptedPDFError as e:
        st.session_state.extracted_doc = None
        st.session_state.error_message = f"🔒 Password Protected: {str(e)}"
    except ScannedOrEmptyPDFError as e:
        st.session_state.extracted_doc = None
        st.session_state.error_message = f"📷 Scanned / Unreadable Document: {str(e)}"
    except (MalformedPDFError, OversizedPDFError) as e:
        st.session_state.extracted_doc = None
        st.session_state.error_message = f"⚠️ Document Boundary Issue: {str(e)}"
    except Exception as e:
        st.session_state.extracted_doc = None
        st.session_state.error_message = f"❌ Unexpected Extraction Error: {str(e)}"

# Show Error if present
if st.session_state.error_message:
    st.error(st.session_state.error_message)


# ---------------------------------------------------------------------------
# Section C: Document Workspace & Analysis Trigger
# ---------------------------------------------------------------------------

doc: Optional[ExtractedDocument] = st.session_state.extracted_doc

if doc:
    est_reading_time = max(1, round(doc.total_words / 200))
    doc_size_kb = round(doc.total_chars / 1024, 1)

    st.markdown(f"""
        <div class="apex-glass-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                <div>
                    <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Active Document Workspace</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin-top: 4px;">{_html_text(doc.filename)}</div>
                </div>
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #34D399; font-size: 0.75rem; font-weight: 700; padding: 4px 12px; border-radius: 9999px;">
                    ✓ Digital Text Grounded
                </div>
            </div>
            <div class="doc-stat-grid">
                <div class="doc-stat-box">
                    <div class="doc-stat-label">Total Pages</div>
                    <div class="doc-stat-value">{doc.total_pages}</div>
                </div>
                <div class="doc-stat-box">
                    <div class="doc-stat-label">Word Count</div>
                    <div class="doc-stat-value">{doc.total_words:,}</div>
                </div>
                <div class="doc-stat-box">
                    <div class="doc-stat-label">Estimated Size</div>
                    <div class="doc-stat-value">~{doc_size_kb} KB</div>
                </div>
                <div class="doc-stat-box">
                    <div class="doc-stat-label">Human Read Time</div>
                    <div class="doc-stat-value">~{est_reading_time} min</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    for warning in doc.warnings:
        st.warning(warning)

    # Action Controls
    col_cta, col_remove = st.columns([3, 1], gap="medium")
    with col_cta:
        analyze_clicked = st.button(
            f"⚡ Synthesize Research Intelligence ({_html_text(config.model_display_name)})",
            type="primary",
            use_container_width=True,
            disabled=not is_api_configured(),
        )

    with col_remove:
        if st.button("✕ Remove Paper", use_container_width=True, type="secondary"):
            st.session_state.uploader_key += 1
            st.session_state.extracted_doc = None
            st.session_state.analysis_result = None
            st.session_state.active_filename = None
            st.session_state.error_message = None
            st.rerun()

    if not is_api_configured():
        st.info("💡 Server Setup: Configure `GEMINI_API_KEY` in your `.env` file to enable one-click research synthesis.")

    # Synthesis Execution Flow (Real Progressive Stages)
    if analyze_clicked:
        progress_area = st.empty()
        with progress_area.container():
            st.markdown("""
                <div class="apex-glass-card" style="text-align: center; border-color: rgba(139, 92, 246, 0.4);">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem;">
                        Synthesizing Research Intelligence
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8;">
                        Extracting core methodology, experimental datasets, findings and 5 key takeaways...
                    </div>
                </div>
            """, unsafe_allow_html=True)
            stage_bar = st.progress(15)

        try:
            service = GeminiService()
            stage_bar.progress(40)
            analysis = service.analyze_paper(doc)
            stage_bar.progress(100)
            st.session_state.analysis_result = analysis
            progress_area.empty()
            st.toast("Research Intelligence Synthesis Complete", icon="✨")
        except Exception as e:
            progress_area.empty()
            st.error(f"❌ Synthesis error: {str(e)}")


# ---------------------------------------------------------------------------
# Section D: Research Results Workspace (The Visual Highlight)
# ---------------------------------------------------------------------------

analysis: Optional[PaperAnalysis] = st.session_state.analysis_result

if analysis and doc:
    st.markdown("---")

    # Overview Banner
    actual_engine = getattr(analysis, "model_used", None) or config.model_name
    is_fallback = getattr(analysis, "is_fallback_used", False) or (
        actual_engine and actual_engine != config.model_name and not str(actual_engine).startswith("Omitted")
    )
    stage_models = getattr(analysis, "stage_models", {})
    fallback_badge = f'<span style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.35); color: #FBBF24; padding: 2px 8px; border-radius: 9999px; font-weight: 700; margin-left: 8px;">⚠️ Fallback Active</span>' if is_fallback else ''
    authors_chips = "".join(f'<span class="author-chip">{_html_text(a)}</span>' for a in analysis.authors) if analysis.authors else '<span class="author-chip">Not mentioned in paper</span>'

    st.markdown(f"""
        <div class="overview-banner">
            <div style="font-size: 0.75rem; color: #A78BFA; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
                Executive Research Intelligence Overview
            </div>
            <div class="paper-title">{_html_text(analysis.paper_title)}</div>
            <div class="paper-authors">
                <span style="font-weight: 600; color: #CBD5E1;">Authors:</span>
                {authors_chips}
            </div>
            <div style="display: flex; gap: 12px; margin-top: 1.25rem; font-size: 0.78rem; color: #64748B; flex-wrap: wrap; align-items: center;">
                <div>Engine: <strong style="color: #38BDF8;">{_html_text(actual_engine)}</strong>{fallback_badge}</div>
                <div>•</div>
                <div>Coverage: <strong style="color: #34D399;">Evidence-Grounded</strong></div>
                <div>•</div>
                <div>Pages: <strong style="color: #E2E8F0;">{doc.total_pages}</strong></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if stage_models and len(stage_models) > 1:
        with st.expander("🔍 Multi-Stage Pipeline & Model Attribution Breakdown", expanded=False):
            st.caption("Document processing was partitioned across stages. Each entry shows the model used, or indicates when a stage could not be processed.")
            cols = st.columns(min(len(stage_models), 4))
            for idx, (stg_name, mdl) in enumerate(stage_models.items()):
                with cols[idx % len(cols)]:
                    status_col = "#64748B" if str(mdl).startswith("Omitted") else ("#38BDF8" if mdl == config.model_name else "#FBBF24")
                    st.markdown(f"""
                        <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;">
                            <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase;">{_html_text(stg_name)}</div>
                            <div style="font-size: 0.82rem; font-weight: 700; color: {status_col}; font-family: monospace;">{_html_text(mdl)}</div>
                        </div>
                    """, unsafe_allow_html=True)

    # 5 Core Research Analysis Sections
    st.markdown("""
        <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin-bottom: 1rem;">
            Core Research Dimensions
        </div>
    """, unsafe_allow_html=True)

    tab_prob, tab_meth, tab_data, tab_res, tab_lim = st.tabs([
        "🎯 1. Problem Statement",
        "🔬 2. Methodology",
        "📊 3. Dataset & Setup",
        "📈 4. Key Results",
        "⚠️ 5. Limitations & Boundaries",
    ])

    def render_apex_section(tab, title: str, icon: str, section_data):
        with tab:
            badge_class = "badge-explicit" if section_data.is_explicit else "badge-inferred"
            badge_text = "Explicit Author Evidence" if section_data.is_explicit else "Contextual Interpretation"

            st.markdown(f"""
                <div class="apex-glass-card">
                    <div class="section-header">
                        <div class="section-title">
                            <span>{icon}</span> {_html_text(title)}
                        </div>
                        <div class="{badge_class}">{badge_text}</div>
                    </div>
                    <div style="font-size: 0.98rem; color: #E2E8F0; line-height: 1.65; white-space: pre-line;">
                        {_html_text(section_data.content)}
                    </div>
                    <div class="evidence-tag">📍 Evidence Citations: {_html_text(section_data.evidence_pages)}</div>
                </div>
            """, unsafe_allow_html=True)

    render_apex_section(tab_prob, "Problem Addressed", "🎯", analysis.problem)
    render_apex_section(tab_meth, "Methodology & Architecture", "🔬", analysis.method)
    render_apex_section(tab_data, "Dataset & Experimental Setup", "📊", analysis.dataset)
    render_apex_section(tab_res, "Empirical Results & Findings", "📈", analysis.results)
    render_apex_section(tab_lim, "Stated & Inferred Limitations", "⚠️", analysis.limitations)

    # Five Key Takeaway Cards (Numbered 01–05)
    st.markdown("""
        <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin-top: 2rem; margin-bottom: 0.75rem;">
            💡 Five Essential Takeaways
        </div>
    """, unsafe_allow_html=True)

    for item in analysis.takeaways:
        num_str = f"{item.number:02d}"
        st.markdown(f"""
            <div class="takeaway-card">
                <div class="takeaway-num">{num_str}</div>
                <div style="flex: 1;">
                    <div class="takeaway-title">{_html_text(item.title)}</div>
                    <div class="takeaway-desc">{_html_text(item.summary)}</div>
                    <div class="evidence-tag" style="margin-top: 8px;">Source: {_html_text(item.page_reference)}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Academic Integrity & Completeness Notes
    if analysis.disclosure_notes and analysis.disclosure_notes.strip():
        with st.expander("🔍 Academic Integrity & Completeness Disclosures", expanded=False):
            st.info(analysis.disclosure_notes)

    # Section E: Report Export
    st.markdown("---")
    st.markdown("""
        <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin-bottom: 0.5rem;">
            Export Research Dossier
        </div>
        <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 1rem;">
            Download the complete, publication-grade intelligence report in Markdown format for offline review or assignment submission.
        </div>
    """, unsafe_allow_html=True)

    col_dl, col_blank = st.columns([1, 2])
    with col_dl:
        markdown_report = generate_markdown_report(analysis, doc, model_name=actual_engine)
        clean_name = "".join(c for c in analysis.paper_title[:28] if c.isalnum() or c in (' ', '_', '-')).strip()
        clean_name = clean_name.replace(" ", "_").lower() or "research_intelligence"
        clean_name = re.sub(r"_+", "_", clean_name)

        st.download_button(
            label="📥 Download Full Intelligence Report (.md)",
            data=markdown_report,
            file_name=f"{clean_name}_apex_report.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary",
        )


# ---------------------------------------------------------------------------
# Empty Landing State
# ---------------------------------------------------------------------------

elif not doc and not st.session_state.error_message:
    st.markdown("""
        <div style="text-align: center; padding: 2.5rem 1rem; border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 18px; margin-top: 1rem; background: rgba(13, 18, 31, 0.3);">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📄</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC;">Upload an academic research paper or try our sample</div>
            <div style="font-size: 0.85rem; color: #64748B; max-width: 480px; margin: 0.4rem auto 1rem auto;">
                Drop any PDF from arXiv, IEEE, ACM, Nature, or click "Try Sample Paper" above to experience instant research intelligence.
            </div>
            <div style="display: inline-flex; gap: 20px; font-size: 0.78rem; color: #94A3B8;">
                <div>🔒 Server-side zero leakage</div>
                <div>⚡ Citation-Linked Synthesis</div>
                <div>📍 Page-level citations</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(f"""
    <div class="apex-footer">
        <strong>{_html_text(config.app_name)}</strong> • Powered by <strong>Google {_html_text(config.model_display_name)}</strong> & <strong>PyPDF</strong><br>
        Built for academic research acceleration with strict verification.
    </div>
""", unsafe_allow_html=True)
